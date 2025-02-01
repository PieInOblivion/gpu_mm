use std::sync::Arc;

use crate::{
    dataloader::error::VKMLEngineError, gpu::vk_gpu::{GPUMemory, GPU}, model::{layer::LayerDesc, layer_params::LayerType, model::ModelDesc, tensor::TensorDesc}, thread_pool::thread_pool::ThreadPool};

use super::cpu_compute::CPUCompute;

pub enum DeviceLocation {
    CPU,
    GPU(usize)
}

pub enum ComputeLocation {
    CPU(Vec<f32>),
    GPU {
        gpu_idx: usize,
        memory: GPUMemory,
    },
    Parameterless
}

pub struct ComputeTensor {
    pub desc: TensorDesc,
    pub location: ComputeLocation,
}

pub struct ComputeLayer {
    pub desc: LayerDesc,
    pub weights: ComputeTensor,
    pub biases: ComputeTensor,
    pub activations: ComputeTensor,
}

pub struct ComputeManager {
    model_desc: ModelDesc,
    gpus: Vec<GPU>,
    upto_gpu: usize,
    cpu: CPUCompute,
    thread_pool: Arc<ThreadPool>,
    pub layers: Vec<ComputeLayer>,
}

impl ComputeManager {
    pub fn new(desc: ModelDesc, thread_pool: Arc<ThreadPool>) -> Result<Self, VKMLEngineError> {
        let gpus = Self::available_gpus()?;
        Self::new_with(desc, gpus, None, thread_pool)
    }

    pub fn new_with(desc: ModelDesc, gpus: Vec<GPU>, cpu_memory_limit_bytes: Option<u64>, thread_pool: Arc<ThreadPool>) -> Result<Self, VKMLEngineError> {
        let cpu = CPUCompute::new(cpu_memory_limit_bytes, thread_pool.clone());

        let total_memory = desc.total_memory_requirements();
        let total_available: u64 = gpus.iter()
            .map(|gpu| gpu.available_memory())
            .sum::<u64>()
            + cpu.memory_tracking.get_available();

        // This will not be the most accurate as it doesn't account for any allocation overhead
        // but should be close enough for this concept
        if total_memory > total_available {
            return Err(VKMLEngineError::OutOfMemory(
                format!("Model requires {} bytes but only {} available", 
                    total_memory, total_available)
            ));
        }
        
        let mut manager = Self {
            model_desc: desc.clone(),
            gpus,
            upto_gpu: 0,
            cpu,
            thread_pool,
            layers: Vec::with_capacity(desc.layers.len()),
        };

        // Allocate layers sequentially
        for layer_desc in desc.layers.into_iter() {
            let layer_memory = layer_desc.memory_requirements();
            let target_device = manager.find_optimal_device(layer_memory)?;
            let compute_layer = manager.allocate_layer(layer_desc, &target_device)?;
            manager.layers.push(compute_layer);
        }

        Ok(manager)
    }

    pub fn available_gpus() -> Result<Vec<GPU>, VKMLEngineError> {
        let gpu_info = GPU::available_gpus()?;
        let mut gpus = Vec::with_capacity(gpu_info.len());
        
        for info in gpu_info {
            if let Ok(gpu) = GPU::new(info.device_index) {
                gpus.push(gpu);
            }
        }
        
        Ok(gpus)
    }

    fn find_optimal_device(&mut self, size: u64) -> Result<DeviceLocation, VKMLEngineError> {
        // Only check GPUs starting from our current one to maintain sequential blocks
        for idx in self.upto_gpu..self.gpus.len() {
            let gpu = &self.gpus[idx];
            if size <= gpu.available_memory() {
                self.upto_gpu = idx;
                return Ok(DeviceLocation::GPU(idx));
            }
        }

        // If no GPU has enough memory, try CPU
        if self.cpu.memory_tracking.get_available() >= size {
            // Rust creates empy range when the start is equal or greater than end
            // Has one downside that if someone runs code with 18446744073709551615 GPUS on 64bit systems or 4294967295 on 32bit systems it will skip their last GPU :(
            self.upto_gpu = usize::MAX;
            Ok(DeviceLocation::CPU)
        } else {
            Err(VKMLEngineError::OutOfMemory("No device has enough memory".into()))
        }
    }

    fn allocate_layer(&mut self, layer_desc: LayerDesc, target_device: &DeviceLocation) -> Result<ComputeLayer, VKMLEngineError> {
        let (weights, biases) = if layer_desc.requires_parameters {
            (
                self.allocate_tensor(&layer_desc.weights, target_device)?,
                self.allocate_tensor(&layer_desc.biases, target_device)?
            )
        } else {
            // Create tensors with None location for non-parameter layers
            (
                ComputeTensor {
                    desc: layer_desc.weights.clone(),
                    location: ComputeLocation::Parameterless,
                },
                ComputeTensor {
                    desc: layer_desc.biases.clone(),
                    location: ComputeLocation::Parameterless,
                }
            )
        };

        // Get input shape from previous layer if it exists
        let input_shape = self.layers.last()
            .map(|layer| layer.activations.desc.shape.as_slice());
    
        // Calculate output shape using model's batch size
        let output_shape = layer_desc.output_shape(self.model_desc.batch_size, input_shape)?;
    
        // Create tensor descriptor for activations
        let activation_desc = TensorDesc::new(output_shape);
        // Allocate the activation tensor on target device
        let activations = self.allocate_tensor(&activation_desc, target_device)?;
    
        Ok(ComputeLayer {
            desc: layer_desc,
            weights,
            biases,
            activations,
        })
    }

    fn allocate_tensor(&mut self, desc: &TensorDesc, target_device: &DeviceLocation) 
        -> Result<ComputeTensor, VKMLEngineError> {
        // Allocate memory on target device
        let size_in_bytes = desc.size() as u64;

        // NOTE: There is probably a better place to put this
        // The most optimal value depends on each machine. This will serve as a general value for now
        let parallel_threshold = 10000;

        let total_elements = desc.shape.iter().product();
        
        let initial_data = {
            if total_elements < parallel_threshold {
                self.model_desc.weight_init.init(&desc.shape, total_elements)
            } else {
                self.model_desc.weight_init.par_init(&desc.shape, total_elements, parallel_threshold, self.thread_pool.clone())
            }
        };

        let location = match target_device {
            DeviceLocation::CPU => {
                // Allocate CPU memory
                self.cpu.memory_tracking.allocate(size_in_bytes)?;
                ComputeLocation::CPU(initial_data)
            },
            DeviceLocation::GPU(idx) => {
                // Allocate GPU memory
                let gpu = &mut self.gpus[*idx];
                gpu.allocate_memory(size_in_bytes)?;
                
                let gpu_memory = gpu.move_to_gpu_as_f32(&initial_data)
                    .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                
                ComputeLocation::GPU {
                    gpu_idx: *idx,
                    memory: gpu_memory,
                }
            }
        };

        Ok(ComputeTensor {
            desc: desc.clone(),
            location,
        })
    }

    fn move_tensor_to_device(
        &self,
        tensor: &mut ComputeTensor,
        target_device: &DeviceLocation
    ) -> Result<(), VKMLEngineError> {
        // If tensor is Parameterless, no need to move anything
        if let ComputeLocation::Parameterless = tensor.location {
            return Ok(());
        }
    
        // If already on target device, nothing to do
        match (&tensor.location, target_device) {
            (ComputeLocation::CPU(_), DeviceLocation::CPU) => return Ok(()),
            (ComputeLocation::GPU { gpu_idx, .. }, DeviceLocation::GPU(target_idx))
                if gpu_idx == target_idx => return Ok(()),
            _ => {}
        }

        // Get size for memory tracking
        let size = tensor.desc.size() as u64;
    
        // Allocate on new device first
        match target_device {
            DeviceLocation::CPU => {
                self.cpu.memory_tracking.allocate(size)?;
            },
            DeviceLocation::GPU(idx) => {
                self.gpus[*idx].allocate_memory(size)?;
            }
        }
    
        // Perform the move
        let new_location = match &tensor.location {
            ComputeLocation::CPU(data) => {
                match target_device {
                    DeviceLocation::CPU => unreachable!(), // Handled above
                    DeviceLocation::GPU(idx) => {
                        // CPU to GPU
                        let gpu = &self.gpus[*idx];
                        let gpu_mem = gpu.move_to_gpu_as_f32(data)
                            .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
    
                        ComputeLocation::GPU {
                            gpu_idx: *idx,
                            memory: gpu_mem,
                        }
                    }
                }
            },
            ComputeLocation::GPU { gpu_idx, memory } => {
                match target_device {
                    DeviceLocation::CPU => {
                        // GPU to CPU
                        let gpu = &self.gpus[*gpu_idx];
                        let cpu_data = gpu.read_memory(memory)
                            .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                        
                        ComputeLocation::CPU(cpu_data)
                    },
                    DeviceLocation::GPU(target_idx) => {
                        // GPU to different GPU
                        let source_gpu = &self.gpus[*gpu_idx];
                        let cpu_data = source_gpu.read_memory(memory)
                            .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                        
                        let target_gpu = &self.gpus[*target_idx];
                        let new_gpu_mem = target_gpu.move_to_gpu_as_f32(&cpu_data)
                            .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                        
                        ComputeLocation::GPU {
                            gpu_idx: *target_idx,
                            memory: new_gpu_mem,
                        }
                    }
                }
            },
            ComputeLocation::Parameterless => unreachable!(),
        };
    
        // Deallocate from old device
        match &tensor.location {
            ComputeLocation::CPU(_) => {
                self.cpu.memory_tracking.deallocate(size);
            },
            ComputeLocation::GPU { gpu_idx, .. } => {
                self.gpus[*gpu_idx].deallocate_memory(size);
            },
            ComputeLocation::Parameterless => unreachable!(),
        }
    
        // Update tensor location
        tensor.location = new_location;
        Ok(())
    }

    pub fn move_model_to_cpu(&mut self) -> Result<(), VKMLEngineError> {
        // First check if we have enough CPU memory for all GPU tensors
        let mut total_size_needed: u64 = 0;
        
        // Calculate total memory needed from GPU tensors
        for layer in &self.layers {
            if let ComputeLocation::GPU { .. } = layer.weights.location {
                total_size_needed += layer.weights.desc.size() as u64;
            }
            
            if let ComputeLocation::GPU { .. } = layer.biases.location {
                total_size_needed += layer.biases.desc.size() as u64;
            }
            
            if let ComputeLocation::GPU { .. } = layer.activations.location {
                total_size_needed += layer.activations.desc.size() as u64;
            }
        }
        
        // Check if we have enough CPU memory available
        if total_size_needed > self.cpu.memory_tracking.get_available() {
            return Err(VKMLEngineError::OutOfMemory(
                format!("Not enough CPU memory to move GPU tensors: need {} bytes but only {} available",
                    total_size_needed, self.cpu.memory_tracking.get_available())
            ));
        }

        // Create a list of moves we need to perform
        let mut moves = Vec::new();
        
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let ComputeLocation::GPU { gpu_idx, .. } = layer.weights.location {
                moves.push((layer_idx, "weights", gpu_idx));
            }
            if let ComputeLocation::GPU { gpu_idx, .. } = layer.biases.location {
                moves.push((layer_idx, "biases", gpu_idx));
            }
            if let ComputeLocation::GPU { gpu_idx, .. } = layer.activations.location {
                moves.push((layer_idx, "activations", gpu_idx));
            }
        }
        
        // Now perform all the moves
        for (layer_idx, tensor_type, gpu_idx) in moves {
            // Get mutable references to the specific tensors we want to move
            let tensor = match tensor_type {
                "weights" => &mut self.layers[layer_idx].weights,
                "biases" => &mut self.layers[layer_idx].biases,
                "activations" => &mut self.layers[layer_idx].activations,
                _ => unreachable!(),
            };
            
            // Perform the actual move
            if let ComputeLocation::GPU { ref memory, .. } = tensor.location {
                // Read the data from GPU
                let data = self.gpus[gpu_idx].read_memory(memory)
                    .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                
                // Allocate CPU memory
                let size = tensor.desc.size() as u64;
                self.cpu.memory_tracking.allocate(size)?;
                
                // Free GPU memory
                self.gpus[gpu_idx].deallocate_memory(size);
                
                // Update tensor location
                tensor.location = ComputeLocation::CPU(data);
            }
        }
        
        Ok(())
    }

    pub fn print_model_stats(&self) {
        let mut total_params = 0usize;
        let mut total_memory = 0u64;
        
        println!("\nModel Statistics");
        println!("================");
        println!("\nBatch Size: {}", self.model_desc.batch_size);
        println!("\nLayer Details:");
        println!("{:-<90}", "");
        println!("{:<4} {:<20} {:<15} {:<15} {:<15} {:<15}", 
            "No.", "Type", "Parameters", "Memory (MB)", "Output Shape", "Device");
        println!("{:-<90}", "");

        for (i, layer) in self.layers.iter().enumerate() {
            let params = match &layer.desc.layer_type {
                LayerType::Linear(params) => {
                    // weights: out_features * in_features, biases: out_features
                    params.in_features * params.out_features + params.out_features
                },
                LayerType::Conv2D(params) => {
                    // weights: out_channels * in_channels * kernel_size.0 * kernel_size.1
                    // biases: out_channels
                    params.out_features * params.in_features * params.kernel_h.unwrap() * params.kernel_w.unwrap() + params.out_features
                },
                LayerType::ReLU |
                LayerType::LeakyReLU { .. } |
                LayerType::Sigmoid |
                LayerType::Softmax { .. } |
                LayerType::Tanh |
                LayerType::GELU |
                LayerType::SiLU => 0,
            };

            let memory_bytes = layer.desc.memory_requirements();
            let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
            
            // Get output shape from activations tensor
            let output_shape = layer.activations.desc.shape
                .iter()
                .map(|&d| d.to_string())
                .collect::<Vec<_>>()
                .join("×");

            let device_location = match &layer.activations.location {
                ComputeLocation::CPU(_) => "CPU",
                ComputeLocation::GPU { gpu_idx, .. } => &format!("GPU {}", gpu_idx),
                ComputeLocation::Parameterless => "N/A",
            };

            let layer_desc = match &layer.desc.layer_type {
                LayerType::Linear(params) => 
                    format!("Linear({}, {})", params.in_features, params.out_features),
                LayerType::Conv2D(params) => {
                    let k_h = params.kernel_h.unwrap_or(3);
                    let k_w = params.kernel_w.unwrap_or(3);
                    let s_h = params.stride_h.unwrap_or(1);
                    let s_w = params.stride_w.unwrap_or(1);
                    let p_h = params.padding_h.unwrap_or(1);
                    let p_w = params.padding_w.unwrap_or(1);
                    format!(
                        "Conv2D({}, {}, {}×{}, s={:?}, p={:?})", 
                        params.in_features, params.out_features, 
                        k_h, k_w,
                        (s_h, s_w), (p_h, p_w)
                    )
                },
                LayerType::ReLU => "ReLU".to_string(),
                LayerType::LeakyReLU(alpha) => format!("LeakyReLU(α={})", alpha),
                LayerType::Sigmoid => "Sigmoid".to_string(),
                LayerType::Softmax(dim) => format!("Softmax(dim={})", dim),
                LayerType::Tanh => "Tanh".to_string(),
                LayerType::GELU => "GELU".to_string(),
                LayerType::SiLU => "SiLU".to_string(),
            };

            println!("{:<4} {:<20} {:<15} {:<15.2} {:<15} {:<15}", 
                i, layer_desc, params, memory_mb, output_shape, device_location);

            total_params += params;
            total_memory += memory_bytes;
        }

        println!("{:-<90}", "");
        println!("\nModel Summary:");
        println!("Total Parameters: {}", total_params);
        println!("Total Memory: {:.2} MB", total_memory as f64 / (1024.0 * 1024.0));
        
        // Memory allocation status
        println!("\nMemory Allocation:");
        println!("CPU Memory Used: {:.2} MB", 
            self.cpu.memory_tracking.get_current() as f64 / (1024.0 * 1024.0));
        println!("CPU Memory Available: {:.2} MB", 
            self.cpu.memory_tracking.get_available() as f64 / (1024.0 * 1024.0));
        
        for (i, gpu) in self.gpus.iter().enumerate() {
            println!("GPU {} Memory Used: {:.2} MB", 
                i, (gpu.total_memory() - gpu.available_memory()) as f64 / (1024.0 * 1024.0));
            println!("GPU {} Memory Available: {:.2} MB", 
                i, gpu.available_memory() as f64 / (1024.0 * 1024.0));
        }
    }

    pub fn print_layer_values(&self, layer_idx: usize) -> Result<(), VKMLEngineError> {
        if layer_idx >= self.layers.len() {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Layer index {} out of bounds (max {})", layer_idx, self.layers.len() - 1)
            ));
        }
    
        let layer = &self.layers[layer_idx];
        
        // Create detailed layer description
        let layer_desc = match &layer.desc.layer_type {
            LayerType::Linear(params) => 
                format!("Linear({}, {})", params.in_features, params.out_features),
            LayerType::Conv2D(params) => {
                let k_h = params.kernel_h.unwrap_or(3);
                let k_w = params.kernel_w.unwrap_or(3);
                let s_h = params.stride_h.unwrap_or(1);
                let s_w = params.stride_w.unwrap_or(1);
                let p_h = params.padding_h.unwrap_or(1);
                let p_w = params.padding_w.unwrap_or(1);
                format!(
                    "Conv2D({}, {}, {}×{}, s={:?}, p={:?})", 
                    params.in_features, params.out_features, 
                    k_h, k_w,
                    (s_h, s_w), (p_h, p_w)
                )
            },
            LayerType::ReLU => "ReLU".to_string(),
            LayerType::LeakyReLU(alpha) => format!("LeakyReLU(α={})", alpha),
            LayerType::Sigmoid => "Sigmoid".to_string(),
            LayerType::Softmax(dim) => format!("Softmax(dim={})", dim),
            LayerType::Tanh => "Tanh".to_string(),
            LayerType::GELU => "GELU".to_string(),
            LayerType::SiLU => "SiLU".to_string(),
        };
    
        println!("\nLayer {} Values ({})", layer_idx, layer_desc);
        println!("{:-<80}", "");
    
        // Helper closure to format arrays nicely
        let format_array = |arr: &[f32], max_items: usize| {
            let mut s = String::from("[");
            for (i, val) in arr.iter().take(max_items).enumerate() {
                if i > 0 { s.push_str(", "); }
                s.push_str(&format!("{:.6}", val));
            }
            if arr.len() > max_items {
                s.push_str(", ...")
            }
            s.push(']');
            s
        };
    
        // Helper function to print tensor information
        let print_tensor_info = |name: &str, tensor: &ComputeTensor, gpu_idx: Option<usize>, 
                                data: &[f32], shape: &[usize]| {
            println!("\n{}:", name);
            println!("Location: {}", match gpu_idx {
                Some(idx) => format!("GPU {}", idx),
                None => "CPU".to_string(),
            });
            println!("Shape: {:?}", shape);
            println!("Values: {}", format_array(data, 10));
            
            // Print additional statistics for the tensor
            if !data.is_empty() {
                let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mean = data.iter().sum::<f32>() / data.len() as f32;
                println!("Stats: min={:.6}, max={:.6}, mean={:.6}", min_val, max_val, mean);
            }
        };
    
        // Print weights if layer has parameters
        if layer.desc.requires_parameters {
            match &layer.weights.location {
                ComputeLocation::CPU(data) => {
                    print_tensor_info("Weights", &layer.weights, None, data, &layer.weights.desc.shape);
                },
                ComputeLocation::GPU { gpu_idx, memory } => {
                    let data = self.gpus[*gpu_idx].read_memory(memory)
                        .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                    print_tensor_info("Weights", &layer.weights, Some(*gpu_idx), &data, &layer.weights.desc.shape);
                },
                ComputeLocation::Parameterless => {
                    println!("\nWeights: No weights (parameterless layer)");
                },
            }
    
            match &layer.biases.location {
                ComputeLocation::CPU(data) => {
                    print_tensor_info("Biases", &layer.biases, None, data, &layer.biases.desc.shape);
                },
                ComputeLocation::GPU { gpu_idx, memory } => {
                    let data = self.gpus[*gpu_idx].read_memory(memory)
                        .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                    print_tensor_info("Biases", &layer.biases, Some(*gpu_idx), &data, &layer.biases.desc.shape);
                },
                ComputeLocation::Parameterless => {
                    println!("\nBiases: No biases (parameterless layer)");
                },
            }
        } else {
            println!("\nParameters: None (activation layer)");
        }
    
        // Print activations
        match &layer.activations.location {
            ComputeLocation::CPU(data) => {
                print_tensor_info("Activations", &layer.activations, None, data, &layer.activations.desc.shape);
            },
            ComputeLocation::GPU { gpu_idx, memory } => {
                let data = self.gpus[*gpu_idx].read_memory(memory)
                    .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                print_tensor_info("Activations", &layer.activations, Some(*gpu_idx), &data, &layer.activations.desc.shape);
            },
            ComputeLocation::Parameterless => {
                println!("\nActivations: No activations (parameterless layer)");
            },
        }
    
        Ok(())
    }
}

impl Drop for ComputeManager {
    fn drop(&mut self) {
        // Clear layers first to ensure proper GPU memory cleanup
        self.layers.clear();
    }
}
