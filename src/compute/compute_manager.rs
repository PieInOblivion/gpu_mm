use std::sync::Arc;

use crate::{
    dataloader::error::VKMLEngineError, gpu::{gpu_memory::GPUMemory, vk_gpu::GPU}, model::{graph_model::GraphModel, layer_desc::LayerDesc, layer_type::LayerType, tensor_desc::TensorDesc, weight_init::WeightInit}, thread_pool::thread_pool::ThreadPool};

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
    pub weight_gradients: ComputeTensor,
    pub bias_gradients: ComputeTensor,
    pub activation_gradients: Option<ComputeTensor>, // Optional only for input buffer
}

pub struct ComputeManager {
    pub gpus: Vec<GPU>,
    upto_gpu: usize,
    pub cpu: CPUCompute,
    thread_pool: Arc<ThreadPool>,

    pub model: GraphModel,
}

impl ComputeManager {
    pub fn new(model: GraphModel, thread_pool: Arc<ThreadPool>) -> Result<Self, VKMLEngineError> {
        let gpus = Self::available_gpus()?;
        Self::new_with(model, thread_pool, gpus, None)
    }

    pub fn new_with(model: GraphModel, thread_pool: Arc<ThreadPool>, gpus: Vec<GPU>, cpu_memory_limit_bytes: Option<u64>) -> Result<Self, VKMLEngineError> {
        if model.verified.is_none() {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("GraphModel has not been verified. Use .verify()")
            ));
        }

        let cpu = CPUCompute::new(cpu_memory_limit_bytes, thread_pool.clone());

        let total_memory = model.total_memory_requirements();
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
            gpus,
            upto_gpu: 0,
            cpu,
            thread_pool,
            model,
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

    fn allocate_layer(
        &mut self,
        mut layer_desc: LayerDesc,
        input_shapes: Vec<&TensorDesc>,
        weight_init: &WeightInit,
        target_device: &DeviceLocation
    ) -> Result<ComputeLayer, VKMLEngineError> {
        // Calculate output shape using model's batch size, create tensor descriptor for activations
        let activation_desc = if input_shapes.is_empty() {
            // This is an input layer
            layer_desc.output_shape(self.model.batch_size, None)?
        } else if input_shapes.len() == 1 {
            // Standard single-input layer
            layer_desc.output_shape(self.model.batch_size, Some(input_shapes[0]))?
        } else {
            // Multi-input layer - needs custom handling based on layer type
            // For now, just use the first input as a simplification
            layer_desc.output_shape(self.model.batch_size, Some(input_shapes[0]))?
        };

        // Update activation gradients parameter to match the activation size for non-parameter layers
        if !layer_desc.requires_parameters && layer_desc.layer_type.requires_gradients() {
            layer_desc.activation_gradients = Some(activation_desc.clone());
        }
        
        let (weights, biases, weight_gradients, bias_gradients) = if layer_desc.requires_parameters {
            (
                self.allocate_tensor(&layer_desc.weights, target_device, weight_init)?,
                self.allocate_tensor(&layer_desc.biases, target_device, weight_init)?,
                self.allocate_tensor(&layer_desc.weight_gradients, target_device, &WeightInit::Constant(0.0))?,
                self.allocate_tensor(&layer_desc.bias_gradients, target_device, &WeightInit::Constant(0.0))?
            )
        } else {
            (
                ComputeTensor {
                    desc: layer_desc.weights.clone(),
                    location: ComputeLocation::Parameterless,
                },
                ComputeTensor {
                    desc: layer_desc.biases.clone(),
                    location: ComputeLocation::Parameterless,
                },
                ComputeTensor {
                    desc: layer_desc.biases.clone(),
                    location: ComputeLocation::Parameterless,
                },
                ComputeTensor {
                    desc: layer_desc.biases.clone(),
                    location: ComputeLocation::Parameterless,
                }
            )
        };

        // Allocate the activation tensor on target device
        let activations = self.allocate_tensor(&activation_desc, target_device, &WeightInit::Constant(0.0))?;

        // Only allocate activation gradients if the layer needs them
        let activation_gradients = if let Some(grad_desc) = &layer_desc.activation_gradients {
            Some(self.allocate_tensor(grad_desc, target_device, &WeightInit::Constant(0.0))?)
        } else {
            None
        };

        Ok(ComputeLayer {
            desc: layer_desc,
            weights,
            biases,
            activations,
            weight_gradients,
            bias_gradients,
            activation_gradients
        })
    }

    fn allocate_tensor(&mut self, desc: &TensorDesc, target_device: &DeviceLocation, weight_init: &WeightInit) 
        -> Result<ComputeTensor, VKMLEngineError> {
        // Allocate memory on target device
        let size_in_bytes = desc.size_in_bytes() as u64;

        // NOTE: There is probably a better place to put this
        // The most optimal value depends on each machine. This will serve as a general value for now
        // Constant type weight_init probably never needs to be multi-threaded?
        let parallel_threshold = 10000;

        let total_elements = desc.num_elements();
        
        let initial_data = {
            if total_elements < parallel_threshold {
                weight_init.init(desc, total_elements)
            } else {
                weight_init.par_init(desc, total_elements, parallel_threshold, self.thread_pool.clone())
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
        let size = tensor.desc.size_in_bytes() as u64;
    
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
                total_size_needed += layer.weights.desc.size_in_bytes() as u64;
            }
            
            if let ComputeLocation::GPU { .. } = layer.biases.location {
                total_size_needed += layer.biases.desc.size_in_bytes() as u64;
            }
            
            if let ComputeLocation::GPU { .. } = layer.activations.location {
                total_size_needed += layer.activations.desc.size_in_bytes() as u64;
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
                let size = tensor.desc.size_in_bytes() as u64;
                self.cpu.memory_tracking.allocate(size)?;
                
                // Free GPU memory
                self.gpus[gpu_idx].deallocate_memory(size);
                
                // Update tensor location
                tensor.location = ComputeLocation::CPU(data);
            }
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
