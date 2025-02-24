use crate::{compute::compute_manager::{ComputeLocation, ComputeTensor}, dataloader::error::VKMLEngineError, model::{graph_model::LayerId, layer_type::LayerType}};

use super::compute_manager::ComputeManager;

pub fn print_model_stats(cm: &ComputeManager) {
    let mut total_params = 0usize;
    let mut total_memory = 0u64;
    
    println!("\nModel Statistics");
    println!("================");
    println!("\nBatch Size: {}", cm.model.batch_size);
    println!("\nLayer Details:");
    println!("{:-<90}", "");
    println!("{:<4} {:<20} {:<15} {:<15} {:<15} {:<15}", 
        "ID", "Type", "Parameters", "Memory (MB)", "Output Shape", "Device");
    println!("{:-<90}", "");

    // Get the execution order if available, otherwise just use the layer IDs
    let layer_ids = if let Some(verified) = &cm.model.verified {
        verified.execution_order.clone()
    } else {
        cm.model.layers.keys().cloned().collect()
    };

    for id in layer_ids {
        if let Some(layer) = cm.model.layers.get(&id) {
            let params = match &layer.layer_desc.layer_type {
                LayerType::InputBuffer { features, .. } => *features,
                LayerType::Linear(params) => {
                    params.in_features * params.out_features + params.out_features
                },
                LayerType::Conv2D(params) => {
                    params.out_features * params.in_features * 
                    params.kernel_h.unwrap_or(3) * params.kernel_w.unwrap_or(3) + 
                    params.out_features
                },
                _ => 0,
            };

            let memory_bytes = layer.layer_desc.memory_requirements();
            let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
            
            // Get output shape from activations tensor if compute_layer exists
            let output_shape = if let Some(compute_layer) = &layer.compute_layer {
                compute_layer.activations.desc.to_dims()
                    .iter()
                    .map(|&d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("×")
            } else {
                "Unallocated".to_string()
            };

            let device_location = if let Some(compute_layer) = &layer.compute_layer {
                match &compute_layer.activations.location {
                    ComputeLocation::CPU(_) => "CPU",
                    ComputeLocation::GPU { gpu_idx, .. } => &format!("GPU {}", gpu_idx),
                    ComputeLocation::Parameterless => "N/A",
                }
            } else {
                "Unallocated"
            };

            let layer_desc = match &layer.layer_desc.layer_type {
                LayerType::InputBuffer { features, track_gradients } => {
                    if *track_gradients {
                        format!("InputBuffer({}, with gradients)", features)
                    } else {
                        format!("InputBuffer({})", features)
                    }
                },
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
                LayerType::Flatten => "Flatten".to_string(),
                LayerType::ReLU => "ReLU".to_string(),
                LayerType::LeakyReLU(alpha) => format!("LeakyReLU(α={})", alpha),
                LayerType::Sigmoid => "Sigmoid".to_string(),
                LayerType::Softmax(dim) => format!("Softmax(dim={})", dim),
                LayerType::Tanh => "Tanh".to_string(),
                LayerType::GELU => "GELU".to_string(),
                LayerType::SiLU => "SiLU".to_string(),
            };

            println!("{:<4} {:<20} {:<15} {:<15.2} {:<15} {:<15}", 
                id, layer_desc, params, memory_mb, output_shape, device_location);

            total_params += params;
            total_memory += memory_bytes;
        }
    }

    println!("{:-<90}", "");
    println!("\nGraph Structure:");
    println!("Entry points: {:?}", cm.model.verified.as_ref().map_or(vec![], |v| v.entry_points.clone()));
    println!("Exit points: {:?}", cm.model.verified.as_ref().map_or(vec![], |v| v.exit_points.clone()));
    
    println!("\nModel Summary:");
    println!("Total Parameters: {}", total_params);
    println!("Total Memory: {:.2} MB", total_memory as f64 / (1024.0 * 1024.0));
    
    // Memory allocation status
    println!("\nMemory Allocation:");
    println!("CPU Memory Used: {:.2} MB", 
        cm.cpu.memory_tracking.get_current() as f64 / (1024.0 * 1024.0));
    println!("CPU Memory Available: {:.2} MB", 
        cm.cpu.memory_tracking.get_available() as f64 / (1024.0 * 1024.0));
    
    for (i, gpu) in cm.gpus.iter().enumerate() {
        println!("GPU {} Memory Used: {:.2} MB", 
            i, (gpu.total_memory() - gpu.available_memory()) as f64 / (1024.0 * 1024.0));
        println!("GPU {} Memory Available: {:.2} MB", 
            i, gpu.available_memory() as f64 / (1024.0 * 1024.0));
    }
}

pub fn print_layer_values(cm: &ComputeManager, layer_id: LayerId) -> Result<(), VKMLEngineError> {
    let layer = cm.model.layers.get(&layer_id).ok_or(
        VKMLEngineError::VulkanLoadError(format!("Layer ID {} not found", layer_id))
    )?;
    
    let compute_layer = layer.compute_layer.as_ref().ok_or(
        VKMLEngineError::VulkanLoadError(format!("Layer ID {} not allocated", layer_id))
    )?;
    
    // Create detailed layer description
    let layer_desc = match &layer.layer_desc.layer_type {
        LayerType::InputBuffer { features, track_gradients } => {
            if *track_gradients {
                format!("InputBuffer({}, with gradients)", features)
            } else {
                format!("InputBuffer({})", features)
            }
        },
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
        LayerType::Flatten => "Flatten".to_string(),
        LayerType::ReLU => "ReLU".to_string(),
        LayerType::LeakyReLU(alpha) => format!("LeakyReLU(α={})", alpha),
        LayerType::Sigmoid => "Sigmoid".to_string(),
        LayerType::Softmax(dim) => format!("Softmax(dim={})", dim),
        LayerType::Tanh => "Tanh".to_string(),
        LayerType::GELU => "GELU".to_string(),
        LayerType::SiLU => "SiLU".to_string(),
    };

    println!("\nLayer {} Values ({})", layer_id, layer_desc);
    println!("{:-<120}", "");
    println!("Input connections: {:?}", layer.input_layers);
    println!("Output connections: {:?}", layer.output_layers);

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

    // Helper function to print tensor statistics
    let print_tensor_stats = |data: &[f32]| {
        if !data.is_empty() {
            let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let variance = data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / data.len() as f32;
            let std_dev = variance.sqrt();
            
            println!("  Stats:");
            println!("    Min: {:.6}", min_val);
            println!("    Max: {:.6}", max_val);
            println!("    Mean: {:.6}", mean);
            println!("    Std Dev: {:.6}", std_dev);
            
            // Count non-zero elements
            let non_zero = data.iter().filter(|&&x| x != 0.0).count();
            println!("    Non-zero elements: {} ({:.2}%)", 
                non_zero, 
                (non_zero as f32 / data.len() as f32) * 100.0
            );
        }
    };

    // Helper function to print tensor information
    let print_tensor_info = |name: &str, tensor: &ComputeTensor, gpu_idx: Option<usize>, 
                            data: &[f32], shape: &[usize]| {
        println!("\n{}:", name);
        println!("  Location: {}", match gpu_idx {
            Some(idx) => format!("GPU {}", idx),
            None => "CPU".to_string(),
        });
        println!("  Shape: {:?}", shape);
        println!("  Size in memory: {:.2} MB", 
            (data.len() * std::mem::size_of::<f32>()) as f32 / (1024.0 * 1024.0));
        println!("  Values: {}", format_array(data, 10));
        
        print_tensor_stats(data);
    };

    // Get tensor data based on location
    let get_tensor_data = |tensor: &ComputeTensor| -> Result<(Vec<f32>, Option<usize>), VKMLEngineError> {
        match &tensor.location {
            ComputeLocation::CPU(data) => {
                Ok((data.clone(), None))
            },
            ComputeLocation::GPU { gpu_idx, memory } => {
                let data = cm.gpus[*gpu_idx].read_memory(memory)
                    .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                Ok((data, Some(*gpu_idx)))
            },
            ComputeLocation::Parameterless => {
                Ok((vec![], None))
            },
        }
    };

    // Print forward pass tensors
    if compute_layer.desc.requires_parameters {
        let (weights_data, gpu_idx) = get_tensor_data(&compute_layer.weights)?;
        print_tensor_info("Weights", &compute_layer.weights, gpu_idx, &weights_data, &compute_layer.weights.desc.to_dims());

        let (biases_data, gpu_idx) = get_tensor_data(&compute_layer.biases)?;
        print_tensor_info("Biases", &compute_layer.biases, gpu_idx, &biases_data, &compute_layer.biases.desc.to_dims());
    } else {
        println!("\nParameters: None (activation layer)");
    }

    // Print activations
    let (activations_data, gpu_idx) = get_tensor_data(&compute_layer.activations)?;
    print_tensor_info("Activations", &compute_layer.activations, gpu_idx, &activations_data, &compute_layer.activations.desc.to_dims());

    // Print gradient tensors
    println!("\nGradients:");
    if compute_layer.desc.requires_parameters {
        let (weight_grads_data, gpu_idx) = get_tensor_data(&compute_layer.weight_gradients)?;
        print_tensor_info("Weight Gradients", &compute_layer.weight_gradients, gpu_idx, &weight_grads_data, &compute_layer.weight_gradients.desc.to_dims());

        let (bias_grads_data, gpu_idx) = get_tensor_data(&compute_layer.bias_gradients)?;
        print_tensor_info("Bias Gradients", &compute_layer.bias_gradients, gpu_idx, &bias_grads_data, &compute_layer.bias_gradients.desc.to_dims());
    }

    if let Some(activation_gradients) = &compute_layer.activation_gradients {
        let (act_grads_data, gpu_idx) = get_tensor_data(activation_gradients)?;
        print_tensor_info("Activation Gradients", activation_gradients, gpu_idx, &act_grads_data, &activation_gradients.desc.to_dims());
    } else {
        println!("  Activation Gradients: Disabled for this layer");
    }

    Ok(())
}