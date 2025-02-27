use crate::{compute::compute_manager::{ComputeManager, ComputeTensor}, dataloader::error::VKMLEngineError, model::layer_connection::LayerId};

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

    let execution_order = cm.get_execution_order_slice();
    
    if execution_order.is_empty() {
        println!("Warning: Model not verified, execution order may be incorrect");
        return;
    }

    for &layer_id in execution_order {
        if let Some(layer) = cm.model.layers.get(&layer_id) {
            let output_tensor_name = cm.get_layer_output_tensor_name(layer_id)
                .unwrap_or("output");
            
            // Find the execution step for this layer
            let output_shapes = cm.execution_pipeline.iter()
                .find(|step| step.layer_id == layer_id)
                .map(|step| {
                    step.output_tensors.iter()
                      .map(|shape| shape.to_dims()
                          .iter()
                          .map(|&d| d.to_string())
                          .collect::<Vec<_>>()
                          .join("×"))
                      .collect::<Vec<_>>()
                      .join(", ")
                })
                .unwrap_or_else(|| "Unknown".to_string());
            
            let memory_bytes = cm.get_layer_memory_usage(layer_id);
            let params = cm.calculate_layer_parameters(layer_id);
            
            let device_location = cm.get_layer_tensor(layer_id, output_tensor_name)
                .map(|t| cm.get_device_description(t))
                .unwrap_or_else(|| "Unallocated".to_string());
            
            println!("{:<4} {:<20} {:<15} {:<15} {:<15} {:<15}", 
                layer_id, layer.layer.to_string(), params, cm.format_memory_mb(memory_bytes), 
                output_shapes, device_location);
            
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
    println!("Total Memory: {}", cm.format_memory_mb(total_memory));
    
    println!("\nMemory Allocation:");
    for (device, used, available) in cm.get_memory_usage_summary() {
        println!("{} Memory Used: {}", device, used);
        println!("{} Memory Available: {}", device, available);
    }
}

pub fn print_layer_values(cm: &ComputeManager, layer_id: LayerId) -> Result<(), VKMLEngineError> {
    let layer = cm.model.layers.get(&layer_id).ok_or(
        VKMLEngineError::VulkanLoadError(format!("Layer ID {} not found", layer_id))
    )?;
    
    println!("\nLayer {} Values ({})", layer_id, layer.layer.to_string());
    println!("{:-<120}", "");
    println!("Input connections: {:?}", layer.input_connections);
    println!("Output connections: {:?}", layer.output_connections);

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
            
            let non_zero = data.iter().filter(|&&x| x != 0.0).count();
            println!("    Non-zero elements: {} ({:.2}%)", 
                non_zero, 
                (non_zero as f32 / data.len() as f32) * 100.0
            );
        }
    };

    let print_tensor_info = |name: &str, tensor: &ComputeTensor, gpu_idx: Option<usize>, 
                            data: &[f32], shape: &[usize]| {
        println!("\n{}:", name);
        println!("  Location: {}", match gpu_idx {
            Some(idx) => format!("GPU {}", idx),
            None => "CPU".to_string(),
        });
        println!("  Shape: {:?}", shape);
        println!("  Size in memory: {}", cm.format_memory_mb(
            (data.len() * std::mem::size_of::<f32>()) as u64));
        println!("  Values: {}", format_array(data, 10));
        
        print_tensor_stats(data);
    };

    if let Some(tensor_names) = cm.get_layer_tensor_names(layer_id) {
        for name in tensor_names {
            if let Some(tensor) = cm.get_layer_tensor(layer_id, name) {
                let (data, gpu_idx) = cm.get_tensor_data(tensor)?;
                print_tensor_info(
                    name, 
                    tensor, 
                    gpu_idx, 
                    &data, 
                    &tensor.desc.to_dims()
                );
            }
        }
    }

    if let Some(output_shapes) = cm.get_layer_output_shapes(layer_id) {
        for shape in output_shapes {
            println!("\nOutput Tensor Shape: {:?}", shape.to_dims());
        }
    }

    Ok(())
}