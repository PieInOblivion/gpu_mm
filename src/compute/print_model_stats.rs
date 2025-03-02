use crate::{compute::compute_manager::ComputeManager, dataloader::error::VKMLEngineError, model::layer_connection::{LayerConnection, LayerId}, tensor::compute_tensor::ComputeTensor};

pub fn print_model_stats(cm: &ComputeManager) {
    let mut total_params = 0usize;
    let mut total_memory = 0u64;
    
    println!("\nModel Statistics");
    println!("================");
    println!("\nBatch Size: {}", cm.model.batch_size);
    println!("\nLayer Details:");
    println!("{:-<125}", "");
    println!("{:<4} {:<12} {:<12} {:<10} {:<18} {:<18} {:<12} {:<20} {}", 
        "ID", "Type", "Params", "Memory", "Input Shape", "Output Shape", "Device", "Connections", "Config");
    println!("{:-<125}", "");

    let execution_order = cm.get_execution_order_slice();
    
    if execution_order.is_empty() {
        println!("Warning: Model not verified, execution order may be incorrect");
        return;
    }

    // Sort the layer IDs in ascending order for consistent output
    let mut ordered_layer_ids: Vec<LayerId> = execution_order.to_vec();
    ordered_layer_ids.sort();

    for &layer_id in &ordered_layer_ids {
        if let Some(layer) = cm.model.layers.get(&layer_id) {
            let output_tensor_name = cm.get_layer_output_tensor_name(layer_id)
                .unwrap_or("output");
            
            // Get input shapes
            let input_shapes_str = if layer.input_connections.is_empty() {
                "None".to_string()
            } else {
                cm.execution_pipeline.iter()
                    .find(|step| step.layer_id == layer_id)
                    .map(|step| {
                        step.input_tensors.iter()
                            .map(|shape| {
                                let dims = shape.to_dims();
                                if dims.len() <= 4 {
                                    dims.iter()
                                        .map(|&d| d.to_string())
                                        .collect::<Vec<_>>()
                                        .join("×")
                                } else {
                                    format!("{}d tensor", dims.len())
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_else(|| "Unknown".to_string())
            };
            
            // Get output shapes
            let output_shapes_str = cm.execution_pipeline.iter()
                .find(|step| step.layer_id == layer_id)
                .map(|step| {
                    step.output_tensors.iter()
                        .map(|shape| {
                            let dims = shape.to_dims();
                            if dims.len() <= 4 {
                                dims.iter()
                                    .map(|&d| d.to_string())
                                    .collect::<Vec<_>>()
                                    .join("×")
                            } else {
                                format!("{}d tensor", dims.len())
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_else(|| "Unknown".to_string());
            
            // Format input and output connections
            let connections_str = format_layer_connections(&layer.input_connections, &layer.output_connections);
            
            let memory_bytes = cm.get_layer_memory_usage(layer_id);
            let params = cm.calculate_layer_parameters(layer_id);
            
            let device_location = cm.get_layer_tensor(layer_id, output_tensor_name)
                .map(|t| cm.get_device_description(t))
                .unwrap_or_else(|| "Unallocated".to_string());
            
            let layer_type = layer.layer.name();
            let layer_config = layer.layer.config_string().unwrap_or_default();
            
            println!("{:<4} {:<12} {:<12} {:<10} {:<18} {:<18} {:<12} {:<20} {}", 
                layer_id, 
                layer_type, 
                params, 
                cm.format_memory_mb(memory_bytes), 
                input_shapes_str, 
                output_shapes_str, 
                device_location,
                connections_str,
                layer_config);
            
            total_params += params;
            total_memory += memory_bytes;
        }
    }

    println!("{:-<125}", "");
    
    // Print a more detailed connection view, also in sorted order
    println!("\nLayer Connections (detailed):");
    println!("{:-<80}", "");
    println!("{:<4} {:<35} {:<35}", "ID", "Inputs From (layer:output)", "Outputs To (layer:output)");
    println!("{:-<80}", "");
    
    for &layer_id in &ordered_layer_ids {
        if let Some(layer) = cm.model.layers.get(&layer_id) {
            // Preserve original connection order
            let inputs_str = if layer.input_connections.is_empty() {
                "None".to_string()
            } else {
                layer.input_connections.iter()
                    .map(|conn| match conn {
                        LayerConnection::DefaultOutput(id) => format!("{}:0", id),
                        LayerConnection::SpecificOutput(id, idx) => format!("{}:{}", id, idx),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            
            let outputs_str = if layer.output_connections.is_empty() {
                "None".to_string()
            } else {
                layer.output_connections.iter()
                    .map(|conn| match conn {
                        LayerConnection::DefaultOutput(id) => format!("{}:0", id),
                        LayerConnection::SpecificOutput(id, idx) => format!("{}:{}", id, idx),
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            
            println!("{:<4} {:<35} {:<35}", 
                layer_id, 
                inputs_str,
                outputs_str);
        }
    }
    
    println!("{:-<80}", "");
    
    // Sort entry and exit points for consistent display
    let mut entry_points = cm.model.verified.as_ref().map_or(vec![], |v| v.entry_points.clone());
    let mut exit_points = cm.model.verified.as_ref().map_or(vec![], |v| v.exit_points.clone());
    entry_points.sort();
    exit_points.sort();
    
    println!("\nGraph Structure:");
    println!("Entry points: {:?}", entry_points);
    println!("Exit points: {:?}", exit_points);
    
    println!("\nModel Summary:");
    println!("Total Parameters: {}", total_params);
    println!("Total Memory: {}", cm.format_memory_mb(total_memory));
    
    println!("\nMemory Allocation:");
    for (device, used, available) in cm.get_memory_usage_summary() {
        println!("{} Memory Used: {}", device, used);
        println!("{} Memory Available: {}", device, available);
    }
}

// Helper function to format layer connections in a compact way
fn format_layer_connections(inputs: &[LayerConnection], outputs: &[LayerConnection]) -> String {
    // Maintain original order of connections
    let in_ids: Vec<String> = inputs.iter()
        .map(|conn| match conn {
            LayerConnection::DefaultOutput(id) => id.to_string(),
            LayerConnection::SpecificOutput(id, _) => id.to_string(),
        })
        .collect();
    
    let out_ids: Vec<String> = outputs.iter()
        .map(|conn| match conn {
            LayerConnection::DefaultOutput(id) => id.to_string(),
            LayerConnection::SpecificOutput(id, _) => id.to_string(),
        })
        .collect();
    
    if in_ids.is_empty() && out_ids.is_empty() {
        return "None".to_string();
    }
    
    let mut result = String::new();
    
    if !in_ids.is_empty() {
        result.push_str(&format!("in:[{}]", in_ids.join(",")));
    }
    
    if !out_ids.is_empty() {
        if !result.is_empty() {
            result.push_str(" ");
        }
        result.push_str(&format!("out:[{}]", out_ids.join(",")));
    }
    
    result
}

pub fn print_layer_values(cm: &ComputeManager, layer_id: LayerId) -> Result<(), VKMLEngineError> {
    let layer = cm.model.layers.get(&layer_id).ok_or(
        VKMLEngineError::VulkanLoadError(format!("Layer ID {} not found", layer_id))
    )?;
    
    println!("\nLayer {} Values ({})", layer_id, layer.layer.name());
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