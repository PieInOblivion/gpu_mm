use std::collections::HashMap;

use crate::{
    dataloader::error::VKMLEngineError, 
    model::instruction::Instruction, tensor::{compute_tensor::ComputeTensor, tensor_data::TensorData, tensor_desc::TensorDesc}
};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct ReshapeLayer {
    target_shape: TensorDesc,
}

impl ReshapeLayer {
    pub fn new(target_shape: TensorDesc) -> Self {
        Self { target_shape }
    }

    pub fn flatten() -> Self {
        Self { 
            target_shape: TensorDesc::Matrix { 
                rows: 0,  // Will be replaced with batch size 
                cols: 0   // Will be inferred 
            } 
        }
    }
}

impl Layer for ReshapeLayer {
    fn output_shapes(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() != 1 {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Reshape layer requires exactly 1 input, got {}", input_shapes.len())
            ));
        }

        let input_shape = input_shapes[0];
        let input_elements = input_shape.num_elements();

        // Resolve the target shape, handling special cases for 0 dimensions
        let resolved_shape = match &self.target_shape {
            TensorDesc::Vector { length } => {
                if *length == 0 {
                    TensorDesc::Vector { length: input_elements }
                } else {
                    if *length != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into vector of length {}", 
                                    input_elements, length)
                        ));
                    }
                    self.target_shape.clone()
                }
            },
            TensorDesc::Matrix { rows, cols } => {
                if *rows == 0 && *cols == 0 {
                    if input_elements % batch_size != 0 {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot flatten {} elements into batches of size {}, not evenly divisible", 
                                    input_elements, batch_size)
                        ));
                    }
                    TensorDesc::Matrix { 
                        rows: batch_size, 
                        cols: input_elements / batch_size 
                    }
                } else if *rows == 0 {
                    // Infer rows
                    let inferred_rows = input_elements / cols;
                    if inferred_rows * cols != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into matrix with {} columns", 
                                    input_elements, cols)
                        ));
                    }
                    TensorDesc::Matrix {
                        rows: inferred_rows,
                        cols: *cols
                    }
                } else if *cols == 0 {
                    // Infer cols
                    let inferred_cols = input_elements / rows;
                    if inferred_cols * rows != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into matrix with {} rows", 
                                    input_elements, rows)
                        ));
                    }
                    TensorDesc::Matrix {
                        rows: *rows,
                        cols: inferred_cols
                    }
                } else {
                    if rows * cols != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into matrix of size {}x{}", 
                                    input_elements, rows, cols)
                        ));
                    }
                    self.target_shape.clone()
                }
            },
            TensorDesc::Tensor4D { batch, channels, height, width } => {
                let zeros = (*batch == 0) as usize + (*channels == 0) as usize +
                    (*height == 0) as usize + (*width == 0) as usize;
                
                if zeros > 1 {
                    return Err(VKMLEngineError::VulkanLoadError(
                        "At most one dimension can be inferred (set to 0) in Tensor4D".to_string()
                    ));
                } else if zeros == 1 {
                    let known_product = 
                        (if *batch == 0 { 1 } else { *batch }) *
                        (if *channels == 0 { 1 } else { *channels }) *
                        (if *height == 0 { 1 } else { *height }) *
                        (if *width == 0 { 1 } else { *width });
                    
                    let inferred_dim = input_elements / known_product;
                    
                    if inferred_dim * known_product != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into tensor4D with product {}", 
                                    input_elements, known_product)
                        ));
                    }
                    
                    TensorDesc::Tensor4D {
                        batch: if *batch == 0 { inferred_dim } else { *batch },
                        channels: if *channels == 0 { inferred_dim } else { *channels },
                        height: if *height == 0 { inferred_dim } else { *height },
                        width: if *width == 0 { inferred_dim } else { *width },
                    }
                } else {
                    let total_elements = batch * channels * height * width;
                    if total_elements != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into tensor4D with {} elements", 
                                    input_elements, total_elements)
                        ));
                    }
                    self.target_shape.clone()
                }
            }
        };

        Ok(vec![resolved_shape])
    }
    
    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        output_shape.size_in_bytes() as u64
    }
    
    fn requires_gradients(&self) -> bool {
        true
    }
    
    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        "Reshape".to_string()
    }
    
    fn config_string(&self) -> Option<String> {
        let shape_str = match &self.target_shape {
            TensorDesc::Vector { length } => format!("vector(length={})", length),
            TensorDesc::Matrix { rows, cols } => {
                if *rows == 0 && *cols == 0 {
                    "flatten".to_string()
                } else {
                    format!("matrix(rows={}, cols={})", rows, cols)
                }
            },
            TensorDesc::Tensor4D { batch, channels, height, width } => 
                format!("tensor4d(batch={}, channels={}, height={}, width={})", 
                    batch, channels, height, width),
        };
        Some(format!("target_shape={}", shape_str))
    }
    
    fn out_features(&self) -> usize {
        match &self.target_shape {
            TensorDesc::Vector { length } => *length,
            TensorDesc::Matrix { rows, cols } => rows * cols,
            TensorDesc::Tensor4D { batch, channels, height, width } => batch * channels * height * width,
        }
    }

    fn build_layer_exec(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<LayerExecution, VKMLEngineError> {
        if input_shapes.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Reshape layer requires an input".to_string()
            ));
        }
        
        let input_shape = input_shapes[0];
        
        let mut tensors = HashMap::new();
        
        tensors.insert("input".to_string(), ComputeTensor {
            desc: input_shape.clone(),
            data: TensorData::Unallocated,
        });

        let output_shapes = self.output_shapes(batch_size, &[input_shape])?;
        let output_shape = output_shapes[0].clone();
        
        tensors.insert("output".to_string(), ComputeTensor {
            desc: output_shape.clone(),
            data: TensorData::Unallocated,
        });
        
        let instructions = vec![
            Instruction::ReadInput {
                layer_idx: 0,
                layer_tensor_idx: 0,
                dst: "input".to_string(),
            },
            
            Instruction::Reshape {
                src: "input".to_string(),
                dst: "output".to_string(),
                new_shape: output_shape,
            },
        ];
        
        Ok(LayerExecution {
            tensors,
            instructions,
            outputs: vec!["output".to_string()],
        })
    }
}