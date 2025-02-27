use std::collections::HashMap;

use crate::{
    compute::{compute_manager::ComputeTensor, location::ComputeLocation}, 
    dataloader::error::VKMLEngineError, 
    model::{instruction::Instruction, tensor_desc::TensorDesc}
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
                    TensorDesc::Matrix { 
                        rows: batch_size, 
                        cols: input_elements 
                    }
                } else {
                    // Validate total elements
                    if *length != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into vector of length {}", 
                                    input_elements, length)
                        ));
                    }
                    TensorDesc::Vector { length: *length }
                }
            },
            TensorDesc::Matrix { rows, cols } => {
                let (matrix_rows, matrix_cols) = if *rows == 0 && *cols == 0 {
                    // Flatten case
                    (batch_size, input_elements)
                } else if *rows == 0 {
                    // Infer rows
                    let inferred_rows = input_elements / cols;
                    if inferred_rows * cols != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into matrix with {} columns", 
                                    input_elements, cols)
                        ));
                    }
                    (inferred_rows, *cols)
                } else if *cols == 0 {
                    // Infer cols
                    let inferred_cols = input_elements / rows;
                    if inferred_cols * rows != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into matrix with {} rows", 
                                    input_elements, rows)
                        ));
                    }
                    (*rows, inferred_cols)
                } else {
                    // Validate total elements
                    if rows * cols != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into matrix of size {}x{}", 
                                    input_elements, rows, cols)
                        ));
                    }
                    (*rows, *cols)
                };
                TensorDesc::Matrix { 
                    rows: matrix_rows, 
                    cols: matrix_cols 
                }
            },
            TensorDesc::Tensor4D { batch, channels, height, width } => {
                let (t_batch, t_channels, t_height, t_width) = if *batch == 0 && 
                    *channels == 0 && *height == 0 && *width == 0 {
                    // Complete reshape, infer all dimensions
                    (batch_size, 1, 1, input_elements)
                } else {
                    // Validate dimensions
                    let total_elements = 
                        (if *batch == 0 { batch_size } else { *batch }) *
                        (if *channels == 0 { 1 } else { *channels }) *
                        (if *height == 0 { 1 } else { *height }) *
                        (if *width == 0 { 1 } else { *width });
                    
                    if total_elements != input_elements {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Cannot reshape {} elements into tensor4D", input_elements)
                        ));
                    }
                    
                    (
                        if *batch == 0 { batch_size } else { *batch },
                        if *channels == 0 { 1 } else { *channels },
                        if *height == 0 { 1 } else { *height },
                        if *width == 0 { 1 } else { *width }
                    )
                };
                
                TensorDesc::Tensor4D { 
                    batch: t_batch, 
                    channels: t_channels, 
                    height: t_height, 
                    width: t_width 
                }
            }
        };

        Ok(vec![resolved_shape])
    }
    
    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        output_shape.size_in_bytes() as u64
    }
    
    fn requires_parameters(&self) -> bool {
        false
    }
    
    fn requires_gradients(&self) -> bool {
        true
    }
    
    fn parameter_shapes(&self, _input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)> {
        None
    }
    
    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        "Reshape".to_string()
    }
    
    fn to_string(&self) -> String {
        format!("Reshape({:?})", self.target_shape)
    }
    
    fn in_features(&self) -> usize {
        0  // Dynamic based on input
    }
    
    fn out_features(&self) -> usize {
        match &self.target_shape {
            TensorDesc::Vector { length } => *length,
            TensorDesc::Matrix { rows, cols } => rows * cols,
            TensorDesc::Tensor4D { batch, channels, height, width } => batch * channels * height * width,
        }
    }

    fn build_layer_exec(&self, batch_size: usize, input_shape: &TensorDesc) -> Result<LayerExecution, VKMLEngineError> {
        let mut tensors = HashMap::new();
        
        tensors.insert("input".to_string(), ComputeTensor {
            desc: input_shape.clone(),
            location: ComputeLocation::Unallocated,
        });

        let output_shapes = self.output_shapes(batch_size, &[input_shape])?;
        let output_shape = output_shapes[0].clone();
        
        tensors.insert("output".to_string(), ComputeTensor {
            desc: output_shape.clone(),
            location: ComputeLocation::Unallocated,
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