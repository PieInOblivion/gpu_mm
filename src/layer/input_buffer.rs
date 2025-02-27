use std::collections::HashMap;

use crate::{compute::{compute_manager::ComputeTensor, location::ComputeLocation}, dataloader::error::VKMLEngineError, model::tensor_desc::TensorDesc};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct InputLayer {
    pub out_features: usize,
    pub track_gradients: bool,
}

impl InputLayer {
    pub fn new(out_features: usize) -> Self {
        Self {
            out_features,
            track_gradients: false,
        }
    }
    
    pub fn new_with(out_features: usize, track_gradients: bool) -> Self {
        Self {
            out_features,
            track_gradients,
        }
    }
}

impl Layer for InputLayer {
    fn output_shapes(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        // Input layers ignore input_shapes since they're entry points
        if !input_shapes.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("InputBuffer expects 0 inputs, got {}", input_shapes.len())
            ));
        }
        
        Ok(vec![TensorDesc::Matrix {
            rows: batch_size,
            cols: self.out_features
        }])
    }
    
    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Only need memory for activations
        let activation_memory = output_shape.size_in_bytes() as u64;
        
        let gradient_memory = if self.track_gradients {
            output_shape.size_in_bytes() as u64
        } else {
            0
        };
        
        activation_memory + gradient_memory
    }
    
    fn requires_parameters(&self) -> bool {
        false
    }
    
    fn requires_gradients(&self) -> bool {
        self.track_gradients
    }
    
    fn parameter_shapes(&self, _input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)> {
        None
    }
    
    fn input_requirements(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }

    fn name(&self) -> String {
        "InputBuffer".to_string()
    }
    
    fn to_string(&self) -> String {
        if self.track_gradients {
            format!("InputBuffer({}, with gradients)", self.out_features)
        } else {
            format!("InputBuffer({})", self.out_features)
        }
    }
    
    fn in_features(&self) -> usize {
        0
    }
    
    fn out_features(&self) -> usize {
        self.out_features
    }

    fn build_layer_exec(&self, batch_size: usize, _input_shape: &TensorDesc) -> Result<LayerExecution, VKMLEngineError> {
        let mut tensors = HashMap::new();
        
        tensors.insert("output".to_string(), ComputeTensor {
            desc: TensorDesc::new_matrix(batch_size, self.out_features),
            location: ComputeLocation::Unallocated,
        });
        
        if self.track_gradients {
            tensors.insert("grad_output".to_string(), ComputeTensor {
                desc: TensorDesc::new_matrix(batch_size, self.out_features),
                location: ComputeLocation::Unallocated,
            });
        }
        
        // We transfer cpu or other gpu data into this. Layer needs to only exist
        let instructions = vec![];
        
        Ok(LayerExecution {
            tensors,
            instructions,
            outputs: vec!["output".to_string()],
        })
    }
}