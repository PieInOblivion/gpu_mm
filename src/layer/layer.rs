use crate::{dataloader::error::VKMLEngineError, model::tensor_desc::TensorDesc};

use super::execution::LayerExecution;

pub trait Layer {
    // Calculate the output shapes for all outputs of this layer
    fn output_shapes(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<Vec<TensorDesc>, VKMLEngineError>;
    
    // Get memory requirements for this layer
    fn memory_requirements(&self, input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64;
    
    // Whether this layer requires trainable parameters
    fn requires_parameters(&self) -> bool;
    
    // Whether this layer requires gradient tracking
    fn requires_gradients(&self) -> bool;
    
    // For parameterized layers, describes the required weight and bias tensors
    fn parameter_shapes(&self, input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)>;
    
    // For graph verification, how many inputs this layer requires (min and max)
    fn input_requirements(&self) -> (usize, Option<usize>);

    // Return a string representation of the layers name
    fn name(&self) -> String;
    
    // Returns a string representation of the layer configuration
    fn to_string(&self) -> String;
    
    // Get input features
    fn in_features(&self) -> usize;
    
    // Get output features
    fn out_features(&self) -> usize;

    // Generate tensor descriptions, instructions, and outputs for this layer
    fn build_layer_exec(&self, batch_size: usize, input_shape: &TensorDesc) -> Result<LayerExecution, VKMLEngineError>;
}