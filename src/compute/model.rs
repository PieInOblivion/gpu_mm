use crate::{
    utils::dataloader_error::DataLoaderError,
    compute::compute_manager::ComputeManager,
};
use super::{layer::{Layer, LayerType}, tensor::Tensor};

pub struct Model {
    layers: Vec<Layer>,
    compute: ComputeManager,
}

impl Model {
    pub fn new(compute: ComputeManager) -> Self {
        Self {
            layers: Vec::new(),
            compute,
        }
    }

    pub fn add_layer(&mut self, layer_type: LayerType) -> Result<(), DataLoaderError> {
        let mut layer = Layer::new(layer_type)?;
        
        // Find optimal device for layer's tensors
        let weights_size = layer.weights.size() as u64;
        let biases_size = layer.biases.size() as u64;
        let total_size = weights_size + biases_size;
        
        let target_device = self.compute.find_optimal_device(total_size)?;
        
        // Move tensors to optimal device
        self.compute.move_tensor_to_device(&mut layer.weights, &target_device)?;
        self.compute.move_tensor_to_device(&mut layer.biases, &target_device)?;
        
        self.layers.push(layer);
        Ok(())
    }

    pub fn add_layers(&mut self, layer_type: Vec<LayerType>) -> Result<(), DataLoaderError> {
        for layer in layer_type.iter() {
            self.add_layer(layer.clone())?;
        }
        Ok(())
    }

    pub fn forward(&mut self, mut input: Tensor) -> Result<Tensor, DataLoaderError> {
        todo!()
    }
}