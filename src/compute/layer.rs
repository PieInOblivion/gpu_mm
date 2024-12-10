use crate::utils::dataloader_error::DataLoaderError;

use super::tensor::Tensor;

pub struct Layer {
    pub weights: Tensor,
    pub biases: Tensor,
    layer_type: LayerType,
}

#[derive(Clone)]
pub enum LayerType {
    Linear { in_features: usize, out_features: usize },
    Conv2D { in_channels: usize, out_channels: usize, kernel_size: (usize, usize) },
    ReLU,
    // Add other layer types as needed
}

impl Layer {
    pub fn new(layer_type: LayerType) -> Result<Self, DataLoaderError> {
        let (weights, biases) = match &layer_type {
            LayerType::Linear { in_features, out_features } => {
                let w = Tensor::new(vec![*out_features, *in_features])?;
                let b = Tensor::new(vec![*out_features])?;
                (w, b)
            },
            LayerType::Conv2D { in_channels, out_channels, kernel_size } => {
                let w = Tensor::new(vec![
                    *out_channels, 
                    *in_channels, 
                    kernel_size.0, 
                    kernel_size.1
                ])?;
                let b = Tensor::new(vec![*out_channels])?;
                (w, b)
            },
            LayerType::ReLU => {
                // ReLU has no parameters
                (Tensor::new(vec![0])?, Tensor::new(vec![0])?)
            },
        };

        Ok(Self {
            weights,
            biases,
            layer_type,
        })
    }

    fn memory_size(&self) -> u64 {
        self.weights.size() as u64 + self.biases.size() as u64
    }
}