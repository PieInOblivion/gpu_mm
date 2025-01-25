use crate::dataloader::error::DataLoaderError;

use super::tensor::TensorDesc;

#[derive(Clone)]
pub struct LayerDesc {
    pub weights: TensorDesc,
    pub biases: TensorDesc,
    pub layer_type: LayerType,
    pub requires_parameters: bool,
}

#[derive(Clone)]
pub enum LayerType {
    Linear { in_features: usize, out_features: usize },
    Conv2D {
        in_channels: usize,
        out_channels: usize,
        spatial_dims: (usize, usize),  // (height, width)
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize)
    },

    // Basic activations
    ReLU,
    LeakyReLU { alpha: f32 },
    Sigmoid,
    Softmax { dim: usize },
    Tanh,

    // Modern activations
    GELU,
    SiLU,  // Swish    
}

impl LayerDesc {
    pub fn new(layer_type: LayerType) -> Self {
        match &layer_type {
            LayerType::Linear { in_features, out_features } => {
                let w = TensorDesc::new(vec![*out_features, *in_features]);
                let b = TensorDesc::new(vec![*out_features]);
                Self {
                    weights: w,
                    biases: b,
                    requires_parameters: true,
                    layer_type,
                }
            },
            LayerType::Conv2D { 
                in_channels, 
                out_channels, 
                kernel_size,
                ..
            } => {
                let w = TensorDesc::new(vec![
                    *out_channels, 
                    *in_channels, 
                    kernel_size.0, 
                    kernel_size.1
                ]);
                let b = TensorDesc::new(vec![*out_channels]);
                Self {
                    weights: w,
                    biases: b,
                    requires_parameters: true,
                    layer_type,
                }
            },
            LayerType::ReLU |
            LayerType::LeakyReLU { .. } |
            LayerType::Sigmoid |
            LayerType::Softmax { .. } |
            LayerType::Tanh |
            LayerType::GELU |
            LayerType::SiLU => {
                Self {
                    weights: TensorDesc::new(vec![]),
                    biases: TensorDesc::new(vec![]),
                    requires_parameters: false,
                    layer_type,
                }
            },
        }
    }

    pub fn output_shape(&self, batch_size: usize, input_shape: Option<&[usize]>) -> Result<Vec<usize>, DataLoaderError> {
        match &self.layer_type {
            LayerType::Linear { out_features, .. } => {
                Ok(vec![batch_size, *out_features])
            },
            LayerType::Conv2D { 
                out_channels,
                spatial_dims,
                kernel_size,
                stride,
                padding,
                dilation,
                ..
            } => {
                let h_in = spatial_dims.0;
                let w_in = spatial_dims.1;
                
                let h_out = ((h_in + 2 * padding.0 
                    - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0) + 1;
                let w_out = ((w_in + 2 * padding.1 
                    - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1) + 1;
                
                Ok(vec![batch_size, *out_channels, h_out, w_out])
            },
            // Activation layers preserve input shape
            LayerType::ReLU |
            LayerType::LeakyReLU { .. } |
            LayerType::Sigmoid |
            LayerType::Tanh |
            LayerType::GELU |
            LayerType::SiLU => {
                input_shape.map(|shape| shape.to_vec())
                    .ok_or_else(|| DataLoaderError::VulkanLoadError(
                        "Activation layer requires input shape".into()))
            },
            LayerType::Softmax { dim } => {
                input_shape.map(|shape| shape.to_vec())
                    .ok_or_else(|| DataLoaderError::VulkanLoadError(
                        format!("Softmax layer requires input shape (dim={})", dim)))
            },
        }
    }

    pub fn memory_requirements(&self) -> u64 {
        if !self.requires_parameters {
            return 0;
        }
        (self.weights.size() + self.biases.size()) as u64
    }
}