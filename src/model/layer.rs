use crate::dataloader::error::VKMLEngineError;

use super::{layer_shape::LayerType, tensor::TensorDesc};

#[derive(Clone)]
pub struct LayerDesc {
    pub weights: TensorDesc,
    pub biases: TensorDesc,
    pub layer_type: LayerType,
    pub requires_parameters: bool,
}

impl LayerDesc {
    pub fn new(layer_type: LayerType) -> Self {
        match &layer_type {
            LayerType::Linear(params) => {
                let w = TensorDesc::new(vec![params.out_features, params.in_features]);
                let b = TensorDesc::new(vec![params.out_features]);
                Self {
                    weights: w,
                    biases: b,
                    requires_parameters: true,
                    layer_type,
                }
            },
            LayerType::Conv2D(params) => {
                let w = TensorDesc::new(vec![
                    params.out_features,
                    params.in_features,
                    params.kernel_h.unwrap_or(3),
                    params.kernel_w.unwrap_or(3)
                ]);
                let b = TensorDesc::new(vec![params.out_features]);
                Self {
                    weights: w,
                    biases: b,
                    requires_parameters: true,
                    layer_type,
                }
            },
            _ => Self {
                weights: TensorDesc::new(vec![]),
                biases: TensorDesc::new(vec![]),
                requires_parameters: false,
                layer_type,
            },
        }
    }

    pub fn output_shape(&self, batch_size: usize, input_shape: Option<&[usize]>) -> Result<Vec<usize>, VKMLEngineError> {
        match &self.layer_type {
            LayerType::Linear(params) => {
                Ok(vec![batch_size, params.out_features])
            },
            LayerType::Conv2D(params) => {
                let input_shape = input_shape.ok_or_else(|| 
                    VKMLEngineError::VulkanLoadError("Conv2D requires input shape".into()))?;

                let k_h = params.kernel_h.unwrap_or(3);
                let k_w = params.kernel_w.unwrap_or(3);

                let s_h = params.stride_h.unwrap_or(1);
                let s_w = params.stride_w.unwrap_or(1);

                let p_h = params.padding_h.unwrap_or(1);
                let p_w = params.padding_w.unwrap_or(1);
                
                let h_in = input_shape[2];
                let w_in = input_shape[3];
                
                let h_out = ((h_in + 2 * p_h - k_h) / s_h) + 1;
                let w_out = ((w_in + 2 * p_w - k_w) / s_w) + 1;
                
                Ok(vec![batch_size, params.out_features, h_out, w_out])
            },
            // Activation layers preserve input shape
            _ => {
                input_shape.map(|shape| shape.to_vec())
                    .ok_or_else(|| VKMLEngineError::VulkanLoadError(
                        "Activation layer requires input shape".into()))
            }
        }
    }

    pub fn memory_requirements(&self) -> u64 {
        if !self.requires_parameters {
            return 0;
        }
        (self.weights.size() + self.biases.size()) as u64
    }
}