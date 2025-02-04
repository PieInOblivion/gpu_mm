use crate::dataloader::error::VKMLEngineError;

use super::{layer_type::LayerType, tensor_desc::TensorDesc};

#[derive(Clone)]
pub struct LayerDesc {
    pub weights: TensorDesc,
    pub biases: TensorDesc,
    pub layer_type: LayerType,
    pub requires_parameters: bool,

    pub weight_gradients: TensorDesc,
    pub bias_gradients: TensorDesc,
    pub activation_gradients: Option<TensorDesc>,
}

impl LayerDesc {
    pub fn new(layer_type: LayerType) -> Self {
        match &layer_type {
            LayerType::InputBuffer { features, track_gradients } => {
                Self {
                    weights: TensorDesc::new_vector(0),
                    biases: TensorDesc::new_vector(0),
                    requires_parameters: false,
                    weight_gradients: TensorDesc::new_vector(0),
                    bias_gradients: TensorDesc::new_vector(0),
                    activation_gradients: if *track_gradients {
                        Some(TensorDesc::new_vector(*features))
                    } else {
                        None
                    },
                    layer_type,
                }
            },
            LayerType::Linear(params) => {
                let w = TensorDesc::new_matrix(params.out_features, params.in_features);
                let b = TensorDesc::new_vector(params.out_features);
                
                Self {
                    weights: w.clone(),
                    biases: b.clone(),
                    requires_parameters: true,
                    weight_gradients: w,
                    bias_gradients: b,
                    activation_gradients: Some(TensorDesc::new_vector(params.out_features)),
                    layer_type,
                }
            },
            LayerType::Conv2D(params) => {
                let w = TensorDesc::new_tensor4d(
                    params.out_features,
                    params.in_features,
                    params.kernel_h.unwrap_or(3),
                    params.kernel_w.unwrap_or(3)
                );
                let b = TensorDesc::new_vector(params.out_features);
                
                Self {
                    weights: w.clone(),
                    biases: b.clone(),
                    requires_parameters: true,
                    weight_gradients: w,
                    bias_gradients: b,
                    activation_gradients: Some(TensorDesc::new_vector(params.out_features)),
                    layer_type,
                }
            },
            _ => Self {
                weights: TensorDesc::new_vector(0),
                biases: TensorDesc::new_vector(0),
                requires_parameters: false,
                weight_gradients: TensorDesc::new_vector(0),
                bias_gradients: TensorDesc::new_vector(0),
                activation_gradients: None, // Will be set during model construction
                layer_type,
            },
        }
    }

    pub fn output_shape(&self, batch_size: usize, input_shape: Option<&TensorDesc>) -> Result<TensorDesc, VKMLEngineError> {
        match &self.layer_type {
            LayerType::InputBuffer { features, .. } => {
                Ok(TensorDesc::Matrix {
                    rows: batch_size,
                    cols: *features
                })
            },
            LayerType::Linear(params) => {
                Ok(TensorDesc::Matrix {
                    rows: batch_size,
                    cols: params.out_features
                })
            },
            LayerType::Conv2D(params) => {
                let input_shape = input_shape.ok_or_else(|| 
                    VKMLEngineError::VulkanLoadError("Conv2D requires input shape".into()))?;

                let (_, _, h_in, w_in) = match input_shape {
                    TensorDesc::Tensor4D { batch: _, channels: _, height, width } => {
                        (batch_size, params.in_features, *height, *width)
                    },
                    _ => return Err(VKMLEngineError::VulkanLoadError(
                        "Conv2D requires 4D input tensor".into()
                    ))
                };

                let k_h = params.kernel_h.unwrap_or(3);
                let k_w = params.kernel_w.unwrap_or(3);

                let s_h = params.stride_h.unwrap_or(1);
                let s_w = params.stride_w.unwrap_or(1);

                let p_h = params.padding_h.unwrap_or(1);
                let p_w = params.padding_w.unwrap_or(1);

                let h_out = ((h_in + 2 * p_h - k_h) / s_h) + 1;
                let w_out = ((w_in + 2 * p_w - k_w) / s_w) + 1;
                
                Ok(TensorDesc::Tensor4D {
                    batch: batch_size,
                    channels: params.out_features,
                    height: h_out,
                    width: w_out
                })
            },
            // Activation layers preserve input shape
            _ => {
                input_shape.cloned()
                    .ok_or_else(|| VKMLEngineError::VulkanLoadError(
                        "Activation layer requires input shape".into()
                    ))
            }
        }
    }

    pub fn memory_requirements(&self) -> u64 {
        let parameter_memory = if self.layer_type.requires_parameters() {
            (self.weights.size_in_bytes() + self.biases.size_in_bytes() +     // Parameters
             self.weight_gradients.size_in_bytes() + self.bias_gradients.size_in_bytes()) as u64  // Parameter gradients
        } else {
            0
        };

        let activation_memory = self.activation_gradients
            .as_ref()
            .map(|grad| grad.size_in_bytes() as u64)
            .unwrap_or(0);

        parameter_memory + activation_memory
    }
}