use std::collections::HashMap;

use crate::{compute::{compute_manager::ComputeTensor, location::ComputeLocation}, dataloader::error::VKMLEngineError, model::{instruction::Instruction, tensor_desc::TensorDesc}};

use super::{execution::LayerExecution, layer::Layer};

#[derive(Clone)]
pub struct Conv2DLayer {
    pub in_features: usize,  // Input channels
    pub out_features: usize, // Output channels
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub bias: bool,
}

impl Conv2DLayer {
    pub fn new(in_features: usize, out_features: usize,) -> Self {
        Self {
            in_features,
            out_features,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            padding_h: 0,
            padding_w: 0,
            bias: false,
        }
    }

    pub fn new_with(
        in_features: usize,
        out_features: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        bias: bool
    ) -> Self {
        Self {
            in_features,
            out_features,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            bias,
        }
    }
}

impl Layer for Conv2DLayer {
    fn output_shapes(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<Vec<TensorDesc>, VKMLEngineError> {
        if input_shapes.len() != 1 {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Conv2D layer requires exactly 1 input, got {}", input_shapes.len())
            ));
        }
        
        let input_shape = match input_shapes[0] {
            TensorDesc::Tensor4D { batch: _, channels, height, width } => {
                if *channels != self.in_features {
                    return Err(VKMLEngineError::VulkanLoadError(
                        format!("Conv2D expected {} input channels, got {}", self.in_features, channels)
                    ));
                }
                (*height, *width)
            },
            _ => {
                return Err(VKMLEngineError::VulkanLoadError(
                    format!("Conv2D requires 4D input tensor, got {:?}", input_shapes[0])
                ));
            }
        };
        
        let h_in = input_shape.0;
        let w_in = input_shape.1;
        
        let h_out = ((h_in + 2 * self.padding_h - self.kernel_h) / self.stride_h) + 1;
        let w_out = ((w_in + 2 * self.padding_w - self.kernel_w) / self.stride_w) + 1;
        
        Ok(vec![TensorDesc::Tensor4D {
            batch: batch_size,
            channels: self.out_features,
            height: h_out,
            width: w_out
        }])
    }
    
    fn memory_requirements(&self, _input_shapes: &[&TensorDesc], output_shape: &TensorDesc) -> u64 {
        // Calculate weights size (out_channels * in_channels * kernel_h * kernel_w)
        let weights_size = (self.out_features * self.in_features *
                           self.kernel_h * self.kernel_w *
                           std::mem::size_of::<f32>()) as u64;
        
        // Calculate bias size (out_channels)
        let bias_size = if self.bias {
            (self.out_features * std::mem::size_of::<f32>()) as u64
        } else {
            0
        };
        
        let activation_size = output_shape.size_in_bytes() as u64;
        
        let gradient_size = weights_size + bias_size + activation_size;
        
        weights_size + bias_size + activation_size + gradient_size
    }
    
    fn requires_gradients(&self) -> bool {
        true
    }
    
    fn parameter_shapes(&self, _input_shapes: &[&TensorDesc]) -> Option<(TensorDesc, TensorDesc)> {
        let weights = TensorDesc::Tensor4D {
            batch: self.out_features,
            channels: self.in_features,
            height: self.kernel_h,
            width: self.kernel_w
        };
        
        let biases = TensorDesc::Vector {
            length: self.out_features
        };
        
        Some((weights, biases))
    }

    fn parameter_count(&self, _batch_size: usize, _input_shapes: &[&TensorDesc]) -> usize {
        let weight_params = self.out_features * self.in_features * self.kernel_h * self.kernel_w;
        let bias_params = if self.bias { self.out_features } else { 0 };
        
        weight_params + bias_params
    }
    
    fn input_requirements(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }

    fn name(&self) -> String {
        "Conv2D".to_string()
    }
    
    fn config_string(&self) -> Option<String> {
        Some(format!(
            "in_channels={}, out_channels={}, kernel={}×{}, stride={}×{}, padding={}×{}, bias={}", 
            self.in_features, self.out_features,
            self.kernel_h, self.kernel_w,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.bias
        ))
    }
    
    fn in_features(&self) -> usize {
        self.in_features
    }
    
    fn out_features(&self) -> usize {
        self.out_features
    }

    fn build_layer_exec(&self, batch_size: usize, input_shapes: &[&TensorDesc]) -> Result<LayerExecution, VKMLEngineError> {
        if input_shapes.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Conv2D layer requires an input".to_string()
            ));
        }
        
        let input_shape = input_shapes[0];
        
        let (in_channels, in_height, in_width) = match input_shape {
            TensorDesc::Tensor4D { batch: _, channels, height, width } => {
                if *channels != self.in_features {
                    return Err(VKMLEngineError::VulkanLoadError(
                        format!("Conv2D layer expects {} input channels, got {}", self.in_features, channels)
                    ));
                }
                (*channels, *height, *width)
            },
            _ => return Err(VKMLEngineError::VulkanLoadError(
                "Conv2D layer expects 4D tensor input".into()
            )),
        };
        
        let out_height = ((in_height + 2 * self.padding_h - self.kernel_h) / self.stride_h) + 1;
        let out_width = ((in_width + 2 * self.padding_w - self.kernel_w) / self.stride_w) + 1;
        
        let mut tensors = HashMap::new();
        
        tensors.insert("input".to_string(), ComputeTensor {
            desc: input_shape.clone(),
            location: ComputeLocation::Unallocated,
        });
        
        tensors.insert("weights".to_string(), ComputeTensor {
            desc: TensorDesc::new_tensor4d(
                self.out_features,
                self.in_features,
                self.kernel_h,
                self.kernel_w
            ),
            location: ComputeLocation::Unallocated,
        });
        
        tensors.insert("output".to_string(), ComputeTensor {
            desc: TensorDesc::new_tensor4d(
                batch_size,
                self.out_features,
                out_height,
                out_width
            ),
            location: ComputeLocation::Unallocated,
        });
        
        if self.bias {
            tensors.insert("bias".to_string(), ComputeTensor {
                desc: TensorDesc::new_vector(self.out_features),
                location: ComputeLocation::Unallocated,
            });
        }
        
        let mut instructions = Vec::new();
        
        instructions.push(Instruction::ReadInput {
            layer_idx: 0,
            layer_tensor_idx: 0,
            dst: "input".to_string(),
        });
        
        instructions.push(Instruction::Conv2D {
            src: "input".to_string(),
            weights: "weights".to_string(),
            bias: if self.bias { Some("bias".to_string()) } else { None },
            dst: "output".to_string(),
            stride: (self.stride_h, self.stride_w),
            padding: (self.padding_h, self.padding_w),
        });
        
        Ok(LayerExecution {
            tensors,
            instructions,
            outputs: vec!["output".to_string()],
        })
    }
}