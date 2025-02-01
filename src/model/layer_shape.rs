#[derive(Clone)]
pub struct LayerShape {
    // Common parameters
    pub in_features: usize,
    pub out_features: usize,
    
    // Optional conv parameters
    pub kernel_w: Option<usize>,
    pub kernel_h: Option<usize>,
    pub stride_w: Option<usize>,
    pub stride_h: Option<usize>,
    pub padding_w: Option<usize>,
    pub padding_h: Option<usize>,
    
    // Optional activation parameters
    pub alpha: Option<f32>,     // For LeakyReLU
    pub dim: Option<usize>,     // For Softmax
}

impl Default for LayerShape {
    fn default() -> Self {
        Self {
            in_features: 0,
            out_features: 0,
            kernel_w: None,
            kernel_h: None,
            stride_w: None,
            stride_h: None,
            padding_w: None,
            padding_h: None,
            alpha: None,
            dim: None,
        }
    }
}