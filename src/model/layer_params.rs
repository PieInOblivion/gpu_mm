#[derive(Clone)]
pub struct LayerParams {
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

impl Default for LayerParams {
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

#[derive(Clone)]
pub enum LayerType {
    Linear(LayerParams),
    Conv2D(LayerParams),
    ReLU,
    LeakyReLU(f32),
    Sigmoid,
    Softmax(usize),
    Tanh,
    GELU,
    SiLU,
}

impl LayerType {
    pub fn linear(in_features: usize, out_features: usize) -> Self {
        LayerType::Linear(LayerParams {
            in_features,
            out_features,
            ..Default::default()
        })
    }

    pub fn conv2d(in_channels: usize, out_channels: usize) -> Self {
        LayerType::Conv2D(LayerParams {
            in_features: in_channels,
            out_features: out_channels,
            kernel_w: Some(3),
            kernel_h: Some(3),
            stride_w: Some(1),
            stride_h: Some(1),
            padding_w: Some(0),
            padding_h: Some(0),
            ..Default::default()
        })
    }

    pub fn conv2d_with(
        in_channels: usize, 
        out_channels: usize,
        kernel_w: usize,
        kernel_h: usize,
        stride_w: usize,
        stride_h: usize,
        padding_w: usize,
        padding_h: usize,
    ) -> Self {
        LayerType::Conv2D(LayerParams {
            in_features: in_channels,
            out_features: out_channels,
            kernel_w: Some(kernel_w),
            kernel_h: Some(kernel_h),
            stride_w: Some(stride_w),
            stride_h: Some(stride_h),
            padding_w: Some(padding_w),
            padding_h: Some(padding_h),
            ..Default::default()
        })
    }

    pub fn leaky_relu(alpha: f32) -> Self {
        LayerType::LeakyReLU(alpha)
    }

    pub fn softmax(dim: usize) -> Self {
        LayerType::Softmax(dim)
    }

    pub fn requires_parameters(&self) -> bool {
        matches!(self, LayerType::Linear(_) | LayerType::Conv2D(_))
    }

    pub fn get_params(&self) -> LayerParams {
        match self {
            LayerType::Linear(params) | LayerType::Conv2D(params) => params.clone(),
            _ => LayerParams::default(),
        }
    }
}