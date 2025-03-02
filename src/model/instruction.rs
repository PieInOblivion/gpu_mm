use crate::tensor::tensor_desc::TensorDesc;

// Execution tape instructions
pub enum Instruction {
    // Basic operations
    MatMul { src1: String, src2: String, dst: String },
    Add { src1: String, src2: String, dst: String },
    Sub { src1: String, src2: String, dst: String },
    Mul { src1: String, src2: String, dst: String },
    Div { src1: String, src2: String, dst: String },
    Min { src1: String, src2: String, dst: String },
    Max { src1: String, src2: String, dst: String },
    
    // Convolution
    Conv2D {
        src: String,
        weights: String,
        bias: Option<String>,
        dst: String,
        stride: (usize, usize),
        padding: (usize, usize),
    },
    
    // Activation functions
    ReLU { src: String, dst: String },
    LeakyReLU { src: String, dst: String, alpha: f32 },
    Sigmoid { src: String, dst: String },
    Softmax { src: String, dst: String, dim: usize },
    Tanh { src: String, dst: String },
    GELU { src: String, dst: String },
    SiLU { src: String, dst: String },
    
    // Data movement and shaping
    ReadInput {
        layer_idx: usize,          // Which input port of the current layer
        layer_tensor_idx: usize,   // Which output port of the source layer
        dst: String                // Destination tensor name
    },
    CopyInput {
        layer_idx: usize,          // Which input port of the current layer
        layer_tensor_idx: usize,   // Which output port of the source layer
        dst: String                // Destination tensor name
    },
    Reshape { src: String, dst: String, new_shape: TensorDesc },
    Concat {
        sources: Vec<String>,      // Names of input tensors
        dst: String,               // Name of output tensor
        dim: usize,                // Dimension along which to concatenate
    },
}