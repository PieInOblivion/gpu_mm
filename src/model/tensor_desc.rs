#[derive(Clone, Debug)]
pub enum TensorDesc {
    // For vectors/1D tensors (e.g., biases)
    Vector { length: usize },
    
    // For matrices/2D tensors (e.g., linear layer weights)
    Matrix { rows: usize, cols: usize },
    
    // For 4D tensors (e.g., conv layer weights, image batches)
    Tensor4D {
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    }
}

impl TensorDesc {
    pub fn new_vector(length: usize) -> Self {
        Self::Vector { length }
    }

    pub fn new_matrix(rows: usize, cols: usize) -> Self {
        Self::Matrix { rows, cols }
    }

    pub fn new_tensor4d(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        Self::Tensor4D { batch, channels, height, width }
    }

    pub fn size_in_bytes(&self) -> usize {
        let num_elements = self.num_elements();
        num_elements * std::mem::size_of::<f32>()
    }

    pub fn num_elements(&self) -> usize {
        match &self {
            Self::Vector { length } => *length,
            Self::Matrix { rows, cols } => rows * cols,
            Self::Tensor4D { batch, channels, height, width } => 
                batch * channels * height * width,
        }
    }

    // Convert to Vec<usize> for compatibility with existing code
    pub fn to_dims(&self) -> Vec<usize> {
        match &self {
            Self::Vector { length } => vec![*length],
            Self::Matrix { rows, cols } => vec![*rows, *cols],
            Self::Tensor4D { batch, channels, height, width } => 
                vec![*batch, *channels, *height, *width],
        }
    }
}