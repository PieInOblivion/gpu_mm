#[derive(Clone)]
pub struct TensorDesc {
    pub shape: Vec<usize>
}

impl TensorDesc {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product::<usize>() * std::mem::size_of::<f32>()
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}