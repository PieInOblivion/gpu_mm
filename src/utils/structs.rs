// TODO: Swap to vulkan compute... So all of it
pub enum CUDATypes {}

pub struct MultiplyTensorStep {
    pub x: Vec<CUDATypes>,
    pub y: Vec<CUDATypes>,
}

pub struct MultiplyTensorQueue {
    pub queue: Vec<MultiplyTensorStep>,
    pub bytes_per_step: usize,
}

#[derive(Debug)]
pub struct GPU {

}

impl GPU {

}
