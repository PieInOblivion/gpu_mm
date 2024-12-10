use std::sync::Arc;
use crate::utils::dataloader_error::DataLoaderError;
use crate::compute::compute_manager::{ComputeLocation, ComputeManager};

pub struct Tensor {
    shape: Vec<usize>,
    pub location: ComputeLocation,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Result<Self, DataLoaderError> {
        let size: usize = shape.iter().product();
        Ok(Self {
            shape,
            location: ComputeLocation::CPU(vec![0.0; size]),
        })
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product::<usize>() * std::mem::size_of::<f32>()
    }

    pub fn get_location(&self) -> usize {
        match &self.location {
            ComputeLocation::CPU(_) => usize::MAX,
            ComputeLocation::GPU { gpu_idx, .. } => *gpu_idx,
        }
    }
}