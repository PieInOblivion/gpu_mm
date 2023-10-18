use cudarc::driver::DriverError;
use crate::utils::cuda_buffer::CudaBuffer;

pub struct MemoryPool<T> {
}

impl<T> MemoryPool<T> {

    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<CudaBuffer<T>, DriverError> {
    }

    pub fn deallocate(&mut self, buffer: &CudaBuffer<T>) {
    }
}

pub struct SmartMemoryPool<T> {
}

impl<T> SmartMemoryPool<T> {

    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<CudaBuffer<T>, DriverError> {
    }

    pub fn deallocate(&mut self, buffer: &CudaBuffer<T>) {
    }

    pub fn optimize(&mut self) {
    }
}