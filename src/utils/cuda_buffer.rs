use cudarc::driver::{CudaStream, DriverError};

pub struct CudaBuffer<T> {
}

impl<T> CudaBuffer<T> {

    pub fn allocate(size: usize, alignment: usize) -> Result<Self, DriverError> {
    }

    pub fn copy_to_device_async(&self, data: &[T], stream: &CudaStream) -> Result<(), DriverError> {
    }

    pub fn copy_to_host_async(&self, data: &mut [T], stream: &CudaStream) -> Result<(), DriverError> {
    }
}