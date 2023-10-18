use cudarc::driver::{CudaStream, DriverError};

pub struct CudaContext {
}

impl CudaContext {

    pub fn new(device_id: i32) -> Result<Self, DriverError> {
    }

    pub fn create_stream(&self) -> Result<CudaStream, DriverError> {
    }
}