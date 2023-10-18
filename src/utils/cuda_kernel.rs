extern crate libc;

use std::ffi::CString;
use std::ptr;
use cudarc::driver::{CudaStream, DriverError};
use cudarc::driver::sys::*;
use libc::{c_void, c_int};
use crate::utils::cuda_buffer::CudaBuffer;

#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_uint) -> c_int;

    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> c_int;
    fn cuModuleGetFunction(function: *mut *mut c_void, module: *mut c_void, name: *const c_char) -> c_int;
    fn cuLaunchKernel(function: *mut c_void, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_ulonglong, hStream: *mut c_void, kernelParams: *mut *mut c_void, extra: *mut *mut c_void) -> c_int;
}

pub struct CudaKernel {
    module: CUmodule,
    kernel: CUfunction,
    // TODO
}

impl CudaKernel {

    pub fn launch_prefetch_async<T>(&self, grid_size: usize, block_size: usize, params: &CudaBuffer<T>, stream: &CudaStream) -> Result<(), DriverError> {
    }

    pub fn new(ptx_path: &str, kernel_name: &str) -> Self {
    }

    pub fn launch(&self) {

        unsafe {
        }
    }
}