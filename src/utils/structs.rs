// use cudarc::driver::sys::CUarray_format_enum;

// struct init
// init gpu
// get free memory
// add 3d vec to queue
// copy to gpu memory, return pointer
//    TODO HARD: On too low memory, use CUDA STREAM MEMORY instead, warn user with Result
//    Try run large batches for minimal memory callbacks
//    Might require splitting individual layers when on small budget
// run queue
// keep result in gpu memory, pulling out is manual, unless running CUDA STREAM
// Manual implementation of filepath files streaming wanted. Use async memory copy
// while computing results. Manual memory management needed.
// One GPU struct to manage multiple data types? It's so much easier to develop
// for one type per struct init. But for easy user side it should be one struct for all.

// MultiGPU not recommended using official NCCL
// https://github.com/coreylowman/cudarc/pull/164

// https://docs.rs/cudarc/latest/cudarc/driver/sys/enum.CUarray_format_enum.html
// https://docs.rs/cudarc/latest/cudarc/types/trait.CudaTypeName.html
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9
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
