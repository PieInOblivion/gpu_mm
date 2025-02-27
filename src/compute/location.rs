use crate::gpu::gpu_memory::GPUMemory;

pub enum ComputeLocation {
    CPU(Vec<f32>),
    GPU { gpu_idx: usize, memory: GPUMemory },
    Unallocated,
}