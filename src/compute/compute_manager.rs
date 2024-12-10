use crate::{
    gpu::vk_gpu::{GPU, GPUMemory},
    utils::dataloader_error::DataLoaderError,
};

use super::{cpu_compute::CPUCompute, tensor::Tensor};

pub enum DeviceLocation {
    CPU,
    GPU(usize)
}

pub enum ComputeLocation {
    CPU(Vec<f32>),
    GPU {
        gpu_idx: usize,
        memory: GPUMemory,
    }
}

pub struct ComputeManager {
    gpus: Vec<GPU>,
    upto_gpu: usize,
    cpu: CPUCompute,
}

impl ComputeManager {
    pub fn new() -> Result<Self, DataLoaderError> {
        Ok(Self {
            gpus: Self::available_gpus()?,
            upto_gpu: 0,
            cpu: CPUCompute::new(None),
        })
    }

    pub fn new_with(gpus: Vec<GPU>, cpu_memory_limit_bytes: Option<u64>) -> Self {
        Self {
            gpus,
            upto_gpu: 0,
            cpu: CPUCompute::new(cpu_memory_limit_bytes),
        }
    }

    pub fn available_gpus() -> Result<Vec<GPU>, DataLoaderError> {
        let gpu_info = GPU::available_gpus()?;
        let mut gpus = Vec::with_capacity(gpu_info.len());
        
        for info in gpu_info {
            if let Ok(gpu) = GPU::new(info.device_index) {
                gpus.push(gpu);
            }
        }
        
        Ok(gpus)
    }

    pub fn find_optimal_device(&mut self, size: u64) -> Result<DeviceLocation, DataLoaderError> {
        // Try current GPU first
        if self.upto_gpu < self.gpus.len() {
            let gpu = &self.gpus[self.upto_gpu];
            if size <= gpu.available_memory() {
                return Ok(DeviceLocation::GPU(self.upto_gpu));
            }
        }

        // Try subsequent GPUs
        for idx in (self.upto_gpu + 1)..self.gpus.len() {
            let gpu = &self.gpus[idx];
            if size <= gpu.available_memory() {
                self.upto_gpu = idx;
                return Ok(DeviceLocation::GPU(idx));
            }
        }

        // If no GPU has enough memory, try CPU
        if self.cpu.memory_tracking.get_available() >= size {
            Ok(DeviceLocation::CPU)
        } else {
            Err(DataLoaderError::OutOfMemory("No device has enough memory".into()))
        }
    }

    pub fn move_tensor_to_device(
        &mut self,
        tensor: &mut Tensor,
        target_device: &DeviceLocation
    ) -> Result<(), DataLoaderError> {
        // Get sizes for memory tracking
        let size = tensor.size() as u64;

        // Deallocate from current device
        match &tensor.location {
            ComputeLocation::CPU(_) => {
                self.cpu.memory_tracking.deallocate(size);
            },
            ComputeLocation::GPU { gpu_idx, .. } => {
                if let Some(gpu) = self.gpus.get_mut(*gpu_idx) {
                    gpu.deallocate_memory(size);
                }
            }
        }

        // Allocate on new device
        match target_device {
            DeviceLocation::CPU => {
                self.cpu.memory_tracking.allocate(size)?;
            },
            DeviceLocation::GPU(idx) => {
                self.gpus[*idx].allocate_memory(size)?;
            }
        }

        // Perform the actual move
        match &tensor.location {
            ComputeLocation::CPU(data) => {
                match target_device {
                    DeviceLocation::CPU => {
                        // Already on CPU
                        return Ok(());
                    },
                    DeviceLocation::GPU(idx) => {
                        // CPU to GPU
                        let gpu = &self.gpus[*idx];
                        let gpu_mem = gpu.move_to_gpu_as_f32(data)
                            .map_err(|e| DataLoaderError::VulkanLoadError(e.to_string()))?;

                        tensor.location = ComputeLocation::GPU {
                            gpu_idx: *idx,
                            memory: gpu_mem,
                        };
                    }
                }
            },
            ComputeLocation::GPU { gpu_idx, memory } => {
                match target_device {
                    DeviceLocation::CPU => {
                        // GPU to CPU
                        let gpu = &self.gpus[*gpu_idx];
                        let cpu_data = gpu.read_memory(memory)
                            .map_err(|e| DataLoaderError::VulkanLoadError(e.to_string()))?;
                        tensor.location = ComputeLocation::CPU(cpu_data);
                    },
                    DeviceLocation::GPU(target_idx) => {
                        if target_idx == gpu_idx {
                            // Already on target GPU
                            return Ok(());
                        }
                        
                        // GPU to different GPU
                        // TODO: Gpu-to-gpu memory transfer. Not through CPU
                        let source_gpu = &self.gpus[*gpu_idx];
                        let cpu_data = source_gpu.read_memory(memory)
                            .map_err(|e| DataLoaderError::VulkanLoadError(e.to_string()))?;
                        
                        let target_gpu = &self.gpus[*target_idx];
                        let new_gpu_mem = target_gpu.move_to_gpu_as_f32(&cpu_data)
                            .map_err(|e| DataLoaderError::VulkanLoadError(e.to_string()))?;
                        
                        tensor.location = ComputeLocation::GPU {
                            gpu_idx: *target_idx,
                            memory: new_gpu_mem,
                        };
                    }
                }
            }
        }
        Ok(())
    }
}