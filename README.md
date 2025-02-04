# gpu_mm

This library contains high-level abstractions to make ML model development and usage easy and compute efficient.

## Project Priorities In Order
1. Compute efficiency
2. Ease of use

## Overview
This project was inspired by research showing CUDA's limitations (as demonstrated in [this IEEE paper](https://ieeexplore.ieee.org/document/10036080)). The current focus is on Vulkan support. As Vulkan compute gradually evolves into standardised specifications and extensions, we're currently working with shader computations.

The project aims to provide abstractions at a level similar to PyTorch, including multi-gpu support.

## Current Implementation Details (Assumptions, Descisions and Todo's)

### Image Loading
* Current proof of concept implementation stores all file names in memory
  * Raw filesystems typically don't store file counts and aren't sorted, so we provide users that option for replicatability
  * Direct filesystem read into per batch means end files can never be in the first batch. Requires preread and store of filesystem
  * Future support planned for CSV and other formats
    * This will stop the need for prereading directory
  * Raw binary support planned

### Thread Pool Implementation
* Currently created once, leading to single-threaded usage creating a whole pool of one worker
  * This means tasks are loaded in advance, requiring more memory than running without a work queue
* Thread pool will be implemented as an option in future
* Current batch processing generates entire batch before submitting work
  * Could benefit from periodic queue flushing instead of sequential generate -> push -> work pattern

### GPU Management
* Currently assumes all GPU memory is free
  * Will implement VK_EXT_memory_budget in future (commonly implemented extension)
  * Final implementation will track own usage and initial usage from other processes
    * Will include configurable threshold (e.g., 95% of free memory)
* GPU filtering currently checks compute capability
  * Future investigation needed for non-compute flag GPUs
* GPU-to-GPU movement currently routes through CPU
  * Need to investigate Vulkan device pools
  * Research needed on VK shared memory pool extensions

### Architecture Decisions
* Model, Layer, Tensor etc. act as descriptors only
  * Allows compute manager to handle all data and memory
  * Enables future support for alternative compute managers
* ImageBatch to f32 function assumes little endian storage
* Current GPU memory calculations:
  * Don't account for allocation overhead (acceptable with safe memory threshold)
  * Don't track CPU memory requirements
* Model storage is sequential in memory
  * Prevents small layers being stored out of order on multi-device compute configurations
  * Avoids unnecessary CPU transfers
* Current compute implementation:
  * Sends and waits for single GPU commands
  * Future improvement: multiple simultaneous commands using threadpool or native Vulkan solution

## Building
* Requires [glslc](https://github.com/google/shaderc) in PATH to compile shaders

## References

### Vulkan Resources
* [Cooperative Matrix Performance](https://github.com/jeffbolznv/vk_cooperative_matrix_perf)
* [Vulkan Tutorial PDF](https://vulkan-tutorial.com/resources/vulkan_tutorial_en.pdf)
* [Rust Vulkan Tutorial](https://github.com/unknownue/vulkan-tutorial-rust)
* [Ash-rs](https://github.com/ash-rs/ash)
* [Vulkano](https://github.com/vulkano-rs/vulkano)
* [VK_NV_cooperative_matrix Spec](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_cooperative_matrix.html)
* [VkCooperativeMatrixPropertiesNV Spec](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCooperativeMatrixPropertiesNV.html)
* [VkFFT](https://github.com/DTolm/VkFFT)
* [IEEE Paper](https://ieeexplore.ieee.org/document/10036080)

### CUDA Resources
* [cudarc Documentation](https://docs.rs/cudarc/latest/cudarc/)
* [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)

#### Additional CUDA Projects
* [cudarc](https://github.com/coreylowman/cudarc)
* [cuda-sys](https://github.com/rust-cuda/cuda-sys)
* [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)
* [gpgpu-rs](https://github.com/UpsettingBoy/gpgpu-rs)
* [rust-cudnn](https://github.com/autumnai/rust-cudnn)
* [arrayfire-rust](https://github.com/arrayfire/arrayfire-rust)

### Related Projects
* [Candle](https://github.com/huggingface/candle)
* [AdaptiveCpp](https://adaptivecpp.github.io/AdaptiveCpp/)