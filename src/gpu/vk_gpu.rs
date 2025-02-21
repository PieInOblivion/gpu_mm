use ash::{vk, Entry, Instance, Device};
use std::{ffi::CString, ptr, sync::Arc};

use crate::{compute::memory_tracker::MemoryTracker, dataloader::error::VKMLEngineError};

use super::{compute_pipelines::{ComputePipelines, GPUMemoryOperation}, gpu_memory::GPUMemory, vk_gpu_info::GPUInfo};

// TODO: Performance gains when needing to multiple tasks in sequence
// TODO: Generalise the usage a little bit more
// NOTE: Get it working, then simplify

// TODO: Look at VK_KHR_device_group. I Think we want to stick with individually managed GPUs though

pub struct GPU {
    entry: Arc<Entry>,
    instance: Instance,
    device: Device,
    physical_device: vk::PhysicalDevice,
    compute_queue: vk::Queue,
    command_pool: vk::CommandPool,
    queue_family_index: u32,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    compute_pipelines: ComputePipelines,
    memory_tracker: MemoryTracker
}

impl GPU {
    pub fn new(device_index: usize) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let entry = Arc::new(Entry::load()?);
            let app_name = CString::new("VK GPU")?;
            
            let appinfo = vk::ApplicationInfo {
                s_type: vk::StructureType::APPLICATION_INFO,
                p_next: ptr::null(),
                p_application_name: app_name.as_ptr(),
                application_version: vk::make_api_version(0, 1, 0, 0),
                p_engine_name: app_name.as_ptr(),
                engine_version: vk::make_api_version(0, 1, 0, 0),
                api_version: vk::make_api_version(0, 1, 0, 0),
                _marker: std::marker::PhantomData,
            };

            let create_info = vk::InstanceCreateInfo {
                s_type: vk::StructureType::INSTANCE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::InstanceCreateFlags::empty(),
                p_application_info: &appinfo,
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                pp_enabled_extension_names: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            let instance = entry.create_instance(&create_info, None)?;

            let physical_devices = instance.enumerate_physical_devices()?;
            let physical_device = *physical_devices.get(device_index)
                .ok_or("GPU index out of range")?;

            let queue_family_index = instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(index, _)| index as u32)
                .ok_or("No compute queue family found")?;

            let device_features = vk::PhysicalDeviceFeatures::default();
            let queue_priorities = [1.0];
            
            let queue_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index,
                queue_count: 1,
                p_queue_priorities: queue_priorities.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let device_create_info = vk::DeviceCreateInfo {
                s_type: vk::StructureType::DEVICE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DeviceCreateFlags::empty(),
                queue_create_info_count: 1,
                p_queue_create_infos: &queue_info,
                enabled_layer_count: 0,
                pp_enabled_layer_names: ptr::null(),
                enabled_extension_count: 0,
                pp_enabled_extension_names: ptr::null(),
                p_enabled_features: &device_features,
                _marker: std::marker::PhantomData,
            };

            let device = instance.create_device(physical_device, &device_create_info, None)?;
            let compute_queue = device.get_device_queue(queue_family_index, 0);

            let command_pool_info = vk::CommandPoolCreateInfo {
                s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
                _marker: std::marker::PhantomData,
            };

            let command_pool = device.create_command_pool(&command_pool_info, None)?;

            // Create descriptor set layout
            let bindings = [
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    p_immutable_samplers: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    p_immutable_samplers: ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];

            let descriptor_layout_info = vk::DescriptorSetLayoutCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set_layout = device.create_descriptor_set_layout(&descriptor_layout_info, None)?;

            // Create descriptor pool
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 100, // Support up to 50 GPUMemory objects
            }];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
                s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DescriptorPoolCreateFlags::empty(),
                max_sets: 50,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;

            // Create compute pipelines
            let compute_pipelines = ComputePipelines::new(&device, descriptor_set_layout)?;

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let total_memory = {
                let device_local_heap_index = (0..memory_properties.memory_type_count)
                    .find(|&i| {
                        let memory_type = memory_properties.memory_types[i as usize];
                        memory_type.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    })
                    .map(|i| memory_properties.memory_types[i as usize].heap_index)
                    .unwrap_or(0);
                
                memory_properties.memory_heaps[device_local_heap_index as usize].size
            };

            Ok(Self {
                entry,
                instance,
                device,
                physical_device,
                compute_queue,
                command_pool,
                queue_family_index,
                descriptor_pool,
                descriptor_set_layout,
                compute_pipelines,
                memory_tracker: MemoryTracker::new(total_memory),
            })
        }
    }

    unsafe fn create_instance(entry: &Entry) -> Result<Instance, Box<dyn std::error::Error>> {
        let app_name = CString::new("Compute App")?;
        let appinfo = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: app_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 0, 0),
            _marker: std::marker::PhantomData,
        };

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::InstanceCreateFlags::empty(),
            p_application_info: &appinfo,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
            _marker: std::marker::PhantomData,
        };

        Ok(unsafe { entry.create_instance(&create_info, None) }?)
    }

    pub fn total_memory(&self) -> u64 {
        unsafe {
            let memory_properties = self.instance.get_physical_device_memory_properties(self.physical_device);
            
            // Find the heap index that corresponds to device local (GPU) memory
            let device_local_heap_index = (0..memory_properties.memory_type_count)
                .find(|&i| {
                    let memory_type = memory_properties.memory_types[i as usize];
                    memory_type.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .map(|i| memory_properties.memory_types[i as usize].heap_index)
                .unwrap_or(0);
            
            // Get the size of the device local heap
            memory_properties.memory_heaps[device_local_heap_index as usize].size
        }
    }

    pub fn available_gpus() -> Result<Vec<GPUInfo>, VKMLEngineError> {
        unsafe {
            let entry = Entry::load()
                .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                
            let instance = Self::create_instance(&entry)
                .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
            
            let physical_devices = instance.enumerate_physical_devices()
                .map_err(|e| VKMLEngineError::VulkanLoadError(e.to_string()))?;
                
            // Create GPUInfo for each device and filter for compute support
            let mut gpu_infos: Vec<_> = physical_devices.iter()
                .enumerate()
                .map(|(idx, &device)| GPUInfo::new(&instance, device, idx))
                .filter(|info| info.has_compute)
                .collect();
            
            // Sort GPUs: discrete first, then by memory size
            gpu_infos.sort_by_key(|gpu| {
                (gpu.device_type != vk::PhysicalDeviceType::DISCRETE_GPU,
                    std::cmp::Reverse(gpu.total_memory))
            });
            
            instance.destroy_instance(None);
            Ok(gpu_infos)
        }
    }

    pub fn move_to_gpu_as_f32(&self, data: &[f32]) -> Result<GPUMemory, Box<dyn std::error::Error>> {
        let size_in_bytes = (data.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;

        unsafe {
            // Create buffer for f32 data
            let buffer_info = vk::BufferCreateInfo {
                s_type: vk::StructureType::BUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::BufferCreateFlags::empty(),
                size: size_in_bytes,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            let buffer = self.device.create_buffer(&buffer_info, None)?;
            let mem_requirements = self.device.get_buffer_memory_requirements(buffer);
            
            let memory_type = self.find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let alloc_info = vk::MemoryAllocateInfo {
                s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: mem_requirements.size,
                memory_type_index: memory_type,
                _marker: std::marker::PhantomData,
            };

            let memory = self.device.allocate_memory(&alloc_info, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;

            // Map memory and write data
            let data_ptr = self.device.map_memory(
                memory,
                0,
                size_in_bytes,
                vk::MemoryMapFlags::empty(),
            )? as *mut f32;

            // Copy the data
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                data_ptr,
                data.len()
            );
            
            self.device.unmap_memory(memory);

            // Create descriptor set
            let set_layouts = [self.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: ptr::null(),
                descriptor_pool: self.descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                _marker: std::marker::PhantomData,
            };

            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

            // Update descriptor set
            let buffer_info = vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: size_in_bytes,
            };

            let write_descriptor_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descriptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info,
                p_image_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device.update_descriptor_sets(&[write_descriptor_set], &[]);

            Ok(GPUMemory {
                buffer,
                memory,
                size: size_in_bytes,
                device: Arc::new(self.device.clone()),
                descriptor_set,
            })
        }
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        unsafe {
            let mem_properties = self.instance
                .get_physical_device_memory_properties(self.physical_device);

            for i in 0..mem_properties.memory_type_count {
                if (type_filter & (1 << i)) != 0 
                    && mem_properties.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
                {
                    return Ok(i);
                }
            }

            Err("Failed to find suitable memory type".into())
        }
    }

    fn execute_operation(
        &self,
        src1: &GPUMemory,
        src2: &GPUMemory,
        operation: GPUMemoryOperation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            // Create command buffer
            let alloc_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: ptr::null(),
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                _marker: std::marker::PhantomData,
            };
    
            let command_buffer = self.device.allocate_command_buffers(&alloc_info)?[0];
    
            // Begin command buffer
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: ptr::null(),
                _marker: std::marker::PhantomData,
            };
    
            self.device.begin_command_buffer(command_buffer, &begin_info)?;
    
            // Update descriptor set for src2 buffer
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: src2.buffer,
                offset: 0,
                range: src2.size,
            };
    
            let write_descriptor_set = vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: src1.descriptor_set,
                dst_binding: 1,  // Use binding 1 for second buffer
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &buffer_info,
                p_image_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
                _marker: std::marker::PhantomData,
            };
    
            self.device.update_descriptor_sets(&[write_descriptor_set], &[]);
    
            // Get and bind the specific pipeline for this operation
            let pipeline = self.compute_pipelines.get_pipeline(operation)
                .ok_or("Invalid operation")?;
    
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
    
            // Bind the descriptor set from src1 (which now has both buffers)
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipelines.get_layout(),
                0,
                &[src1.descriptor_set],
                &[],
            );
    
            // Calculate workgroup dispatch size
            let workgroup_size = 256;
            let num_workgroups = (src1.size + workgroup_size as u64 - 1) / workgroup_size as u64;
            
            self.device.cmd_dispatch(
                command_buffer,
                num_workgroups as u32,
                1,
                1,
            );
    
            // End and submit command buffer
            self.device.end_command_buffer(command_buffer)?;
    
            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &command_buffer,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
                _marker: std::marker::PhantomData,
            };
    
            self.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                vk::Fence::null(),
            )?;
    
            self.device.queue_wait_idle(self.compute_queue)?;
    
            // Cleanup
            self.device.free_command_buffers(self.command_pool, &[command_buffer]);
    
            Ok(())
        }
    }

    pub fn read_memory(&self, memory: &GPUMemory) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        unsafe {
            // Create command buffer for memory barrier
            let alloc_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                p_next: ptr::null(),
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                _marker: std::marker::PhantomData,
            };

            let command_buffer = self.device.allocate_command_buffers(&alloc_info)?[0];

            // Begin command buffer
            let begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                p_inheritance_info: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device.begin_command_buffer(command_buffer, &begin_info)?;

            // Add memory barrier to ensure compute shader has completed
            let barrier = vk::MemoryBarrier {
                s_type: vk::StructureType::MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::HOST_READ,
                _marker: std::marker::PhantomData,
            };

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );

            // End and submit command buffer
            self.device.end_command_buffer(command_buffer)?;

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &command_buffer,
                signal_semaphore_count: 0,
                p_signal_semaphores: ptr::null(),
                _marker: std::marker::PhantomData,
            };

            self.device.queue_submit(
                self.compute_queue,
                &[submit_info],
                vk::Fence::null(),
            )?;

            self.device.queue_wait_idle(self.compute_queue)?;

            // Now read the data
            let num_floats = (memory.size as usize) / std::mem::size_of::<f32>();
            let mut output_data = vec![0f32; num_floats];
            
            let data_ptr = self.device.map_memory(
                memory.memory,
                0,
                memory.size,
                vk::MemoryMapFlags::empty(),
            )? as *const f32;

            std::ptr::copy_nonoverlapping(
                data_ptr,
                output_data.as_mut_ptr(),
                num_floats
            );
            
            self.device.unmap_memory(memory.memory);

            // Cleanup
            self.device.free_command_buffers(self.command_pool, &[command_buffer]);

            Ok(output_data)
        }
    }

    pub fn add(&self, src1: &GPUMemory, src2: &GPUMemory) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_operation(src1, src2, GPUMemoryOperation::Add)
    }

    pub fn sub(&self, src1: &GPUMemory, src2: &GPUMemory) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_operation(src1, src2, GPUMemoryOperation::Subtract)
    }

    pub fn mul(&self, src1: &GPUMemory, src2: &GPUMemory) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_operation(src1, src2, GPUMemoryOperation::Multiply)
    }

    pub fn div(&self, src1: &GPUMemory, src2: &GPUMemory) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_operation(src1, src2, GPUMemoryOperation::Divide)
    }

    pub fn allocate_memory(&self, size: u64) -> Result<(), VKMLEngineError> {
        self.memory_tracker.allocate(size)
    }

    pub fn deallocate_memory(&self, size: u64) {
        self.memory_tracker.deallocate(size)
    }

    pub fn available_memory(&self) -> u64 {
        self.memory_tracker.get_available()
    }
}

impl Drop for GPUMemory {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl Drop for GPU {
    fn drop(&mut self) {
        unsafe {
            self.compute_pipelines.cleanup(&self.device);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}