mod gpu;

use compute::{compute_manager::ComputeManager, layer::LayerType, model::Model, tensor::Tensor};
use gpu::vk_gpu::GPU;

mod utils;
use utils::{
    dataloader_config::DataLoaderConfig,
    dataloader_for_images::{DataLoaderForImages, DatasetSplit},
    dataloader_info::print_dataset_info,
    dataloader_iter::MultithreadedDataLoaderIterator,
    image_batch::ImageBatch
};

mod thread_pool;

mod compute;

fn main() {
    // - - - - Data loader and parralel iterator testing - - - -
    let config = DataLoaderConfig {
        shuffle_seed: Some(727),
        batch_size: 64,
        train_ratio: 1.0,
        test_ratio: 0.0,
        drop_last: false,
        prefetch_count: 4,
        ..Default::default()
    }
    .build()
    .unwrap();

    let dl = DataLoaderForImages::new_arc("/home/lucas/Documents/mnist_png/test/0", Some(config)).unwrap();

    print_dataset_info(&dl);

    // Currently the final partial batch has 0 as it's values after the image data as intended
    for batch in dl.par_iter(DatasetSplit::Train) {
        println!();
        println!("BATCH NUM: {:?}", batch.batch_number);
        println!("BATCH LEN!: {:?}", batch.images_this_batch);
        println!("BATCH DATALEN!: {:?}", batch.image_data.len());
        println!("ADDR: {:p}", &batch.images_this_batch);
    }

    
    // - - - - GPU testing - - - -
    // We can interact with GPU instances to test, but all models should use a compute_manager instead
    // Own scope so GPU is cleared between testing areas
    // Drop GPU works properly
    {
        let mut imgb_test1 = ImageBatch::new(16, 0, 0, image::ColorType::L8, 0);
        imgb_test1.image_data[..16].copy_from_slice(&[10, 20, 30, 15, 20, 30, 15, 10, 10, 20, 30, 15, 20, 30, 15, 10]);

        let mut imgb_test2 = ImageBatch::new(16, 0, 0, image::ColorType::L8, 0);
        imgb_test2.image_data[..16].copy_from_slice(&[1, 2, 3, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1]);

        // Initialize GPU
        // NOTE: All GPU computations are f32 for now
        let gpu = GPU::new(0).unwrap();
        println!("{:?}", gpu.total_memory());
        let gpu_mem1 = gpu.move_to_gpu_as_f32(&imgb_test1.to_f32()).unwrap();
        let gpu_mem2 = gpu.move_to_gpu_as_f32(&imgb_test2.to_f32()).unwrap();
        gpu.add(&gpu_mem1, &gpu_mem2);

        println!("{:?}", gpu.read_memory(&gpu_mem1).unwrap());
    }

    // Turns out NVIDIA vulkan can eat .spv that is still text
    // while intel needs it validated and compiled...
    // Doing this at runtime requires external c++ libraries so this code will just have to ship with manually validated shaders

    // - - - - Model and Compute Manager testing - - - -
    let cm = ComputeManager::new().unwrap();
    let cm2 = ComputeManager::new_with(vec![GPU::new(1).unwrap()], None);
    println!("{:?}", GPU::available_gpus());

    {
        let mut imgb_test1 = ImageBatch::new(16, 0, 0, image::ColorType::L8, 0);
        imgb_test1.image_data[..16].copy_from_slice(&[10, 20, 30, 15, 20, 30, 15, 10, 10, 20, 30, 15, 20, 30, 15, 10]);

        let mut imgb_test2 = ImageBatch::new(16, 0, 0, image::ColorType::L8, 0);
        imgb_test2.image_data[..16].copy_from_slice(&[1, 2, 3, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1]);

        // Initialize GPU
        // NOTE: All GPU computations are f32 for now
        let gpu = GPU::new(1).unwrap();
        let gpu_mem1 = gpu.move_to_gpu_as_f32(&imgb_test1.to_f32()).unwrap();
        let gpu_mem2 = gpu.move_to_gpu_as_f32(&imgb_test2.to_f32()).unwrap();
        gpu.add(&gpu_mem1, &gpu_mem2);

        println!("{:?}", gpu.read_memory(&gpu_mem1).unwrap());
    }

    /*
    // Initialize compute manager with available GPUs
    let gpus = ComputeManager::available_gpus().unwrap();
    let cpu_memory_limit = 8 * 1024 * 1024 * 1024; // 8GB
    let compute = ComputeManager::new(
        gpus,
        cpu_memory_limit,
        ColorType::Rgb32F
    );

    // Create model with compute manager
    let mut model = Model::new(compute);

    // Add individual layer
    model.add_layer(LayerType::Linear { 
        in_features: 784, 
        out_features: 128 
    }).unwrap();
    model.add_layer(LayerType::ReLU).unwrap();

    // Add multiple layers at once
    model.add_layers(vec![
        LayerType::Linear { in_features: 128, out_features: 64 },
        LayerType::ReLU,
        LayerType::Linear { in_features: 64, out_features: 10 },
    ]).unwrap();

    // Create input tensor and perform forward pass
    let input = Tensor::new(vec![1, 784]).unwrap();
    let output = model.forward(input).unwrap();
    */
}