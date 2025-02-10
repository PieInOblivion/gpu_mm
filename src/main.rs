mod gpu;

use compute::compute_manager::ComputeManager;
use dataloader::{config::DataLoaderConfig, data_batch::DataBatch, dataloader::{DataLoader, DatasetSplit}, for_imagesdir::DirectoryImageLoader, par_iter::MultithreadedDataLoaderIterator};
use gpu::vk_gpu::GPU;

use model::{layer_type::LayerType, model::ModelDesc, weight_init::WeightInit};
use thread_pool::thread_pool::ThreadPool;

mod thread_pool;

mod compute;

mod model;

mod dataloader;

/* Design descisions and some TODOs
    Current proof of concept implementation of image loader stores all file names in memory
        Raw filesystems (most) don't store file counts and aren't sorted, so we to give users that option for replicatability
        Direct FS read into per batch means end files can never be in the first batch
        Can implement csv and other formats in future to support
        Raw binary support wanted in future

    Thread pool is currently created once. Single threaded usage then means creating a whole pool of one worker.
        Downside is that tasks are then loaded in advance which requires more memory than if the program just ran without a work queue
    Will need to implement threadpool as an option in future

    Thread pool batching also only submits the batch to begin work after generating the whole batch
        This could maybe benefit from an adjustment that flushes to the queue ever x amount instead of generating -> pushing -> working sequentially

    Currently GPU is assumed to have all memory free
        Will use VK_EXT_memory_budget in the future as it seems to be the most commonly implemented extension
        Final implementation will not adjust live, will only keep it's own usage and initial usage from other processes
            With adjustable configuration for threshold, (95% of free, etc)
    
    GPU checks filters by compute capability. In future might be able to use GPUs without compute flag?

    GPU to GPU movement will remain through the cpu for a while
        Need to investigate vulkan device pools, likely want to keep it indivual for more control?
        Does VK have a shared memory pool extension?

    Model, Layer, Tensor etc will act as only descriptors, so that compute manager is able to handle all data and memory
        This will also allow for other compute managers to be used in the future

    ImageBatch to f32 function assumes images to be stored in little endian

    Current GPU memory requirement calculations don't account for allocation overhead. Should be fine for now with a safe memory threshold set
    Also doesn't account for it's own cpu memory requirements, just a models

    Current compute manager stores all model sequentially in memory as to not store small layers on other devices and cause uneeded through cpu transfers

    Current implementation will send a gpu command and wait for it
        In future we can send multiple commands to the queue at once and wait for all to finish. Maybe using our threadpool? Or vulkan native solution
*/


fn main() {
    // Standard implementation, create one threadpool and share it.
    // Otherwise structs that require a threadpool will create their own.
    let thread_pool = ThreadPool::new();

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

    let dl = DirectoryImageLoader::new("/home/lucas/Documents/mnist_png/test/0", Some(config), thread_pool.clone()).unwrap();

    //let dl = DataLoaderForImages::new_arc("/home/lucas/Documents/mnist_png/test/0", Some(config)).unwrap();

    dataloader::info::print_dataset_info(&dl);

    // Currently the final partial batch has 0 as it's values after the valid image data as intended
    // As iterator reuses the struct. But since we have a set size array it must keep the size, hence partial batches have extra 0s, which works out for us
    // Need to not move DataLoader value for the iteration, and try not to use an arc
    for batch in dl.par_iter(DatasetSplit::Train) {
        println!();
        println!("BATCH NUM: {:?}", batch.batch_number);
        println!("BATCH LEN!: {:?}", batch.samples_in_batch);
        println!("BATCH DATALEN!: {:?}", batch.data.len());
        println!("ADDR: {:p}", &batch.data);
    }

    // - - - - GPU testing - - - -
    // We can interact with GPU instances to test, but all models should use a compute_manager instead
    // Own scope so GPU is cleared between testing areas
    // Drop GPU works properly
    {
        let mut data_test1 = DataBatch {
            data: vec![10, 20, 30, 15, 20, 30, 15, 10, 10, 20, 30, 15, 20, 30, 15, 10].into_boxed_slice(),
            samples_in_batch: 0,
            bytes_per_sample: 0,
            format: dataloader::dataloader::SourceFormat::U8,
            labels: None,
            batch_number: 0,
        };
        
        let mut data_test2 = DataBatch {
            data: vec![1, 2, 3, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1].into_boxed_slice(),
            samples_in_batch: 0,
            bytes_per_sample: 0,
            format: dataloader::dataloader::SourceFormat::U8,
            labels: None,
            batch_number: 0,
        };

        // Initialize GPU
        // NOTE: All GPU computations are f32 for now
        let gpu = GPU::new(0).unwrap();
        let gpu_mem1 = gpu.move_to_gpu_as_f32(&data_test1.to_f32()).unwrap();
        let gpu_mem2 = gpu.move_to_gpu_as_f32(&data_test2.to_f32()).unwrap();
        gpu.add(&gpu_mem1, &gpu_mem2).unwrap();

        println!("{:?}", gpu.read_memory(&gpu_mem1).unwrap());
        println!("{:?}", gpu_mem2.read_memory().unwrap());
    }

    // Turns out NVIDIA vulkan can eat .spv that is still text
    // while intel needs it validated and compiled...
    // Doing this at runtime requires external c++ libraries so this code will just have to ship with manually validated shaders

    // - - - - Model and Compute Manager testing - - - -

    println!("{:?}", GPU::available_gpus());

    //let mut m = ModelDesc::new(64);
    let mut m = ModelDesc::new_with(64, WeightInit::He);

    m.add_layer(LayerType::InputBuffer { features: 785, track_gradients: false });

    m.add_layer(LayerType::linear(785, 512));

    m.add_layer(LayerType::ReLU);

    m.add_layer(LayerType::linear(512, 256));

    m.add_layer(LayerType::ReLU);

    m.add_layers(vec![
        LayerType::linear(256, 64),
        LayerType::ReLU,
        LayerType::linear(64, 1)
    ]);

    let cm = ComputeManager::new(m, thread_pool.clone()).unwrap();
    
    //cm.move_model_to_cpu().unwrap();
    
    cm.print_model_stats();

    cm.print_layer_values(2);
/*
    {
        println!("Starting GPU memory tracking verification...\n");
    
        let mut gpu = GPU::new(1).unwrap();
        let total_memory = gpu.total_memory();
        let initial_available = gpu.available_memory();
        
        println!("Total GPU memory: {} MB", total_memory / 1024 / 1024);
        println!("Available memory according to tracker: {} MB\n", initial_available / 1024 / 1024);
    
        // Binary search for maximum allocation size
        let mut low = 1024 * 1024; // 1MB in bytes
        let mut high = initial_available;
        let mut max_successful = 0;
    
        while low <= high {
            let mid = low + (high - low) / 2;
            let num_floats = mid as usize / std::mem::size_of::<f32>();
            print!("Trying to allocate {} MB... ", mid / 1024 / 1024);
    
            let test_data: Vec<f32> = vec![1.0; num_floats];
    
            match gpu.allocate_memory(mid) {
                Ok(()) => {
                    match gpu.move_to_gpu_as_f32(&test_data) {
                        Ok(gpu_mem) => {
                            println!("Success!");
                            max_successful = mid;
                            // Immediately free the memory for next iteration
                            drop(gpu_mem);
                            gpu.deallocate_memory(mid);
                            // Try a larger size
                            low = mid + 1024 * 1024; // Increment by 1MB for finer granularity
                        },
                        Err(e) => {
                            println!("Failed to move to GPU: {}", e);
                            gpu.deallocate_memory(mid);
                            high = mid - 1024 * 1024;
                        }
                    }
                },
                Err(e) => {
                    println!("Memory tracker prevented allocation: {}", e);
                    high = mid - 1024 * 1024;
                }
            }
        }
        
        println!("\nResults:");
        println!("Maximum single allocation: {} MB", max_successful / 1024 / 1024);
        println!("Initial available memory: {} MB", initial_available / 1024 / 1024);
        println!("Final available memory: {} MB", gpu.available_memory() / 1024 / 1024);
        
        // Verify that our tracking matches reality
        let accuracy = (max_successful as f64 / initial_available as f64) * 100.0;
        println!("\nMemory Tracker Accuracy:");
        println!("Our tracker predicted: {} MB", initial_available / 1024 / 1024);
        println!("Actually allocatable: {} MB", max_successful / 1024 / 1024);
        println!("Tracking accuracy: {:.1}%", accuracy);
        
        if accuracy < 90.0 {
            println!("\nWARNING: Memory tracker might be too conservative!");
        } else if accuracy > 100.0 {
            println!("\nWARNING: Memory tracker might be too optimistic!");
        } else {
            println!("\nMemory tracker appears to be reasonably accurate.");
        }
    
        println!("\nTest completed successfully");
    }*/
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