mod gpu;

use compute::compute_manager::ComputeManager;
use dataloader::{config::DataLoaderConfig, data_batch::DataBatch, dataloader::DatasetSplit, for_imagesdir::DirectoryImageLoader, par_iter::MultithreadedDataLoaderIterator};
use gpu::vk_gpu::GPU;

use layer::factory::Layers;
use model::{graph_model::GraphModel, layer_connection::LayerConnection};
use thread_pool::thread_pool::ThreadPool;

mod thread_pool;

mod compute;

mod model;

mod dataloader;

mod layer;

mod tensor;

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

    let mut model = GraphModel::new(64);

   let input1_id = model.add_layer(Layers::input_buffer(100));
   let input2_id = model.add_layer(Layers::input_buffer(50));
   
   let fc1_id = model.add_layer_with(
       model.next_available_id(),
       Layers::linear(100, 64),
       vec![LayerConnection::DefaultOutput(input1_id)],
       None
   );
   
   let relu1_id = model.add_layer_with(
       model.next_available_id(),
       Layers::relu(),
       vec![LayerConnection::DefaultOutput(fc1_id)],
       None
   );
   
   let fc2_id = model.add_layer_with(
       model.next_available_id(),
       Layers::linear(50, 32),
       vec![LayerConnection::DefaultOutput(input2_id)],
       None
   );
   
   let relu2_id = model.add_layer_with(
       model.next_available_id(),
       Layers::relu(),
       vec![LayerConnection::DefaultOutput(fc2_id)],
       None
   );

   let fc3_id = model.add_layer_with(
       model.next_available_id(),
       Layers::linear(100, 32),
       vec![LayerConnection::DefaultOutput(input1_id)],
       None
   );
   
   let relu3_id = model.add_layer_with(
       model.next_available_id(),
       Layers::relu(),
       vec![LayerConnection::DefaultOutput(fc3_id)],
       None
   );
   
   let add_id = model.add_layer_with(
       model.next_available_id(),
       Layers::add(),
       vec![
           LayerConnection::DefaultOutput(relu2_id),
           LayerConnection::DefaultOutput(relu3_id)
       ],
       None
   );
   
   let concat_id = model.add_layer_with(
       model.next_available_id(),
       Layers::concat(1),
       vec![
           LayerConnection::DefaultOutput(relu1_id),
           LayerConnection::DefaultOutput(add_id)
       ],
       None
   );
   
   let final_fc_id = model.add_layer_with(
       model.next_available_id(),
       Layers::linear(96, 10),
       vec![LayerConnection::DefaultOutput(concat_id)],
       None
   );

   let secondary_fc_id = model.add_layer_with(
       model.next_available_id(),
       Layers::linear(32, 5),
       vec![LayerConnection::DefaultOutput(add_id)],
       None
   );

    let cm = ComputeManager::new(model, thread_pool.clone()).unwrap();
    
    cm.print_model_stats();

    cm.print_layer_values(2).unwrap();
}