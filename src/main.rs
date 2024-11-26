use utils::{
    dataloader_config::DataLoaderConfig, dataloader_for_images::{DataLoaderForImages, DatasetSplit}, dataloader_info::print_dataset_info, dataloader_iter::MultithreadedDataLoaderIterator, gpu::{GPU}, image_batch::ImageBatch
};

mod utils;
mod thread_pool;

fn main() {
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

    let mut imgb_test: ImageBatch = ImageBatch::new(dl.image_total_bytes_per_batch, dl.config.batch_size, dl.image_bytes_per_image);
    imgb_test.image_data = Box::new([0; 727]);
    imgb_test.image_data[1] = 6;

    for batch in dl.par_iter(DatasetSplit::Train) {
        println!();
        println!("BATCH NUM: {:?}", batch.batch_number);
        println!("BATCH LEN!: {:?}", batch.images_this_batch);
        println!("BATCH DATALEN!: {:?}", batch.image_data.len());
        println!("ADDR: {:p}", &batch.images_this_batch);
        //thread::sleep(Duration::from_millis(4000));
    }

    // Initialize GPU
    let gpu = GPU::new(0).unwrap(); // Use first available GPU
    let gpu_mem1 = gpu.move_to_gpu(&imgb_test).unwrap();
    let gpu_mem2 = gpu.move_to_gpu(&imgb_test).unwrap();
    gpu_mem1.multiply(&gpu_mem2).unwrap();
    // Multiplied the wrong indicies...
    print!("{:?}", gpu_mem1.read_to_batch().unwrap().image_data)
}
