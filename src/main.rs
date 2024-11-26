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

    for batch in dl.par_iter(DatasetSplit::Train) {
        println!();
        println!("BATCH NUM: {:?}", batch.batch_number);
        println!("BATCH LEN!: {:?}", batch.images_this_batch);
        println!("BATCH DATALEN!: {:?}", batch.image_data.len());
        println!("ADDR: {:p}", &batch.images_this_batch);
        //thread::sleep(Duration::from_millis(4000));
    }

    // GPU testing
    let mut imgb_test1 = ImageBatch::new(16, 0, 0);
    imgb_test1.image_data[..16].copy_from_slice(&[10, 20, 30, 15, 20, 30, 15, 10, 10, 20, 30, 15, 20, 30, 15, 10]);

    let mut imgb_test2 = ImageBatch::new(16, 0, 0);
    imgb_test2.image_data[..16].copy_from_slice(&[1, 2, 3, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 1]);

    // Initialize GPU
    // NOTE: All GPU computations are f32 for now
    let gpu = GPU::new(0, &dl.image_color_type).unwrap();
    let gpu_mem1 = gpu.move_to_gpu_as_f32(&imgb_test1).unwrap();
    let gpu_mem2 = gpu.move_to_gpu_as_f32(&imgb_test2).unwrap();
    gpu_mem1.add(&gpu_mem2).unwrap();

    println!("{:?}", gpu_mem1.read_data().unwrap());
}