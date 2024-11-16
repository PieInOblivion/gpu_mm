use utils::{
    dataloader_config::DataLoaderConfig,
    dataloader_for_images::{DataLoaderForImages, DatasetSplit},
    dataloader_info::print_dataset_info,
    dataloader_iter::MultithreadedDataLoaderIterator
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
        prefetch_threads: 4,
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
}
