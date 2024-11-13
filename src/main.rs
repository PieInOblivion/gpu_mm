use utils::{
    dataloader::{DataLoaderForImages, DatasetSplit},
    dataloader_config::DataLoaderConfig,
    dataloader_info::print_dataset_info,
    dataloader_iter::ParallelDataLoaderIterator,
};

mod utils;

use std::{sync::Arc, thread, time::Duration};

fn main() {
    let config = DataLoaderConfig {
        shuffle_seed: Some(727),
        batch_size: 64,
        train_ratio: 1.0,
        test_ratio: 0.0,
        drop_last: false,
        num_of_batch_prefetches: 4,
        ..Default::default()
    }
    .build()
    .unwrap();

    let dl = Arc::new(DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0", Some(config)).unwrap());

    print_dataset_info(&dl);
    // TODO: Fix return ordering. Currently loads well, but returns batches out of order
    for batch in dl.par_iter(DatasetSplit::Train) {
        println!();
        println!("BATCH NUM: {:?}", batch.batch_number);
        println!("BATCH LEN!: {:?}", batch.images_this_batch);
        println!("BATCH DATALEN!: {:?}", batch.image_data.len());
        println!("ADDR: {:p}", &batch.images_this_batch);
        thread::sleep(Duration::from_millis(4000));
    }
}
