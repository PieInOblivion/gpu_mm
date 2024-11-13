use utils::{
    dataloader::{DataLoaderForImages, DatasetSplit},
    dataloader_config::DataLoaderConfig,
    dataloader_info::print_dataset_info,
    dataloader_iter::ParallelDataLoaderIterator,
};

mod utils;

use std::sync::Arc;

// use cudarc::driver::result;
// use std::sync::{Arc, RwLock};

fn main() {
    let config = DataLoaderConfig {
        shuffle_seed: Some(727),
        batch_size: 2,
        train_ratio: 1.0,
        test_ratio: 0.0,
        drop_last: false,
        ..Default::default()
    }
    .build()
    .unwrap();

    let dl = Arc::new(DataLoaderForImages::new("/home/lucas/Documents/test_images", Some(config)).unwrap());

    print_dataset_info(&dl);

    for batch in dl.par_iter(DatasetSplit::Train) {
        println!("BATCH LEN!: {:?}", batch.images_this_batch);
        println!("BATCH DATALEN!: {:?}", batch.image_data.len());
        println!("ADDR: {:p}", &batch.images_this_batch);
        println!("{:?}", batch.image_data);
        // Might not need the extra work, as the iterator return is always the same memory location? What do I need to do to make it change?
    }
}
