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
        batch_size: 1,
        ..Default::default()
    }
    .build()
    .unwrap();

    let dl = Arc::new(DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0", Some(config)).unwrap());

    print_dataset_info(&dl);

    for batch in dl.par_iter(DatasetSplit::Validation) {
        dbg!(batch);
    }
}
