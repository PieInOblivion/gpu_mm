use utils::{
    dataloader::{DataLoaderForImages, DatasetSplit, ImageBatches},
    dataloader_config::DataLoaderConfig,
    dataloader_info::print_dataset_info,
};

mod utils;

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

    let mut dl =
        DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0", Some(config)).unwrap();

    print_dataset_info(&dl);

    for batch_of_paths in dl.iter(DatasetSplit::Test) {
        dbg!(batch_of_paths);
    }

    let mut ib = ImageBatches::new(&dl);
    ib.load_raw_image_data(dl.next_batch_of_paths(DatasetSplit::Test, 1).unwrap());
    println!("{:?}", ib.images);
}
