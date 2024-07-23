use utils::{
    dataloader::{DataLoaderForImages, DatasetSplit, ImageBatches},
    dataloader_config::DataLoaderConfig,
};

mod utils;

// use cudarc::driver::result;
// use std::sync::{Arc, RwLock};

fn main() {
    let config = DataLoaderConfig {
        shuffle_seed: Some(727),
        batch_size: 1,
        ..Default::default()
    };

    let c2 = DataLoaderConfig::new2(Default::default());

    let mut dl =
        DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0", Some(config)).unwrap();
    dl.load_dataset().unwrap();
    dl.print_dataset_info();

    let mut ib = ImageBatches::new(&dl);
    ib.load_raw_image_data(dl.next_batch_of_paths(DatasetSplit::Test).0);
    println!("{:?}", ib.images);
}
