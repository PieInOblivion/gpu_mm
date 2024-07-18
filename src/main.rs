use utils::dataloader::{
    DataLoaderForImages, DataLoaderForImagesConfig, DatasetSplit, ImageBatches,
};

mod utils;

// use cudarc::driver::result;
// use std::sync::{Arc, RwLock};

fn main() {
    let mut config = DataLoaderForImagesConfig::default();
    config.shuffle_seed = Some(727);
    config.batch_size = 1;

    let mut dl =
        DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0", Some(config)).unwrap();
    dl.load_dataset().unwrap();
    dl.print_dataset_info();

    let mut ib = ImageBatches::new(&dl);
    ib.load_raw_image_data(dl.next_batch_of_paths(DatasetSplit::Test).0);
    println!("{:?}", ib.images);
}
