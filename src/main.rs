use utils::dataloader::{DataLoaderError, DataLoaderForImages, ImageBatches};

mod utils;

// use cudarc::driver::result;
// use std::sync::{Arc, RwLock};

fn main() {
    let mut dl = DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0").unwrap();
    // dl.opt_shuffle_seed = Some(5);
    // dl.opt_shuffle = false;
    dl.load_dataset().unwrap();
    dl.print_dataset_info();

    let mut ib = ImageBatches::new(&dl);
}
