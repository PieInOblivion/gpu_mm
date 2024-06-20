use utils::dataloader::DataLoaderForImages;

mod utils;

// use cudarc::driver::result;
// use std::sync::{Arc, RwLock};

fn main() {
    // dbg!(utils::structs::GPU::new(0));
    let mut dl = DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0").unwrap();
    // println!("{:?}", dl);
    dl.opt_shuffle_seed = Some(727);
    println!("{}", dl.dataset_size);
    println!("{:?}", dl.opt_shuffle_seed);
    dl.opt_train_size = 0.5;
    dl.opt_test_size = 0.3;
    dl.split_dataset().unwrap();
    println!("{}", dl.train_dataset.len());
    println!("{}", dl.test_dataset.len());
    println!("{}", dl.val_dataset.len());
    println!("{:?}", dl.opt_shuffle_seed);
    println!("{}", dl.train_dataset[0]);
}
