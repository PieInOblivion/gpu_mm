use utils::dataloader::DataLoaderForImages;

mod utils;

// use cudarc::driver::result;
// use std::sync::{Arc, RwLock};

fn main() {
    // dbg!(utils::structs::GPU::new(0));
    let mut dl = DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0").unwrap();
    // println!("{:?}", dl);
    // dl.opt_shuffle_seed = Some(110873);
    println!("{}", dl.dataset_size);
    // dl.opt_shuffle = false;
    println!("{:?}", dl.opt_shuffle_seed);
    dl.set_split_ratios(0.8, 0.1).unwrap();
    dl.split_dataset().unwrap();
    println!("{}", dl.train_dataset.len());
    println!("{}", dl.test_dataset.len());
    println!("{}", dl.val_dataset.len());
    println!("{:?}", dl.opt_shuffle_seed);
    println!("{}", dl.train_dataset[0]);
}
