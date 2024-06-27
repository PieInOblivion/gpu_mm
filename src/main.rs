use utils::dataloader::DataLoaderForImages;

mod utils;

// use cudarc::driver::result;
// use std::sync::{Arc, RwLock};

fn main() {
    // dbg!(utils::structs::GPU::new(0));
    let mut dl = DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0").unwrap();
    // println!("{:?}", dl);
    //dl.opt_shuffle_seed = Some(110873);
    //dl.opt_shuffle = false;
    println!("{:?}", dl.opt_shuffle_seed);
    //dl.set_split_ratios(0.8, 0.1).unwrap();
    //dl.split_dataset_basic().unwrap();
    dl.split_dataset().unwrap();
    println!("Final distribution:");
    println!("Train: {}", dl.train_dataset.len());
    println!("Test: {}", dl.test_dataset.len());
    println!("Validation: {}", dl.val_dataset.len());
    println!("{:?}", dl.opt_shuffle_seed);
    println!("{:?}", dl.train_dataset.get(..5));
}
