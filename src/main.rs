use utils::{
    dataloader::{DataLoaderForImages, DatasetSplit, ImageBatch},
    dataloader_config::DataLoaderConfig,
    dataloader_info::print_dataset_info,
    dataloader_iter::ParallelBufferIterator,
};

mod utils;

use std::thread;

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

    let dl =
        DataLoaderForImages::new("/home/lucas/Documents/mnist_png/test/0", Some(config)).unwrap();

    print_dataset_info(&dl);

    for batch_of_paths in dl.iter(DatasetSplit::Test) {
        dbg!(batch_of_paths);
    }

    let mut ib = ImageBatch::new(&dl);
    ib.load_raw_image_data(dl.next_batch_of_paths(DatasetSplit::Test, 1).unwrap());
    println!("{:?}", ib.images);

    let mut counter = 0;
    let iter = ParallelBufferIterator::new(move || {
        if counter >= 50 {
            None
        } else {
            // Simulate some loading time
            thread::sleep(std::time::Duration::from_millis(1000));
            let batch: Vec<i32> = (counter..counter + 10).collect();
            counter += 10;
            Some(batch)
        }
    });

    for batch in iter {
        println!("Processing batch: {:?}", batch);
        // Simulate some processing time
        // thread::sleep(std::time::Duration::from_millis(50));
    }
}
