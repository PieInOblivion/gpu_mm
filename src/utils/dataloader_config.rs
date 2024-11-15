use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rayon::ThreadPoolBuilder;

use super::dataloader_error::DataLoaderError;

pub struct DataLoaderConfig {
    pub data_loading_threads: usize,
    pub num_of_batch_prefetches: usize,
    pub batch_size: usize,
    pub train_ratio: f32,
    pub test_ratio: f32,
    pub sort_dataset: bool,
    pub shuffle: bool,
    pub shuffle_seed: Option<u64>,
    pub rng: Option<Arc<Mutex<StdRng>>>,
    pub drop_last: bool,
}

impl DataLoaderConfig {
    pub fn build(mut self) -> Result<Self, DataLoaderError> {
        if self.data_loading_threads == 0 {
            self.data_loading_threads = num_cpus::get();
        }

        ThreadPoolBuilder::new()
            .num_threads(self.data_loading_threads)
            .build_global()?;

        check_split_ratios(self.train_ratio, self.test_ratio)?;

        Ok(self)
    }
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            data_loading_threads: num_cpus::get(),
            batch_size: 32,
            num_of_batch_prefetches: 1,
            train_ratio: 0.8,
            test_ratio: 0.1,
            sort_dataset: false,
            shuffle: true,
            shuffle_seed: None,
            rng: None,
            drop_last: true,
        }
    }
}

fn check_split_ratios(train_ratio: f32, test_ratio: f32) -> Result<(), DataLoaderError> {
    if train_ratio + test_ratio > 1.0 || train_ratio <= 0.0 || test_ratio < 0.0 {
        return Err(DataLoaderError::InvalidSplitRatios {
            train: train_ratio,
            test: test_ratio,
        });
    }
    Ok(())
}
