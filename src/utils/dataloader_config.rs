pub struct DataLoaderConfig {
    pub threads: usize,
    pub batch_size: usize,
    pub batch_prefetch: bool,
    pub train_ratio: f32,
    pub test_ratio: f32,
    pub sort_dataset: bool,
    pub shuffle: bool,
    pub shuffle_seed: Option<u64>,
    pub drop_last: bool,
}

// config = DataLoaderConfig({})
impl DataLoaderConfig {
    pub fn new(
        threads: usize,
        batch_size: usize,
        batch_prefetch: usize,
        train_ratio: f32,
        test_ratio: f32,
        sort_dataset: bool,
        shuffle: bool,
        shuffle_seed: Option<u64>,
        drop_last: bool,
    ) {
    }
    pub fn new2(config: DataLoaderConfig) {}
}

// TODO: Look at rayon::ThreadPoolBuilder::build_global for thread count setting
impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            threads: num_cpus::get(),
            batch_size: 32,
            batch_prefetch: true,
            train_ratio: 0.8,
            test_ratio: 0.1,
            sort_dataset: false,
            shuffle: true,
            shuffle_seed: None,
            drop_last: true,
        }
    }
}
