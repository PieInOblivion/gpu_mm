use image::{self, ColorType};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

// TODO: Investigate if hashmap solution can be done with random non overlapping number generation,
//       as to not require hashmap + vec<usize> in memory

#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid dataset split ratios. Train: {train}, Test: {test}")]
    InvalidSplitRatios { train: f32, test: f32 },
}

pub struct DataLoaderForImages {
    dir: PathBuf,
    pub dataset_size: usize,
    pub largest_width: u32,
    pub largest_height: u32,
    pub largest_depth: ColorType,
    pub opt_threads: u32,
    pub opt_batch_size: usize,
    pub opt_train_ratio: f32,
    pub opt_test_ratio: f32,
    pub opt_shuffle: bool,
    pub opt_shuffle_seed: Option<u64>,
    pub opt_drop_last: bool,
    pub scanned_largest: bool,
    pub train_dataset: Vec<String>,
    pub test_dataset: Vec<String>,
    pub val_dataset: Vec<String>,
    pub valid_extensions: Vec<String>,
}

impl DataLoaderForImages {
    pub fn new(dir: &str) -> Result<Self, DataLoaderError> {
        let path = Path::new(dir);
        if !path.exists() {
            return Err(DataLoaderError::DirectoryNotFound(dir.to_string()));
        }

        Ok(DataLoaderForImages {
            dir: path.to_owned(),
            dataset_size: 0,
            largest_width: 0,
            largest_height: 0,
            largest_depth: ColorType::L8,
            opt_threads: 0,
            opt_batch_size: 32,
            opt_train_ratio: 0.8,
            opt_test_ratio: 0.1,
            opt_shuffle: true,
            opt_shuffle_seed: None,
            opt_drop_last: false,
            scanned_largest: false,
            train_dataset: Vec::new(),
            test_dataset: Vec::new(),
            val_dataset: Vec::new(),
            valid_extensions: vec!["png".to_string(), "jpg".to_string(), "jpeg".to_string()],
        })
    }

    pub fn set_split_ratios(
        &mut self,
        train_ratio: f32,
        test_ratio: f32,
    ) -> Result<(), DataLoaderError> {
        if train_ratio + test_ratio > 1.0 || train_ratio <= 0.0 || test_ratio < 0.0 {
            return Err(DataLoaderError::InvalidSplitRatios {
                train: train_ratio,
                test: test_ratio,
            });
        }
        self.opt_train_ratio = train_ratio;
        self.opt_test_ratio = test_ratio;
        Ok(())
    }

    fn count_valid_images(&self) -> Result<usize, DataLoaderError> {
        Ok(fs::read_dir(&self.dir)?
            .filter_map(Result::ok)
            .filter(|entry| self.is_valid_extension(entry.path()))
            .count())
    }

    fn is_valid_extension(&self, path: PathBuf) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| self.valid_extensions.contains(&ext.to_lowercase()))
            .unwrap_or(false)
    }

    pub fn split_dataset(&mut self) -> Result<(), DataLoaderError> {
        // First pass: count valid images
        self.dataset_size = self.count_valid_images()?;
        println!("self.dataset_size {}", self.dataset_size);

        // Calculate target sizes
        let target_train_size = (self.dataset_size as f32 * self.opt_train_ratio).round() as usize;
        let target_test_size = (self.dataset_size as f32 * self.opt_test_ratio).round() as usize;
        let target_val_size = self.dataset_size - target_train_size - target_test_size;
        println!("Target distribution:");
        println!("Train: {target_train_size}");
        println!("Test: {target_test_size}");
        println!("Validation: {target_val_size}");

        // Pre-allocate vectors
        self.train_dataset = Vec::with_capacity(target_train_size);
        self.test_dataset = Vec::with_capacity(target_test_size);
        self.val_dataset = Vec::with_capacity(target_val_size);

        // Initialize RNG if shuffling
        let mut rng = if self.opt_shuffle {
            if self.opt_shuffle_seed.is_none() {
                self.opt_shuffle_seed = Some(rand::thread_rng().gen());
            }
            Some(StdRng::seed_from_u64(self.opt_shuffle_seed.unwrap()))
        } else {
            None
        };

        // Second pass: distribute images
        let mut current_train_size = 0;
        let mut current_test_size = 0;
        let mut current_val_size = 0;

        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();

            if self.is_valid_extension(path.clone()) {
                let remaining =
                    self.dataset_size - (current_train_size + current_test_size + current_val_size);
                let remaining_train = target_train_size.saturating_sub(current_train_size);
                let remaining_test = target_test_size.saturating_sub(current_test_size);

                let prob_train = remaining_train as f32 / remaining as f32;
                let prob_test = remaining_test as f32 / remaining as f32;

                // If not shuffling, deterministically assign
                let rand_val = rng.as_mut().map_or(0.0, |r| r.gen::<f32>());

                let file_path = path.to_str().unwrap().to_string();

                if rand_val < prob_train {
                    self.train_dataset.push(file_path);
                    current_train_size += 1;
                } else if rand_val < prob_train + prob_test {
                    self.test_dataset.push(file_path);
                    current_test_size += 1;
                } else {
                    self.val_dataset.push(file_path);
                    current_val_size += 1;
                }
            }
        }

        // Shuffle datasets if opt_shuffle is true
        if let Some(mut shuffle_rng) = rng.as_mut() {
            self.train_dataset.shuffle(&mut shuffle_rng);
            self.test_dataset.shuffle(&mut shuffle_rng);
            self.val_dataset.shuffle(&mut shuffle_rng);
        }

        Ok(())
    }

    pub fn split_dataset_basic(&mut self) -> Result<(), DataLoaderError> {
        // First pass: count valid images
        self.dataset_size = self.count_valid_images()?;
        println!("self.dataset_size {}", self.dataset_size);

        // Calculate target sizes
        let target_train_size = (self.dataset_size as f32 * self.opt_train_ratio).round() as usize;
        let target_test_size = (self.dataset_size as f32 * self.opt_test_ratio).round() as usize;
        let target_val_size = self.dataset_size - target_train_size - target_test_size;
        println!("Target distribution:");
        println!("Train: {target_train_size}");
        println!("Test: {target_test_size}");
        println!("Validation: {target_val_size}");

        // Pre-allocate vectors
        self.train_dataset = Vec::with_capacity(target_train_size);
        self.test_dataset = Vec::with_capacity(target_test_size);
        self.val_dataset = Vec::with_capacity(target_val_size);

        // Initialize RNG
        let mut rng = if self.opt_shuffle {
            if self.opt_shuffle_seed.is_none() {
                self.opt_shuffle_seed = Some(rand::thread_rng().gen());
            }
            Some(StdRng::seed_from_u64(self.opt_shuffle_seed.unwrap()))
        } else {
            None
        };

        // Collect all images
        let mut all_files: Vec<String> = Vec::with_capacity(self.dataset_size);
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if self.is_valid_extension(path.clone()) {
                all_files.push(path.to_str().unwrap().to_string());
            }
        }

        // Shuffle datasets if opt_shuffle is true
        if let Some(mut rng) = rng.as_mut() {
            all_files.shuffle(&mut rng);
        }

        self.train_dataset
            .extend(all_files.drain(..target_train_size));
        self.test_dataset
            .extend(all_files.drain(..target_test_size));
        self.val_dataset.extend(all_files.drain(..));

        Ok(())
    }
}
