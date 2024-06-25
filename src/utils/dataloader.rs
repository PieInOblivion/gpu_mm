use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use image::{self, ColorType};

// TODO: Still doesn't always have perfect dataset bin final counts
// TODO: While the shuffle options work perfectly, the final order of data bins
//       seems to not shuffle properly
// TODO: Inspect an adaptive ratio method
// TODO: Attempt a string Arc method

#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),
    #[error("Not enough valid images in the directory. Required: {required}, Found: {found}")]
    InsufficientImages { required: usize, found: usize },
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid dataset split ratios. Train: {train}, Test: {test}")]
    InvalidSplitRatios { train: f32, test: f32 },
}

#[derive(Debug)]
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
    scanned_largest: bool,
    pub train_dataset: Vec<String>,
    pub test_dataset: Vec<String>,
    pub val_dataset: Vec<String>,
    valid_extensions: Vec<String>,
}

impl DataLoaderForImages {
    pub fn new(dir: &str) -> Result<Self, DataLoaderError> {
        let path = Path::new(dir);
        if !path.exists() {
            return Err(DataLoaderError::DirectoryNotFound(dir.to_string()));
        }

        let mut new_loader = DataLoaderForImages {
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
        };

        new_loader.check_for_largest_image()?;

        Ok(new_loader)
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

    fn check_for_largest_image(&mut self) -> Result<(), DataLoaderError> {
        if self.scanned_largest {
            return Ok(());
        }

        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if self.is_valid_extension(path.clone()) {
                if let Ok(img) = image::open(&path) {
                    let depth = img.color();
                    self.largest_width = self.largest_width.max(img.width());
                    self.largest_height = self.largest_height.max(img.height());

                    if depth.bits_per_pixel() > self.largest_depth.bits_per_pixel() {
                        self.largest_depth = depth;
                    }

                    self.dataset_size += 1;
                }
            }
        }

        self.scanned_largest = true;
        Ok(())
    }

    pub fn split_dataset(&mut self) -> Result<(), DataLoaderError> {
        let train_ratio = self.opt_train_ratio;
        let test_ratio = self.opt_test_ratio;

        let mut rng = if self.opt_shuffle {
            if self.opt_shuffle_seed.is_none() {
                self.opt_shuffle_seed = Some(rand::thread_rng().gen());
            }
            Some(StdRng::seed_from_u64(self.opt_shuffle_seed.unwrap()))
        } else {
            None
        };

        // Pre-allocate vectors based on expected sizes
        let expected_total = self.count_valid_images()?;
        let mut train_set =
            Vec::with_capacity((expected_total as f32 * train_ratio).ceil() as usize);
        let mut test_set = Vec::with_capacity((expected_total as f32 * test_ratio).ceil() as usize);
        let mut val_set = Vec::with_capacity(
            (expected_total as f32 * (1.0 - train_ratio - test_ratio)).ceil() as usize,
        );

        let mut total_images = 0;

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_file() && self.is_valid_extension(entry.path()) {
                    if let Some(filename) = entry.file_name().to_str() {
                        total_images += 1;
                        let random_value = rng.as_mut().map_or(0.0, |rng| rng.gen::<f32>());

                        if random_value < train_ratio {
                            train_set.push(filename.to_string());
                        } else if random_value < train_ratio + test_ratio {
                            test_set.push(filename.to_string());
                        } else {
                            val_set.push(filename.to_string());
                        }
                    }
                }
            }
        }

        if total_images < self.dataset_size {
            return Err(DataLoaderError::InsufficientImages {
                required: self.dataset_size,
                found: total_images,
            });
        }

        self.dataset_size = total_images;

        // Post-process the datasets
        self.post_process_datasets(&mut train_set, &mut test_set, &mut val_set, rng.as_mut());

        // Resize vectors to their actual sizes
        train_set.shrink_to_fit();
        test_set.shrink_to_fit();
        val_set.shrink_to_fit();

        self.train_dataset = train_set;
        self.test_dataset = test_set;
        self.val_dataset = val_set;

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

    fn post_process_datasets(
        &self,
        train_set: &mut Vec<String>,
        test_set: &mut Vec<String>,
        val_set: &mut Vec<String>,
        mut rng: Option<&mut StdRng>,
    ) {
        let total_images = train_set.len() + test_set.len() + val_set.len();
        let target_train_size = (total_images as f32 * self.opt_train_ratio).round() as usize;
        let target_test_size = (total_images as f32 * self.opt_test_ratio).round() as usize;
        let target_val_size = total_images - target_train_size - target_test_size;

        // Ensure each set has at least one item if possible
        if total_images >= 3 {
            self.ensure_non_empty(train_set, test_set, val_set);
        }

        // Adjust sizes to match target sizes
        self.adjust_set_size(train_set, target_train_size, test_set, val_set, &mut rng);
        self.adjust_set_size(test_set, target_test_size, train_set, val_set, &mut rng);
        self.adjust_set_size(val_set, target_val_size, train_set, test_set, &mut rng);
    }

    fn ensure_non_empty(
        &self,
        set1: &mut Vec<String>,
        set2: &mut Vec<String>,
        set3: &mut Vec<String>,
    ) {
        if set1.is_empty() && !set2.is_empty() {
            set1.push(set2.pop().unwrap());
        } else if set1.is_empty() && !set3.is_empty() {
            set1.push(set3.pop().unwrap());
        }
        if set2.is_empty() && !set1.is_empty() {
            set2.push(set1.pop().unwrap());
        } else if set2.is_empty() && !set3.is_empty() {
            set2.push(set3.pop().unwrap());
        }
        if set3.is_empty() && !set1.is_empty() {
            set3.push(set1.pop().unwrap());
        } else if set3.is_empty() && !set2.is_empty() {
            set3.push(set2.pop().unwrap());
        }
    }

    fn adjust_set_size(
        &self,
        set_to_adjust: &mut Vec<String>,
        target_size: usize,
        other_set1: &mut Vec<String>,
        other_set2: &mut Vec<String>,
        rng: &mut Option<&mut StdRng>,
    ) {
        while set_to_adjust.len() > target_size {
            let item = set_to_adjust.pop().unwrap();
            if other_set1.len() <= other_set2.len() {
                other_set1.push(item);
            } else {
                other_set2.push(item);
            }
        }
        while set_to_adjust.len() < target_size {
            if other_set1.len() > other_set2.len() {
                let index = rng.as_mut().map_or(other_set1.len() - 1, |rng| {
                    rng.gen_range(0..other_set1.len())
                });
                let item = other_set1.remove(index);
                set_to_adjust.push(item);
            } else if !other_set2.is_empty() {
                let index = rng.as_mut().map_or(other_set2.len() - 1, |rng| {
                    rng.gen_range(0..other_set2.len())
                });
                let item = other_set2.remove(index);
                set_to_adjust.push(item);
            } else {
                break; // Can't adjust further
            }
        }
    }
}
