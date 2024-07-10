use crossbeam::thread;
use image::{self, ColorType, GenericImageView};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fmt::Debug;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Mutex, RwLock};
use thiserror::Error;

// TODO:
// Simplify and fix the associated prefetch loading functions

#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid dataset split ratios. Train: {train}, Test: {test}")]
    InvalidSplitRatios { train: f32, test: f32 },
    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),
    #[error("Unsupported image format")]
    UnsupportedImageFormat,
    #[error("No images found in the dataset")]
    EmptyDataset,
    #[error("Thread communication error: {0}")]
    ThreadError(String),
    #[error("Failed to send task: {0}")]
    TaskSendError(String),
}

pub enum DatasetSplit {
    Train,
    Test,
    Validation,
}

pub struct ImageBatches {
    images: Pin<Box<[u8]>>,
    images_mutex: Mutex<()>,
    prefetched_images: Pin<Box<[u8]>>,
    prefetch_mutex: Mutex<()>,
    use_prefetch_next: bool,
    images_per_batch: usize,
    bytes_per_image: usize,
    threads: usize,
}

impl ImageBatches {
    pub fn new(dl: &DataLoaderForImages) -> ImageBatches {
        let bytes_per_image = dl.image_width * dl.image_height * dl.image_bytes_per_pixel;
        let total_bytes = bytes_per_image as usize * dl.opt_batch_size;

        let images = vec![0u8; total_bytes].into_boxed_slice();
        let prefetched_images = vec![0u8; total_bytes].into_boxed_slice();

        ImageBatches {
            images: Pin::new(images),
            images_mutex: Mutex::new(()),
            prefetched_images: Pin::new(prefetched_images),
            prefetch_mutex: Mutex::new(()),
            use_prefetch_next: false,
            images_per_batch: dl.opt_batch_size,
            bytes_per_image: bytes_per_image as usize,
            threads: dl.opt_threads,
        }
    }

    pub fn load_raw_image_data(&mut self, paths: Vec<PathBuf>) {
        // TODO: Compare Rayon, Tokio, Crossbeam and std
        // Tokio has async io advantage
        // Crossbeam > std
        // Attempt thread pool style solution

        // This function should lock and unlock correct buffer mutex
    }

    fn image_pathbuf_to_batch_buffer_raw(&mut self, path: &PathBuf, buffer_offset: usize) {
        let img = image::open(path).unwrap();
        let img_rgb = img.to_rgb8();
        let img_bytes = img_rgb.as_raw();

        // Determine which buffer to use based on use_prefetch_next
        let (buffer, mutex) = if self.use_prefetch_next {
            (&mut self.prefetched_images, &self.prefetch_mutex)
        } else {
            (&mut self.images, &self.images_mutex)
        };

        let slice_destination = buffer_offset..(buffer_offset + self.bytes_per_image);
        dbg!(&slice_destination);

        // Use unsafe to get a mutable reference to the pinned data
        unsafe {
            let buffer_slice = buffer.get_unchecked_mut(slice_destination);
            buffer_slice.copy_from_slice(&img_bytes[..self.bytes_per_image]);
        }
    }
}

pub struct DataLoaderForImages {
    pub dir: PathBuf,
    pub dataset: Vec<Box<str>>,
    pub dataset_indices: Vec<usize>,
    pub opt_threads: usize,
    pub opt_batch_size: usize,
    pub opt_batch_prefetch: bool,
    pub opt_train_ratio: f32,
    pub opt_test_ratio: f32,
    pub opt_sort_dataset: bool,
    pub opt_shuffle: bool,
    pub opt_shuffle_seed: Option<u64>,
    pub opt_drop_last: bool,
    pub valid_extensions: Vec<String>,
    pub current_train_batch: usize,
    pub current_test_batch: usize,
    pub current_val_batch: usize,
    pub image_width: u32,
    pub image_height: u32,
    pub image_channels: u32,
    pub image_bytes_per_pixel: u32,
}

impl DataLoaderForImages {
    pub fn new(dir: &str) -> Result<Self, DataLoaderError> {
        let path = Path::new(dir);
        if !path.exists() {
            return Err(DataLoaderError::DirectoryNotFound(dir.to_string()));
        }

        Ok(DataLoaderForImages {
            dir: path.to_owned(),
            dataset: Vec::new(),
            dataset_indices: Vec::new(),
            opt_threads: num_cpus::get(),
            opt_batch_size: 32,
            opt_batch_prefetch: true,
            opt_train_ratio: 0.8,
            opt_test_ratio: 0.1,
            opt_sort_dataset: false,
            opt_shuffle: true,
            opt_shuffle_seed: None,
            opt_drop_last: true,
            valid_extensions: vec!["png".to_string(), "jpg".to_string(), "jpeg".to_string()],
            current_train_batch: 0,
            current_test_batch: 0,
            current_val_batch: 0,
            image_width: 0,
            image_height: 0,
            image_channels: 0,
            image_bytes_per_pixel: 0,
        })
    }

    pub fn load_dataset(&mut self) -> Result<(), DataLoaderError> {
        self.dataset = std::fs::read_dir(&self.dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| self.is_valid_extension(path))
            .filter_map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str().map(|s| s.into()))
            })
            .collect();

        if self.dataset.is_empty() {
            return Err(DataLoaderError::EmptyDataset);
        }

        // read_dir does not guarantee consistancy or sorting of any kind
        if self.opt_sort_dataset {
            self.dataset.sort_unstable();
        }

        self.dataset_indices = Vec::with_capacity(self.dataset.len());
        self.dataset_indices.extend(0..self.dataset.len());

        let mut rng = if self.opt_shuffle {
            if self.opt_shuffle_seed.is_none() {
                self.opt_shuffle_seed = Some(rand::thread_rng().gen());
            }
            Some(StdRng::seed_from_u64(self.opt_shuffle_seed.unwrap()))
        } else {
            None
        };

        if let Some(mut rng) = rng.as_mut() {
            self.dataset_indices.shuffle(&mut rng);
        }

        self.scan_first_image()?;

        Ok(())
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

    fn is_valid_extension(&self, path: &PathBuf) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| self.valid_extensions.contains(&ext.to_lowercase()))
            .unwrap_or(false)
    }

    fn get_split_sizes(&self) -> (usize, usize, usize) {
        let total_size = self.dataset.len();
        let train_size = (total_size as f32 * self.opt_train_ratio) as usize;
        let test_size = (total_size as f32 * self.opt_test_ratio) as usize;
        let val_size = total_size - train_size - test_size;
        (train_size, test_size, val_size)
    }

    fn next_batch_of_paths(&mut self, split: DatasetSplit) -> (Vec<PathBuf>, bool) {
        let (train_size, test_size, _) = self.get_split_sizes();
        let (start_index, end_index, current_batch) = match split {
            DatasetSplit::Train => (0, train_size, &mut self.current_train_batch),
            DatasetSplit::Test => (
                train_size,
                train_size + test_size,
                &mut self.current_test_batch,
            ),
            DatasetSplit::Validation => (
                train_size + test_size,
                self.dataset.len(),
                &mut self.current_val_batch,
            ),
        };

        let split_size = end_index - start_index;
        let total_batches = (split_size + self.opt_batch_size - 1) / self.opt_batch_size;

        let batch_start = *current_batch * self.opt_batch_size;
        let batch = self.dataset_indices[start_index..end_index]
            .iter()
            .cycle()
            .skip(batch_start)
            .take(self.opt_batch_size)
            .map(|&idx| PathBuf::from(&self.dir).join(&*self.dataset[idx]))
            .collect();

        let is_last = *current_batch == total_batches - 1;
        *current_batch = (*current_batch + 1) % total_batches;

        (batch, is_last)
    }

    pub fn scan_first_image(&mut self) -> Result<(), DataLoaderError> {
        let first_image_path = PathBuf::from(&self.dir).join(&*self.dataset[0]);
        let img = image::open(first_image_path)?;
        self.image_width = img.width();
        self.image_height = img.height();
        self.image_channels = img.color().channel_count() as u32;
        self.image_bytes_per_pixel = img.color().bytes_per_pixel() as u32;
        Ok(())
    }

    pub fn print_dataset_info(&self) {
        let total_size = self.dataset.len();
        let train_size = (total_size as f32 * self.opt_train_ratio) as usize;
        let test_size = (total_size as f32 * self.opt_test_ratio) as usize;
        let val_size = total_size - train_size - test_size;

        let train_batches = (train_size + self.opt_batch_size - 1) / self.opt_batch_size;
        let test_batches = (test_size + self.opt_batch_size - 1) / self.opt_batch_size;
        let val_batches = (val_size + self.opt_batch_size - 1) / self.opt_batch_size;

        let train_batch_remainder = train_size % self.opt_batch_size;
        let test_batch_remainder = test_size % self.opt_batch_size;
        let val_batch_remainder = val_size % self.opt_batch_size;

        println!("Dataset Information:");
        println!("-------------------");
        println!("Total size: {}", total_size);
        println!("Batch size: {}", self.opt_batch_size);
        println!();
        println!("Train split:");
        println!(
            "  Size: {} ({:.2}%)",
            train_size,
            self.opt_train_ratio * 100.0
        );
        println!("  Batches: {}", train_batches);
        println!(
            "  Last batch size: {}",
            if train_batch_remainder > 0 {
                train_batch_remainder
            } else {
                self.opt_batch_size
            }
        );
        println!();
        println!("Test split:");
        println!(
            "  Size: {} ({:.2}%)",
            test_size,
            self.opt_test_ratio * 100.0
        );
        println!("  Batches: {}", test_batches);
        println!(
            "  Last batch size: {}",
            if test_batch_remainder > 0 {
                test_batch_remainder
            } else {
                self.opt_batch_size
            }
        );
        println!();
        println!("Validation split:");
        println!(
            "  Size: {} ({:.2}%)",
            val_size,
            (1.0 - self.opt_train_ratio - self.opt_test_ratio) * 100.0
        );
        println!("  Batches: {}", val_batches);
        println!(
            "  Last batch size: {}",
            if val_batch_remainder > 0 {
                val_batch_remainder
            } else {
                self.opt_batch_size
            }
        );
        println!();
        println!("Shuffle: {}", self.opt_shuffle);
        println!("Seed: {:?}", self.opt_shuffle_seed);
    }
}
