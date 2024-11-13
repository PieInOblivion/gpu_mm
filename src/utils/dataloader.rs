use image;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use super::dataloader_config::DataLoaderConfig;
use super::dataloader_error::DataLoaderError;

// TODO: ImageBatches needs to load both buffers if they are both empty
// TODO: Simplify the external and internal usage of these functions
// TODO: Condvar solution for prefetching?
// TODO: Combine ImageBatches and DataLoaderForImages

// TODO: Image label support:
//       - Built in csv support
//       - One hot encoding
//       - BYO array support

// NOTE: Always drop_last for now, as the end of the pixel buffer will include
//       last batches image data

// NOTE: Data represented as enum?
//  Use associated functions to move data around.
// General functions to manipulate data can then be used

#[derive(Copy, Clone)]
pub enum DatasetSplit {
    Train,
    Test,
    Validation,
}

pub struct DataLoaderForImages {
    pub dir: PathBuf,
    pub dataset: Vec<Box<str>>,
    pub dataset_indices: Vec<usize>,
    pub valid_extensions: HashSet<String>,
    pub image_width: u32,
    pub image_height: u32,
    pub image_channels: u32,
    pub image_bytes_per_pixel: u32,
    pub image_bytes_per_image: usize,
    pub image_total_bytes_per_batch: usize,
    pub config: DataLoaderConfig,
}

impl DataLoaderForImages {
    pub fn new(dir: &str, config: Option<DataLoaderConfig>) -> Result<Self, DataLoaderError> {
        let path = Path::new(dir);
        if !path.exists() {
            return Err(DataLoaderError::DirectoryNotFound(dir.to_string()));
        }

        let valid_extensions = image::ImageFormat::all()
            .flat_map(|format| format.extensions_str())
            .map(|ext| ext.to_string())
            .collect();

        let mut loader = DataLoaderForImages {
            dir: path.to_owned(),
            dataset: Vec::new(),
            dataset_indices: Vec::new(),
            valid_extensions,
            image_width: 0,
            image_height: 0,
            image_channels: 0,
            image_bytes_per_pixel: 0,
            image_bytes_per_image: 0,
            image_total_bytes_per_batch: 0,
            config: config.unwrap_or_default(),
        };

        loader.load_dataset()?;
        loader.scan_first_image()?;

        Ok(loader)
    }

    fn load_dataset(&mut self) -> Result<(), DataLoaderError> {
        // TODO: benchmark .par_bridge() from rayon. Also does not guarantee order of original iterator
        self.dataset = std::fs::read_dir(&self.dir)?
            .filter_map(Result::ok)
            .filter(|entry| self.is_valid_extension(&entry.path()))
            .filter_map(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|s| s.to_owned().into_boxed_str())
            })
            .collect();

        if self.dataset.is_empty() {
            return Err(DataLoaderError::EmptyDataset);
        }

        // read_dir does not guarantee consistancy or sorting of any kind since filesystems don't either
        // Useful if trying the same dataset on different systems or filesystems
        if self.config.sort_dataset {
            self.dataset.sort_unstable();
        }

        // .collect() can use size_hint from std::ops::Range
        //self.dataset_indices = Vec::with_capacity(self.dataset.len());
        //self.dataset_indices.extend(0..self.dataset.len());
        self.dataset_indices = (0..self.dataset.len()).collect();

        self.config.rng = if self.config.shuffle {
            if self.config.shuffle_seed.is_none() {
                self.config.shuffle_seed = Some(rand::thread_rng().gen());
            }
            Some(Arc::new(Mutex::new(StdRng::seed_from_u64(self.config.shuffle_seed.unwrap()))))
        } else {
            None
        };

        self.shuffle_whole_dataset()?;

        Ok(())
    }

    pub fn shuffle_whole_dataset(&mut self) -> Result<(), DataLoaderError> {
        let mut rng = self.config.rng.as_ref()
            .ok_or(DataLoaderError::RngNotSet)?
            .lock()
            .map_err(|_| DataLoaderError::RngLockError)?;
        self.dataset_indices.shuffle(&mut *rng);
        Ok(())
    }

    pub fn shuffle_individual_datasets(&mut self) -> Result<(), DataLoaderError> {
        let (train_size, test_size, _) = self.get_split_sizes();

        let mut rng = self.config.rng.as_ref()
            .ok_or(DataLoaderError::RngNotSet)?
            .lock()
            .map_err(|_| DataLoaderError::RngLockError)?;

        self.dataset_indices[0..train_size].shuffle(&mut *rng);
        self.dataset_indices[train_size..train_size + test_size].shuffle(&mut *rng);
        self.dataset_indices[train_size + test_size..].shuffle(&mut *rng);
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
        let train_size = (total_size as f32 * self.config.train_ratio) as usize;
        let test_size = (total_size as f32 * self.config.test_ratio) as usize;
        let val_size = total_size - train_size - test_size;
        (train_size, test_size, val_size)
    }

    // When finished a loop, allow for reshuffling to be an option
    pub fn next_batch_of_paths(
        &self,
        split: DatasetSplit,
        batch_number: usize,
    ) -> Option<Vec<PathBuf>> {
        let (train_size, test_size, _) = self.get_split_sizes();
        let (start_index, end_index) = match split {
            DatasetSplit::Train => (0, train_size),
            DatasetSplit::Test => (train_size, train_size + test_size),
            DatasetSplit::Validation => (train_size + test_size, self.dataset.len()),
        };

        let split_size = end_index - start_index;
        let batch_start = batch_number * self.config.batch_size;

        if batch_start >= split_size {
            return None;
        }

        let batch_end = (batch_start + self.config.batch_size).min(split_size);
        let is_last_batch = batch_end == split_size;

        if self.config.drop_last
            && is_last_batch
            && (batch_end - batch_start) < self.config.batch_size
        {
            return None;
        }

        let batch = self.dataset_indices[start_index + batch_start..start_index + batch_end]
            .par_iter()
            .map(|&idx| PathBuf::from(&self.dir).join(&*self.dataset[idx]))
            .collect();

        Some(batch)
    }

    fn scan_first_image(&mut self) -> Result<(), DataLoaderError> {
        let first_image_path = PathBuf::from(&self.dir).join(&*self.dataset[0]);
        let img = image::open(first_image_path)?;
        self.image_width = img.width();
        self.image_height = img.height();
        self.image_channels = img.color().channel_count() as u32;
        self.image_bytes_per_pixel = img.color().bytes_per_pixel() as u32;

        self.image_bytes_per_image = self.image_width as usize * self.image_height as usize * self.image_bytes_per_pixel as usize;
        self.image_total_bytes_per_batch = self.image_bytes_per_image * self.config.batch_size;
        Ok(())
    }
}
