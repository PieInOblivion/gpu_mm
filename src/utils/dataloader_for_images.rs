use image;
use image::ColorType;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::dataloader_config::DataLoaderConfig;
use super::dataloader_error::DataLoaderError;
use crate::thread_pool::worker::{WorkType, WorkResult};

// TODO: Image label support:
//       - Built in csv support
//       - One hot encoding
//       - BYO array support

// NOTE: The end of the last batch, if partial, pixel buffer will include last batches image data

#[derive(Clone, Copy, Debug)]
pub enum ComputeFormat {
    U8,
    U16,
    F32,
}

impl From<ColorType> for ComputeFormat {
    fn from(color_type: ColorType) -> Self {
        match color_type {
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => ComputeFormat::U8,
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => ComputeFormat::U16,
            ColorType::Rgb32F | ColorType::Rgba32F => ComputeFormat::F32,
            _ => panic!("Unsupported color type"),
        }
    }
}

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
    pub image_color_type: ColorType,
    pub image_compute_format: ComputeFormat,
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
            image_color_type: ColorType::Rgb32F,
            image_compute_format: ComputeFormat::F32,
            config: config.unwrap_or_default(),
        };

        loader.load_dataset()?;
        loader.scan_first_image()?;

        Ok(loader)
    }

    pub fn new_arc(dir: &str, config: Option<DataLoaderConfig>) -> Result<Arc<Self>, DataLoaderError> {
        Ok(Arc::new(Self::new(dir, config)?))
    }

    fn load_dataset(&mut self) -> Result<(), DataLoaderError> {
        // TODO: Use thread pool and batch work
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

    // TODO: When finished a loop, allow for reshuffling to be an option
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

        let indices = &self.dataset_indices[start_index + batch_start..start_index + batch_end];

        let work_items = indices.iter().map(|&idx| WorkType::BuildPath {
            base_dir: self.dir.clone(),
            filename: self.dataset[idx].clone(),
        }).collect();

        let batch = self.config.thread_pool.submit_batch(work_items);
        let paths = batch.wait().into_iter()
            .map(|r| match r {
                WorkResult::BuildPath{path} => path,
                _ => unreachable!(),
            })
            .collect();

        Some(paths)
    }

    fn scan_first_image(&mut self) -> Result<(), DataLoaderError> {
        let first_image_path = PathBuf::from(&self.dir).join(&*self.dataset[0]);
        let img = image::open(first_image_path)?;
        self.image_width = img.width();
        self.image_height = img.height();
        self.image_channels = img.color().channel_count() as u32;
        self.image_bytes_per_pixel = img.color().bytes_per_pixel() as u32;

        self.image_color_type = img.color();
        self.image_compute_format = ComputeFormat::from(img.color());

        self.image_bytes_per_image = self.image_width as usize * self.image_height as usize * self.image_bytes_per_pixel as usize;
        self.image_total_bytes_per_batch = self.image_bytes_per_image * self.config.batch_size;
        Ok(())
    }
}
