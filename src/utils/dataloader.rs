use image;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Mutex;

use super::dataloader_config::DataLoaderConfig;
use super::dataloader_error::DataLoaderError;
use super::dataloader_iter::ImageBatchIterator;

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

#[derive(Copy, Clone)]
pub enum DatasetSplit {
    Train,
    Test,
    Validation,
}

pub struct ImageBatches {
    pub images: Pin<Box<[u8]>>,
    pub images_mutex: Mutex<()>,
    pub images_loaded: bool,
    pub prefetched_images: Pin<Box<[u8]>>,
    pub prefetch_mutex: Mutex<()>,
    pub prefetched_images_loaded: bool,
    pub out_use_prefetch_next: bool,
    pub images_per_batch: usize,
    pub bytes_per_image: usize,
}

impl ImageBatches {
    pub fn new(dl: &DataLoaderForImages) -> ImageBatches {
        let bytes_per_image = dl.image_width * dl.image_height * dl.image_bytes_per_pixel;
        let total_bytes = bytes_per_image as usize * dl.config.batch_size;

        let images = vec![0u8; total_bytes].into_boxed_slice();
        let prefetched_images = vec![0u8; total_bytes].into_boxed_slice();

        ImageBatches {
            images: Pin::new(images),
            images_mutex: Mutex::new(()),
            images_loaded: false,
            prefetched_images: Pin::new(prefetched_images),
            prefetch_mutex: Mutex::new(()),
            prefetched_images_loaded: false,
            out_use_prefetch_next: false,
            images_per_batch: dl.config.batch_size,
            bytes_per_image: bytes_per_image as usize,
        }
    }

    pub fn load_raw_image_data(&mut self, paths: Vec<PathBuf>) {
        let (buffer, mutex) = if self.out_use_prefetch_next {
            (&mut self.prefetched_images, &self.prefetch_mutex)
        } else {
            (&mut self.images, &self.images_mutex)
        };

        let _mutex_lock = mutex.lock().unwrap();

        buffer
            .par_chunks_exact_mut(self.bytes_per_image)
            .zip(paths.par_iter())
            .for_each(|(chunk, path)| Self::path_to_buffer_copy(path, chunk));
    }

    fn path_to_buffer_copy(path: &PathBuf, slice: &mut [u8]) {
        let img = image::open(path).unwrap();
        slice.copy_from_slice(img.as_bytes());
    }
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

        let mut rng = if self.config.shuffle {
            if self.config.shuffle_seed.is_none() {
                self.config.shuffle_seed = Some(rand::thread_rng().gen());
            }
            Some(StdRng::seed_from_u64(self.config.shuffle_seed.unwrap()))
        } else {
            None
        };

        // TODO: Can this be deterministically parallelised?
        if let Some(mut rng) = rng.as_mut() {
            self.dataset_indices.shuffle(&mut rng);
        }

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
            .iter()
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
        Ok(())
    }

    pub fn iter(&self, split: DatasetSplit) -> ImageBatchIterator {
        ImageBatchIterator::new(self, split)
    }
}

