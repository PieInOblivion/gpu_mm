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

// TODO: ImageBatches needs to load both buffers if they are both empty
// TODO: Simplify the external and internal usage of these functions
// TODO: Condvar solution for prefetching?

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
    pub current_train_batch: usize,
    pub current_test_batch: usize,
    pub current_val_batch: usize,
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
            current_train_batch: 0,
            current_test_batch: 0,
            current_val_batch: 0,
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

    // TODO: Make it an iterator, or an iterator version of it
    // When finished a loop, allow for reshuffling to be an option
    pub fn next_batch_of_paths(&mut self, split: DatasetSplit) -> (Vec<PathBuf>, bool) {
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
        let total_batches = (split_size + self.config.batch_size - 1) / self.config.batch_size;

        let batch_start = *current_batch * self.config.batch_size;
        let batch = self.dataset_indices[start_index..end_index]
            .iter()
            .cycle()
            .skip(batch_start)
            .take(self.config.batch_size)
            .map(|&idx| PathBuf::from(&self.dir).join(&*self.dataset[idx]))
            .collect();

        let is_last = *current_batch == total_batches - 1;
        *current_batch = (*current_batch + 1) % total_batches;

        (batch, is_last)
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
}
