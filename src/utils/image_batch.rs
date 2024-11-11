use std::{path::PathBuf, pin::Pin};

use rayon::{iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator}, slice::ParallelSliceMut};

use super::dataloader::DataLoaderForImages;

pub struct ImageBatch {
    pub images: Pin<Box<[u8]>>,
    pub images_per_batch: usize,
    pub bytes_per_image: usize,
}

impl ImageBatch {
    pub fn new(dl: &DataLoaderForImages) -> ImageBatch {
        let bytes_per_image = dl.image_width * dl.image_height * dl.image_bytes_per_pixel;
        let total_bytes = bytes_per_image as usize * dl.config.batch_size;

        let images = vec![0u8; total_bytes].into_boxed_slice();

        ImageBatch {
            images: Pin::new(images),
            images_per_batch: dl.config.batch_size,
            bytes_per_image: bytes_per_image as usize,
        }
    }

    pub fn load_raw_image_data(&mut self, paths: Vec<PathBuf>) {
        self.images
            .par_chunks_exact_mut(self.bytes_per_image)
            .zip(paths.par_iter())
            .for_each(|(chunk, path)| Self::path_to_buffer_copy(path, chunk));
    }

    fn path_to_buffer_copy(path: &PathBuf, slice: &mut [u8]) {
        let img = image::open(path).unwrap();
        slice.copy_from_slice(img.as_bytes());
    }
}