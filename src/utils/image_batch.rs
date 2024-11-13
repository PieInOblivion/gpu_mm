use std::path::PathBuf;

use rayon::{iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator}, slice::ParallelSliceMut};

use super::dataloader::DataLoaderForImages;

#[derive(Clone)]
pub struct ImageBatch {
    pub image_data: Box<[u8]>,
    pub images_this_batch: usize,
    pub bytes_per_image: usize,
}

impl ImageBatch {
    pub fn new(dl: &DataLoaderForImages) -> ImageBatch {
        ImageBatch {
            image_data: vec![0u8; dl.image_total_bytes_per_batch].into_boxed_slice(),
            images_this_batch: dl.config.batch_size,
            bytes_per_image: dl.image_bytes_per_image
        }
    }

    pub fn load_raw_image_data(&mut self, paths: &Vec<PathBuf>) {
        self.images_this_batch = paths.len();

        self.image_data
            .par_chunks_exact_mut(self.bytes_per_image)
            .zip(paths.par_iter())
            .for_each(|(chunk, path)| Self::path_to_buffer_copy(path, chunk));
    }

    fn path_to_buffer_copy(path: &PathBuf, slice: &mut [u8]) {
        let img = image::open(path).unwrap();
        slice.copy_from_slice(img.as_bytes());
    }
}


pub struct IteratorImageBatch {
    pub image_data: &'static [u8],
    pub images_this_batch: usize,
    pub bytes_per_image: usize,
}