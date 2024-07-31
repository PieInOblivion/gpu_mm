use std::path::PathBuf;

use super::dataloader::{DataLoaderForImages, DatasetSplit};

pub struct ImageBatchIterator<'a> {
    data_loader: &'a DataLoaderForImages,
    split: DatasetSplit,
    current_batch: usize,
}

impl<'a> ImageBatchIterator<'a> {
    pub fn new(data_loader: &'a DataLoaderForImages, split: DatasetSplit) -> Self {
        ImageBatchIterator {
            data_loader,
            split,
            current_batch: 0,
        }
    }
}

impl<'a> Iterator for ImageBatchIterator<'a> {
    type Item = Vec<PathBuf>;

    // TODO: Option to reshuffle after the last batch is served
    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.data_loader.next_batch_of_paths(self.split, self.current_batch);
        self.current_batch += 1;
        batch
    }
}