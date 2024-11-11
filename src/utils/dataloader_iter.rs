use std::{pin::Pin, sync::Arc};

use super::dataloader::{DataLoaderForImages, DatasetSplit, ImageBatch};

pub struct ImageBatchIterator<'a> {
    data_loader: &'a DataLoaderForImages,
    split: DatasetSplit,
    current_batch_index: usize,
    batch: ImageBatch,
}

impl<'a> ImageBatchIterator<'a> {
    pub fn new(data_loader: &'a DataLoaderForImages, split: DatasetSplit) -> Self {
        ImageBatchIterator {
            data_loader,
            split,
            current_batch_index: 0,
            batch: ImageBatch::new(data_loader),
        }
    }
}

impl<'a> Iterator for ImageBatchIterator<'a> {
    type Item = Pin<Box<[u8]>>;

    // TODO: Option to reshuffle after the last batch is served
    fn next(&mut self) -> Option<Self::Item> {
        let paths = self
            .data_loader
            .next_batch_of_paths(self.split, self.current_batch_index);
        self.current_batch_index += 1;

        if let Some(next_paths) = paths {
            self.batch.load_raw_image_data(next_paths);
            // Return reference to the pinned memory instead
            Some(self.batch.images.clone())
        } else {
            None
        }
    }
}

// -------------------------------------------------------------

use crossbeam_channel::{bounded, Receiver};
use std::thread;

pub struct ParallelBufferIterator<T: Send + 'static> {
    receiver: Receiver<Option<Vec<T>>>,
}

impl<T: Send + 'static> ParallelBufferIterator<T> {
    pub fn new<F>(mut loader: F) -> Self
    where
        F: FnMut() -> Option<Vec<T>> + Send + 'static,
    {
        let (sender, receiver) = bounded(1);

        thread::spawn(move || {
            while let Some(batch) = loader() {
                if sender.send(Some(batch)).is_err() {
                    break;
                }
            }
            let _ = sender.send(None);
        });

        ParallelBufferIterator { receiver }
    }
}

impl<T: Send + 'static> Iterator for ParallelBufferIterator<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(Some(batch)) => Some(batch),
            _ => None,
        }
    }
}

// -------------------------------------------------------------

pub struct ParallelImageBatchIterator {
    receiver: Receiver<Option<Pin<Box<[u8]>>>>,
}

impl ParallelImageBatchIterator {
    pub fn new(data_loader: Arc<DataLoaderForImages>, split: DatasetSplit) -> Self {
        let (sender, receiver) = bounded(1);
        let mut current_batch_index = 0;

        thread::spawn(move || {
            let mut batch = ImageBatch::new(&data_loader);

            loop {
                let paths = data_loader.next_batch_of_paths(split, current_batch_index);
                current_batch_index += 1;

                match paths {
                    Some(next_paths) => {
                        batch.load_raw_image_data(next_paths);
                        if sender.send(Some(batch.images.clone())).is_err() {
                            break;
                        }
                    }
                    None => {
                        let _ = sender.send(None);
                        break;
                    }
                }
            }
        });

        ParallelImageBatchIterator { receiver }
    }
}

impl Iterator for ParallelImageBatchIterator {
    type Item = Pin<Box<[u8]>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(Some(batch)) => Some(batch),
            _ => None,
        }
    }
}

pub fn create_parallel_image_batch_iterator(
    data_loader: DataLoaderForImages,
    split: DatasetSplit,
) -> ParallelImageBatchIterator {
    let data_loader = Arc::new(data_loader);
    ParallelImageBatchIterator::new(data_loader, split)
}
