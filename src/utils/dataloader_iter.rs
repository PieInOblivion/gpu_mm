use std::{pin::Pin, sync::Arc, thread};

use crossbeam_channel::{Receiver, bounded};

use super::{dataloader::{DataLoaderForImages, DatasetSplit}, image_batch::ImageBatch};

pub struct ParallelImageBatchIterator {
    receiver: Receiver<Option<Pin<Box<[u8]>>>>,
}

impl ParallelImageBatchIterator {
    fn new(data_loader: Arc<DataLoaderForImages>, split: DatasetSplit) -> Self {
        let (sender, receiver) = bounded(1);
        let mut batch = ImageBatch::new(&data_loader);
        let mut current_batch_index = 0;

        thread::spawn(move || {
            while let Some(paths) = data_loader.next_batch_of_paths(split, current_batch_index) {
                batch.load_raw_image_data(paths);
                if sender.send(Some(batch.images.clone())).is_err() {
                    break;
                }
                current_batch_index += 1;
            }
            let _ = sender.send(None);
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

pub trait ParallelDataLoaderIterator {
    fn par_iter(self: Arc<Self>, split: DatasetSplit) -> ParallelImageBatchIterator;
}

impl ParallelDataLoaderIterator for DataLoaderForImages {
    fn par_iter(self: Arc<Self>, split: DatasetSplit) -> ParallelImageBatchIterator {
        ParallelImageBatchIterator::new(self, split)
    }
}