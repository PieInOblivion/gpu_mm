use std::collections::VecDeque;
use std::sync::Arc;

use crate::thread_pool::worker::{WorkType, WorkFuture, WorkResult};

use super::{
    dataloader_for_images::{DataLoaderForImages, DatasetSplit},
    image_batch::ImageBatch,
};

pub struct MultithreadedImageBatchIterator {
    dataloader: Arc<DataLoaderForImages>,
    split: DatasetSplit,
    next_batch: usize,
    pending_futures: VecDeque<WorkFuture>,
    max_pending: usize,
}

impl MultithreadedImageBatchIterator {
    fn new(dl: Arc<DataLoaderForImages>, split: DatasetSplit) -> Self {
        let max_pending = dl.config.prefetch_count;
        let mut iterator = MultithreadedImageBatchIterator { 
            dataloader: Arc::clone(&dl),
            split,
            next_batch: 0,
            pending_futures: VecDeque::with_capacity(max_pending),
            max_pending,
        };

        iterator.request_next_batches();

        iterator
    }

    fn request_next_batches(&mut self) {
        while self.pending_futures.len() < self.max_pending {
            let batch_number = self.next_batch + self.pending_futures.len();
            
            if let Some(paths) = self.dataloader.next_batch_of_paths(self.split, batch_number) {
                let work = WorkType::LoadImageBatch {
                    dataloader: Arc::clone(&self.dataloader),
                    batch_number,
                    paths,
                };
                
                let future = self.dataloader.config.thread_pool.submit_work(work);
                self.pending_futures.push_back(future);
            } else {
                break
            }
        }
    }

    fn wait_for_next_batch(&mut self) -> Option<ImageBatch> {
        let future = self.pending_futures.pop_front()?;

        match future.wait() {
            WorkResult::LoadImageBatch { batch, .. } => {
                debug_assert_eq!(batch.batch_number, self.next_batch,
                    "Iterator sequence error: expected batch {}, got {}", 
                    self.next_batch, batch.batch_number);
                Some(batch)
            },
            _ => unreachable!("Unexpected work result type"),
        }
    }
}

impl Iterator for MultithreadedImageBatchIterator {
    type Item = ImageBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(batch) = self.wait_for_next_batch() {
            self.next_batch += 1;
            self.request_next_batches();
            Some(batch)
        } else {
            None
        }
    }
}

pub trait MultithreadedDataLoaderIterator {
    fn par_iter(&self, split: DatasetSplit) -> MultithreadedImageBatchIterator;
}

impl MultithreadedDataLoaderIterator for Arc<DataLoaderForImages> {
    fn par_iter(&self, split: DatasetSplit) -> MultithreadedImageBatchIterator {
        MultithreadedImageBatchIterator::new(Arc::clone(self), split)
    }
}