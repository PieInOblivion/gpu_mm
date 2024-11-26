use std::{collections::VecDeque, pin::Pin, sync::Arc};

use crate::thread_pool::worker::{WorkType, WorkFuture, WorkResult};

use super::{
    dataloader_for_images::{DataLoaderForImages, DatasetSplit},
    image_batch::IteratorImageBatch,
};

struct PendingBatch {
    future: WorkFuture,
    batch_number: usize,
}

pub struct MultithreadedImageBatchIterator {
    dataloader: Arc<DataLoaderForImages>,
    split: DatasetSplit,
    pinned_buffer: Pin<Box<[u8]>>,
    next_batch: usize,
    pending_futures: VecDeque<PendingBatch>,
    max_pending: usize,
}

impl MultithreadedImageBatchIterator {
    fn new(dl: Arc<DataLoaderForImages>, split: DatasetSplit) -> Self {
        let max_pending = dl.config.prefetch_count;
        let mut iterator = MultithreadedImageBatchIterator { 
            dataloader: Arc::clone(&dl),
            split,
            pinned_buffer: Pin::new(vec![0u8; dl.image_total_bytes_per_batch].into_boxed_slice()),
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
                self.pending_futures.push_back(PendingBatch {
                    future,
                    batch_number,
                });
            } else {
                break
            }
        }
    }

    fn wait_for_next_batch(&mut self) -> Option<(usize, IteratorImageBatch)> {
        let position = self.pending_futures.iter()
            .position(|pending| pending.batch_number == self.next_batch)?;

        let PendingBatch { future, batch_number } = self.pending_futures.remove(position)?;

        match future.wait() {
            WorkResult::LoadImageBatch { batch_number: received_batch, batch } => {
                debug_assert_eq!(batch_number, received_batch, 
                    "Batch number mismatch: expected {}, got {}", 
                    batch_number, received_batch);

                self.pinned_buffer.copy_from_slice(&batch.image_data);
                
                Some((batch_number, IteratorImageBatch {
                    image_data: unsafe {
                        std::slice::from_raw_parts(
                            self.pinned_buffer.as_ref().as_ptr(),
                            self.pinned_buffer.len()
                        )
                    },
                    images_this_batch: batch.images_this_batch,
                    bytes_per_image: batch.bytes_per_image,
                    batch_number,
                }))
            },
            _ => unreachable!("Unexpected work result type"),
        }
    }
}

impl Iterator for MultithreadedImageBatchIterator {
    type Item = IteratorImageBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((batch_number, batch)) = self.wait_for_next_batch() {
            debug_assert_eq!(batch_number, self.next_batch,
                "Iterator sequence error: expected batch {}, got {}", 
                self.next_batch, batch_number);
            
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