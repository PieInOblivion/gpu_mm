use std::{collections::VecDeque, pin::Pin, sync::{atomic::{AtomicUsize, Ordering}, Arc}, thread};

use crossbeam_channel::{bounded, Receiver, Sender};

use super::{
    dataloader::{DataLoaderForImages, DatasetSplit},
    image_batch::{ImageBatch, IteratorImageBatch}
};


pub struct ParallelImageBatchIterator {
    receiver: Receiver<Option<(usize, ImageBatch)>>,
    pinned_buffer: Pin<Box<[u8]>>,
    next_batch: usize,
    pending_batches: VecDeque<(usize, ImageBatch)>,
}

impl ParallelImageBatchIterator {
    fn new(dl: Arc<DataLoaderForImages>, split: DatasetSplit) -> Self {
        let (sender, receiver) = bounded(1);
        let pinned_buffer = Pin::new(vec![0u8; dl.image_total_bytes_per_batch].into_boxed_slice());
        
        let batch_counter = Arc::new(AtomicUsize::new(0));
        let active_workers = Arc::new(AtomicUsize::new(dl.config.num_of_batch_prefetches));
        
        for _ in 0..dl.config.num_of_batch_prefetches {
            let dl_clone = Arc::clone(&dl);
            let sender_clone = sender.clone();
            let counter_clone = Arc::clone(&batch_counter);
            let workers_clone = Arc::clone(&active_workers);
            
            thread::spawn(move || {
                Self::prefetch_worker(dl_clone, split, sender_clone, counter_clone, workers_clone);
            });
        }

        ParallelImageBatchIterator { 
            receiver,
            pinned_buffer,
            next_batch: 0,
            pending_batches: VecDeque::new(),
        }
    }

    fn prefetch_worker(
        dl: Arc<DataLoaderForImages>,
        split: DatasetSplit,
        sender: Sender<Option<(usize, ImageBatch)>>,
        batch_counter: Arc<AtomicUsize>,
        active_workers: Arc<AtomicUsize>,
    ) {
        let mut batch = ImageBatch::new(&dl);

        loop {
            let current_batch = batch_counter.fetch_add(1, Ordering::SeqCst);
            
            match dl.next_batch_of_paths(split, current_batch) {
                Some(paths) => {
                    batch.load_raw_image_data(&paths);
                    println!("BATCH LOADED AND SENT: {}", current_batch);
                    if sender.send(Some((current_batch, batch.clone()))).is_err() {
                        break;
                    }
                }
                None => {
                    if active_workers.fetch_sub(1, Ordering::SeqCst) == 1 {
                        let _ = sender.send(None);
                    }
                    break;
                }
            }
        }
    }
}

impl Iterator for ParallelImageBatchIterator {
    type Item = IteratorImageBatch;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv().ok()? {
            Some((batch_num, batch)) => {
                //print!("{:?}", batch.image_data);
                self.pinned_buffer.copy_from_slice(&batch.image_data);
                Some(IteratorImageBatch {
                    image_data: unsafe {
                        std::slice::from_raw_parts(
                            self.pinned_buffer.as_ref().as_ptr(),
                            self.pinned_buffer.len())
                        },
                    images_this_batch: batch.images_this_batch,
                    bytes_per_image: batch.bytes_per_image,
                    batch_number: batch_num,
                })
            }
            None => None,
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