use std::{pin::Pin, sync::Arc, thread};

use crossbeam_channel::{Receiver, bounded};

use super::{
    dataloader::{DataLoaderForImages, DatasetSplit},
    image_batch::{ImageBatch, IteratorImageBatch}
};


pub struct ParallelImageBatchIterator {
    receiver: Receiver<Option<ImageBatch>>,
    pinned_buffer: Pin<Box<[u8]>>,
}

impl ParallelImageBatchIterator {
    fn new(dl: Arc<DataLoaderForImages>, split: DatasetSplit) -> Self {
        let (sender, receiver) = bounded(1);

        let pinned_buffer = Pin::new(vec![0u8; dl.image_total_bytes_per_batch].into_boxed_slice());

        let mut current_batch_index = 0;

        thread::spawn(move || {
            let mut batch = ImageBatch::new(&dl);
            
            while let Some(paths) = dl.next_batch_of_paths(split, current_batch_index) {
                batch.load_raw_image_data(&paths);
                
                if sender.send(Some(batch.clone())).is_err() {
                    break;
                }

                current_batch_index += 1;
            }

            let _ = sender.send(None);
        });

        ParallelImageBatchIterator { 
            receiver,
            pinned_buffer,
        }
    }
}

impl Iterator for ParallelImageBatchIterator {
    type Item = IteratorImageBatch;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv().ok()? {
            Some(batch) => {
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