use std::{collections::VecDeque, sync::{atomic::AtomicUsize, Arc, Condvar, Mutex}};

use super::worker::{WorkType, WorkFutureBatch, WorkFuture, WorkItem, WorkQueue, Worker};

pub struct ThreadPool {
    pub work_queue: Arc<WorkQueue>,
    workers: Vec<Worker>,
}

impl ThreadPool {
    pub fn new(size: usize) -> ThreadPool {

        let work_queue = Arc::new(WorkQueue {
            queue: Mutex::new(VecDeque::new()),
            items_count: AtomicUsize::new(0),
            condvar: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            workers.push(Worker::new(
                id,
                Arc::clone(&work_queue),
            ));
        }

        ThreadPool {
            work_queue,
            workers,
        }
    }

    pub fn submit_work(&self, work: WorkType) -> WorkFuture {
        let future = WorkFuture::new();
        
        let work_item = WorkItem {
            work,
            future: future.clone(),
        };
        
        self.work_queue.submit_work_item(work_item);

        self.work_queue.condvar.notify_one();
        
        future
    }

    pub fn submit_batch(&self, work_items: Vec<WorkType>) -> WorkFutureBatch {
        let mut futures = Vec::with_capacity(work_items.len());
        let mut batch_items = Vec::with_capacity(work_items.len());
        
        for work in work_items {
            let future = WorkFuture::new();
            futures.push(future.clone());
            
            batch_items.push(WorkItem {
                work,
                future,
            });
        }
        
        self.work_queue.submit_work_batch(batch_items);
        
        for _ in 0..futures.len().min(self.workers.len()) {
            self.work_queue.condvar.notify_one();
        }

        WorkFutureBatch { futures }
    }

}