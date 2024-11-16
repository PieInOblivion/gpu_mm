use std::{collections::VecDeque, sync::{atomic::{AtomicUsize, Ordering}, Arc, Condvar, Mutex}};

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

        {
            let mut queue = self.work_queue.queue.lock().unwrap();
            queue.push_back(work_item);
            self.work_queue.items_count.fetch_add(1, Ordering::SeqCst);
        }
        self.work_queue.condvar.notify_one();

        future
    }

    pub fn submit_batch(&self, work_items: Vec<WorkType>) -> WorkFutureBatch {
        let mut futures = Vec::with_capacity(work_items.len());
        
        {
            let mut queue = self.work_queue.queue.lock().unwrap();
            
            for work in work_items {
                let future = WorkFuture::new();
                futures.push(future.clone());
                
                queue.push_back(WorkItem {
                    work,
                    future,
                });
            }
            
            self.work_queue.items_count.fetch_add(futures.len(), Ordering::SeqCst);
        }
        
        for _ in 0..futures.len().min(self.workers.len()) {
            self.work_queue.condvar.notify_one();
        }

        WorkFutureBatch { futures }
    }

}