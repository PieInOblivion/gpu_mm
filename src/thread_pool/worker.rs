use std::path::PathBuf;
use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::VecDeque;
use std::thread;

use crate::utils;
use utils::dataloader_for_images::DataLoaderForImages;
use utils::image_batch::ImageBatch;

#[derive(Copy, Clone)]
pub struct DataPtr(*mut u8);
unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

pub enum WorkType {
    LoadImageBatch {
        dataloader: Arc<DataLoaderForImages>,
        batch_number: usize,
        paths: Vec<PathBuf>,
    },
    LoadSingleImage {
        path: PathBuf,
        start_idx: usize,
        end_idx: usize,
        data_ptr: DataPtr,
    },
    BuildPath {
        base_dir: PathBuf,
        filename: Box<str>,
    }
}

pub enum WorkResult {
    LoadImageBatch {
        batch_number: usize,
        batch: ImageBatch,
    },
    LoadSingleImage,
    BuildPath {
        path: PathBuf
    },
}

#[derive(Clone)]
pub struct WorkFuture {
    pub state: Arc<(Mutex<Option<WorkResult>>, Condvar)>,
}

impl WorkFuture {
    pub fn new() -> Self {
        WorkFuture {
            state: Arc::new((Mutex::new(None), Condvar::new())),
        }
    }

    pub fn wait(self) -> WorkResult {
        let (lock, cvar) = &*self.state;
        let mut result = lock.lock().unwrap();
        while result.is_none() {
            result = cvar.wait(result).unwrap();
        }
        result.take().unwrap()
    }

    pub fn is_complete(&self) -> bool {
        self.state.0.lock().unwrap().is_some()
    }

    pub fn complete(&self, result: WorkResult) {
        let (lock, cvar) = &*self.state;
        *lock.lock().unwrap() = Some(result);
        cvar.notify_one();
    }
}

pub struct WorkQueue {
    pub queue: Mutex<VecDeque<WorkItem>>,
    pub items_count: AtomicUsize,
    pub condvar: Condvar,
}

pub struct WorkItem {
    pub work: WorkType,
    pub future: WorkFuture,
}

pub struct WorkFutureBatch {
    pub futures: Vec<WorkFuture>,
}

impl WorkFutureBatch {
    pub fn is_complete(&self) -> bool {
        self.futures.iter().all(|f| f.is_complete())
    }

    pub fn wait(self) -> Vec<WorkResult> {
        self.futures
            .into_iter()
            .map(|future| future.wait())
            .collect()
    }
}

pub struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    pub fn new(id: usize, work_queue: Arc<WorkQueue>) -> Worker {
        let thread = thread::spawn(move || {
            loop {
                if let Some(work_item) = Self::wait_and_get_next_work(&work_queue) {
                    Self::process_work(&work_queue, work_item);
                }
            }
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }

    fn wait_and_get_next_work(work_queue: &Arc<WorkQueue>) -> Option<WorkItem> {
        let mut queue = work_queue.queue.lock().unwrap();
        while work_queue.items_count.load(Ordering::SeqCst) == 0 {
            queue = work_queue.condvar.wait(queue).unwrap();
        }

        Self::try_pop_work_item(&mut queue, work_queue)
    }

    fn try_get_work(work_queue: &Arc<WorkQueue>) -> Option<WorkItem> {
        let mut queue = work_queue.queue.lock().unwrap();
        Self::try_pop_work_item(&mut queue, work_queue)
    }

    fn try_pop_work_item(queue: &mut VecDeque<WorkItem>, work_queue: &Arc<WorkQueue>) -> Option<WorkItem> {
        if work_queue.items_count.load(Ordering::SeqCst) > 0 {
            let item = queue.pop_front();
            if item.is_some() {
                work_queue.items_count.fetch_sub(1, Ordering::SeqCst);
            }
            item
        } else {
            None
        }
    }

    fn process_work(work_queue: &Arc<WorkQueue>, work_item: WorkItem) {
        let result = match work_item.work {
            WorkType::LoadImageBatch {dataloader, batch_number, paths} => {
                Self::load_image_batch(work_queue, dataloader, batch_number, paths)
            },
            WorkType::LoadSingleImage {path, start_idx, end_idx, data_ptr} => {
                Self::load_single_image(path, start_idx, end_idx, data_ptr)
            }
            WorkType::BuildPath {base_dir, filename} => {
                WorkResult::BuildPath {path: base_dir.join(&*filename)}
            },
        };

        work_item.future.complete(result);
    }

    fn load_image_batch(work_queue: &Arc<WorkQueue>,
        dataloader: Arc<DataLoaderForImages>,
        batch_number: usize,
        paths: Vec<PathBuf>
    ) -> WorkResult {
        let mut batch = ImageBatch::new(
            dataloader.image_total_bytes_per_batch,
            dataloader.config.batch_size,
            dataloader.image_bytes_per_image
        );

        batch.images_this_batch = paths.len();

        let data_ptr = DataPtr(batch.image_data.as_mut_ptr());


        let work_items = paths.iter().enumerate()
            .map(|(idx, path)| {
                let start = idx * dataloader.image_bytes_per_image;
                let end = start + dataloader.image_bytes_per_image;
                
                WorkType::LoadSingleImage {
                    path: path.clone(),
                    start_idx: start,
                    end_idx: end,
                    data_ptr,
                }
            })
            .collect();

        let work_batch = dataloader.config.thread_pool.submit_batch(work_items);

        // TODO: Try generalise the work while waiting pattern
        // Process other work while waiting for images to load
        while !work_batch.is_complete() {
            if let Some(other_work) = Self::try_get_work(work_queue) {
                Self::process_work(work_queue, other_work);
            }
        }

        WorkResult::LoadImageBatch {
            batch_number,
            batch
        }
    }

    fn load_single_image(path: PathBuf, start_idx: usize, end_idx: usize, data_ptr: DataPtr) -> WorkResult {
        let img = image::open(path).unwrap();
        let bytes = img.as_bytes();
        assert_eq!(bytes.len(), end_idx - start_idx);

        // SAFETY: Each task has a unique slice range, so no overlapping writes
        // TODO: Benchmark how much faster this is instead of returning each image and having parent thread combine them
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                data_ptr.0.add(start_idx),
                end_idx - start_idx
            );
        }
        
        WorkResult::LoadSingleImage
    }
}