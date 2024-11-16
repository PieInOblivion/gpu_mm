pub struct ImageBatch {
    pub image_data: Box<[u8]>,
    pub images_this_batch: usize,
    pub bytes_per_image: usize,
}

impl ImageBatch {
    pub fn new(total_bytes_per_batch: usize, batch_size: usize, bytes_per_image: usize) -> ImageBatch {
        ImageBatch {
            image_data: vec![0u8; total_bytes_per_batch].into_boxed_slice(),
            images_this_batch: batch_size,
            bytes_per_image
        }
    }
}

pub struct IteratorImageBatch {
    pub image_data: &'static [u8],
    pub images_this_batch: usize,
    pub bytes_per_image: usize,
    pub batch_number: usize, 
}