pub struct Batch {
    pub data: Box<[u8]>,
    pub samples_in_batch: usize,
    pub bytes_per_sample: usize,
    pub format: DataFormat,
    pub labels: Option<Vec<LabelType>>,
    pub batch_number: usize,
}