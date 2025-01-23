use crate::utils::dataloader_error::DataLoaderError;

use super::datasource::DataSource;

#[derive(Copy, Clone)]
pub enum DatasetSplit {
    Train,
    Test,
    Validation,
}

#[derive(Clone, Copy, Debug)]
pub enum ComputeFormat {
    U8,
    U16,
    F32,
}

pub trait DataLoader {
    type Item;
    type Batch;

    fn new(source: DataSource, config: Option<DataLoaderConfig>) -> Result<Self, DataLoaderError>
    where
        Self: Sized;

    fn get_batch(&self, split: DatasetSplit, batch_number: usize) -> Option<Self::Batch>;
    fn shuffle_whole_dataset(&mut self) -> Result<(), DataLoaderError>;
    fn shuffle_individual_datasets(&mut self) -> Result<(), DataLoaderError>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}