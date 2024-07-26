use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid dataset split ratios. Train: {train}, Test: {test}")]
    InvalidSplitRatios { train: f32, test: f32 },

    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("No images found in the dataset")]
    EmptyDataset,

    #[error("Failed to build thread pool: {0}")]
    ThreadPoolBuildError(#[from] rayon::ThreadPoolBuildError),
}