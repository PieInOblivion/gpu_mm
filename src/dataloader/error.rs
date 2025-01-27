use thiserror::Error;

// TODO: Seperate concerns of error types
#[derive(Error, Debug)]
pub enum VKMLEngineError {
    // IO and System Errors
    #[error("Directory not found: {0}")]
    DirectoryNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),

    // TODO: Redo error types and such
    #[error("Invalid dataset split ratios. Train: {train}, Test: {test}")]
    InvalidSplitRatios { train: f32, test: f32 },

    #[error("No images found in the dataset")]
    EmptyDataset,

    #[error("Random number generator (shuffle_seed) not set or enabled")]
    RngNotSet,

    #[error("Failed to acquire lock on RNG")]
    RngLockError,

    // Vulkan/Ash, compute and other errors
    #[error("Vulkan/Ash error: {0}")]
    VulkanLoadError(String),

    #[error("Out of memory error: {0}")]
    OutOfMemory(String),
}