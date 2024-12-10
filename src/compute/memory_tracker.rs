use crate::utils::dataloader_error::DataLoaderError;

// Always works in byte counts. Unspecific function names or variables can assume to return or use byte counts

pub struct MemoryTracker {
    maximum: u64,
    current: u64
}

impl MemoryTracker {
    pub fn new(maximum: u64) -> Self {
        Self {
            maximum,
            current: 0,
        }
    }

    pub fn allocate(&mut self, size: u64) -> Result<(), DataLoaderError> {
        let new_usage = match self.current.checked_add(size) {
            Some(usage) => usage,
            None => return Err(DataLoaderError::OutOfMemory(
                format!("Memory allocation would overflow: current {} + size {}", 
                    self.current, size)
            )),
        };
        if new_usage > self.maximum {
            return Err(DataLoaderError::OutOfMemory(
                format!("Memory limit exceeded: tried to allocate {} bytes when {} of {} bytes are used", 
                    size, self.current, self.maximum)
            ));
        }
        self.current = new_usage;
        Ok(())
    }

    pub fn deallocate(&mut self, size: u64) {
        self.current = self.current.saturating_sub(size);
    }

    pub fn get_current(&self) -> u64 {
        self.current
    }

    pub fn get_maximum(&self) -> u64 {
        self.maximum
    }

    pub fn get_available(&self) -> u64 {
        self.maximum - self.current
    }
}