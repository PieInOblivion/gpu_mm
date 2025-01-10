use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::f32::consts::PI;
use std::sync::Arc;

use crate::thread_pool::thread_pool::ThreadPool;
use crate::thread_pool::worker::{DataPtrF32, WorkType};

#[derive(Clone)]
pub enum WeightInit {
    Xavier,              // Good for tanh activation
    He,                  // Good for ReLU activation
    LeCun,              // Good for SELU activation
    UniformRandom {     // Simple uniform random in range
        min: f32,
        max: f32,
    },
    Constant(f32),
}

impl WeightInit {
    // Box-Muller transform to generate normal distribution
    pub fn normal_sample(mean: f32, std_dev: f32) -> f32 {
        let mut rng = thread_rng();
        let uniform = Uniform::new(0.0f32, 1.0);
        
        let u1 = uniform.sample(&mut rng);
        let u2 = uniform.sample(&mut rng);
        
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std_dev * z
    }

    pub fn init(&self, shape: &[usize], total_elements: usize) -> Vec<f32> {
        // For Linear layers: shape is [out_features, in_features]
        let (fan_in, fan_out) = if shape.len() == 2 {
            (shape[1], shape[0])  // in_features, out_features
        } else if shape.len() == 4 {
            // For Conv2D: [out_channels, in_channels, kernel_h, kernel_w]
            let kernel_size = shape[2] * shape[3];
            (shape[1] * kernel_size, shape[0] * kernel_size)
        } else {
            // For 1D tensors (like biases) or other shapes
            (1, shape[0])
        };
        
        match self {
            WeightInit::Xavier => {
                let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                let dist = Uniform::new(-limit, limit);
                let mut rng = thread_rng();
                (0..total_elements)
                    .map(|_| dist.sample(&mut rng))
                    .collect()
            },
            
            WeightInit::He => {
                let std_dev = (2.0 / fan_in as f32).sqrt();
                (0..total_elements)
                    .map(|_| Self::normal_sample(0.0, std_dev))
                    .collect()
            },
            
            WeightInit::LeCun => {
                let std_dev = (1.0 / fan_in as f32).sqrt();
                (0..total_elements)
                    .map(|_| Self::normal_sample(0.0, std_dev))
                    .collect()
            },
            
            WeightInit::UniformRandom { min, max } => {
                let dist = Uniform::new(*min, *max);
                let mut rng = thread_rng();
                (0..total_elements)
                    .map(|_| dist.sample(&mut rng))
                    .collect()
            },
            
            WeightInit::Constant(value) => {
                vec![*value; total_elements]
            },
        }
    }

    pub fn par_init(&self, shape: &[usize], total_elements: usize, chunk_size: usize, thread_pool: Arc<ThreadPool>) -> Vec<f32> {
        let mut result = vec![0.0; total_elements];
        let (fan_in, fan_out) = self.calculate_fan_in_out(&shape);
        let num_chunks = (total_elements + chunk_size - 1) / chunk_size;
        let data_ptr = DataPtrF32(result.as_mut_ptr());
        let mut work_items = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(total_elements);
            
            work_items.push(WorkType::WeightInitChunk {
                init_type: self.clone(),
                start_idx: start,
                end_idx: end,
                data_ptr,
                fan_in,
                fan_out,
            });
        }

        let work_batch = thread_pool.submit_batch(work_items);
        work_batch.wait();

        result
    }

    pub fn calculate_fan_in_out(&self, shape: &[usize]) -> (usize, usize) {
        match shape.len() {
            2 => (shape[1], shape[0]),  // Linear: (in_features, out_features)
            4 => {
                // Conv2D: [out_channels, in_channels, kernel_h, kernel_w]
                let kernel_size = shape[2] * shape[3];
                (shape[1] * kernel_size, shape[0] * kernel_size)
            },
            _ => (1, shape[0]), // 1D tensors or other shapes
        }
    }
}