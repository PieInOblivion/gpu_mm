use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::f32::consts::PI;

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
    fn normal_sample(mean: f32, std_dev: f32) -> f32 {
        let mut rng = thread_rng();
        let uniform = Uniform::new(0.0f32, 1.0);
        
        let u1 = uniform.sample(&mut rng);
        let u2 = uniform.sample(&mut rng);
        
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std_dev * z
    }

    pub fn init(&self, shape: &[usize]) -> Vec<f32> {
        let total_elements = shape.iter().product();
        
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
}