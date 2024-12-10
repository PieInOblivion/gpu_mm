use image::ColorType;

pub struct ImageBatch {
    pub image_data: Box<[u8]>,
    pub images_this_batch: usize,
    pub bytes_per_image: usize,
    pub color_type: ColorType,
    pub batch_number: usize,
}

// TODO: Support for image label data

impl ImageBatch {
    pub fn new(total_bytes_per_batch: usize, batch_size: usize, bytes_per_image: usize, color_type: ColorType, batch_number: usize) -> ImageBatch {
        ImageBatch {
            image_data: vec![0u8; total_bytes_per_batch].into_boxed_slice(),
            images_this_batch: batch_size,
            bytes_per_image,
            color_type,
            batch_number,
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let components_per_pixel = self.color_type.channel_count() as usize;
        let bytes_per_component = self.color_type.bytes_per_pixel() as usize / components_per_pixel;
        let num_components = self.image_data.len() / bytes_per_component;
        let mut result = vec![0.0f32; num_components];

        match self.color_type {
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => {
                // Direct conversion of each u8 component to f32
                for i in 0..num_components {
                    result[i] = self.image_data[i] as f32;
                }
            },
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => {
                // Convert each u16 component to f32
                for i in 0..num_components {
                    let val = u16::from_le_bytes([
                        self.image_data[i * 2],
                        self.image_data[i * 2 + 1]
                    ]);
                    result[i] = val as f32;
                }
            },
            ColorType::Rgb32F | ColorType::Rgba32F => {
                // Each component is already f32, just need to reinterpret bytes
                for i in 0..num_components {
                    result[i] = f32::from_le_bytes([
                        self.image_data[i * 4],
                        self.image_data[i * 4 + 1],
                        self.image_data[i * 4 + 2],
                        self.image_data[i * 4 + 3],
                    ]);
                }
            },
            _ => panic!("Unsupported color type"),
        }

        result
    }
}