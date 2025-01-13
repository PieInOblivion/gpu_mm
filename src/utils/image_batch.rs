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

        let mut result = Vec::with_capacity(num_components);
        unsafe {
            result.set_len(num_components);
        }

        match self.color_type {
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => {
                for (i, &x) in self.image_data.iter().enumerate() {
                    result[i] = x as f32;
                }
            }
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => {
                for (i, chunk) in self.image_data.chunks_exact(2).enumerate() {
                    result[i] = u16::from_le_bytes([chunk[0], chunk[1]]) as f32;
                }
            }
            ColorType::Rgb32F | ColorType::Rgba32F => {
                for (i, chunk) in self.image_data.chunks_exact(4).enumerate() {
                    result[i] =
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            _ => panic!("Unsupported colour type in to_f32 ImageBatch"),
        }

        result
    }
}