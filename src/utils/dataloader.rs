use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;
use std::io::Error;
use std::path::Path;

use image::{self, ColorType};

#[derive(Debug)]
pub struct DataLoaderForImages {
    dir: Option<String>,
    pub dataset_size: usize,
    pub largest_width: u32,
    pub largest_height: u32,
    pub largest_depth: ColorType,
    pub opt_threads: u32,
    pub opt_batch_size: usize,
    pub opt_train_size: f32,
    pub opt_test_size: f32,
    pub opt_shuffle: bool,
    pub opt_shuffle_seed: Option<u64>,
    pub opt_drop_last: bool,
    scanned_largest: bool,
    pub train_dataset: Vec<String>,
    pub test_dataset: Vec<String>,
    pub val_dataset: Vec<String>,
}

impl Default for DataLoaderForImages {
    fn default() -> Self {
        DataLoaderForImages {
            dir: None,
            dataset_size: 0,
            largest_width: 0,
            largest_height: 0,
            largest_depth: ColorType::L8,
            opt_threads: 0,
            opt_batch_size: 32,
            opt_train_size: 1.0,
            opt_test_size: 0.0,
            opt_shuffle: true,
            opt_shuffle_seed: None,
            opt_drop_last: false,
            scanned_largest: false,
            train_dataset: Vec::new(),
            test_dataset: Vec::new(),
            val_dataset: Vec::new(),
        }
    }
}

impl DataLoaderForImages {
    pub fn new(dir: &str) -> Result<Self, Error> {
        let path = Path::new(dir);
        if !path.exists() {
            return Err(Error::new(
                std::io::ErrorKind::NotFound,
                "Directory not found",
            ));
        }

        let mut new_loader = DataLoaderForImages {
            dir: Some(dir.to_owned()),
            ..Default::default()
        };

        for entry in fs::read_dir(&path)? {
            let path = entry?.path();
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_str().unwrap().to_lowercase();
                if ext_str == "png" || ext_str == "jpg" || ext_str == "jpeg" {
                    if let Ok(img) = image::open(&path) {
                        new_loader.largest_width = img.width();
                        new_loader.largest_height = img.height();
                        new_loader.largest_depth = img.color();
                        break;
                    }
                }
            }
        }

        Ok(new_loader)
    }

    pub fn check_for_largest_image(&mut self) -> Result<(), Error> {
        if self.scanned_largest {
            return Ok(());
        }

        let path = Path::new(self.dir.as_ref().unwrap());

        for entry in fs::read_dir(&path)? {
            let path = entry?.path();
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_str().unwrap().to_lowercase();
                if ext_str == "png" || ext_str == "jpg" || ext_str == "jpeg" {
                    if let Ok(img) = image::open(&path) {
                        let depth = img.color();
                        self.largest_width = self.largest_width.max(img.width());
                        self.largest_height = self.largest_height.max(img.height());

                        if depth.bits_per_pixel() > self.largest_depth.bits_per_pixel() {
                            self.largest_depth = depth;
                        }

                        self.dataset_size += 1;
                    }
                }
            }
        }

        self.scanned_largest = true;
        Ok(())
    }

    pub fn split_dataset(&mut self) -> Result<(), Error> {
        let path = Path::new(self.dir.as_ref().unwrap());
        let mut image_paths: Vec<String> = Vec::new();

        for entry in fs::read_dir(&path)? {
            let path = entry?.path();
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_str().unwrap().to_lowercase();
                if ext_str == "png" || ext_str == "jpg" || ext_str == "jpeg" {
                    if let Some(filename) = path.file_name() {
                        image_paths.push(filename.to_str().unwrap().to_string());
                    }
                }
            }
        }

        if self.opt_shuffle {
            if self.opt_shuffle_seed.is_none() {
                self.opt_shuffle_seed = Some(rand::thread_rng().gen());
            }
            let mut rng = StdRng::seed_from_u64(self.opt_shuffle_seed.unwrap());
            image_paths.shuffle(&mut rng);
        }

        let total_size = image_paths.len();
        let train_size = (total_size as f32 * self.opt_train_size).round() as usize;
        let test_size = (total_size as f32 * self.opt_test_size).round() as usize;

        self.train_dataset = image_paths[..train_size].to_vec();
        self.test_dataset = image_paths[train_size..train_size + test_size].to_vec();
        self.val_dataset = image_paths[train_size + test_size..].to_vec();

        Ok(())
    }
}
