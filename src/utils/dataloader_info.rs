use super::dataloader_for_images::DataLoaderForImages;

pub fn print_dataset_info(dl: &DataLoaderForImages) {
    let total_size = dl.dataset.len();
    let train_size = (total_size as f32 * dl.config.train_ratio) as usize;
    let test_size = (total_size as f32 * dl.config.test_ratio) as usize;
    let val_size = total_size - train_size - test_size;

    let train_batches = (train_size + dl.config.batch_size - 1) / dl.config.batch_size;
    let test_batches = (test_size + dl.config.batch_size - 1) / dl.config.batch_size;
    let val_batches = (val_size + dl.config.batch_size - 1) / dl.config.batch_size;

    let train_batch_remainder = train_size % dl.config.batch_size;
    let test_batch_remainder = test_size % dl.config.batch_size;
    let val_batch_remainder = val_size % dl.config.batch_size;

    println!("Image Information:");
    println!("-------------------");
    println!("Colour Type: {:?}", dl.image_color_type);
    println!("Compute Format: {:?}", dl.image_compute_format);
    println!();
    println!("Dataset Information:");
    println!("-------------------");
    println!("Total size: {}", total_size);
    println!("Batch size: {}", dl.config.batch_size);
    println!();
    println!("Train split:");
    println!(
        "  Size: {} ({:.2}%)",
        train_size,
        dl.config.train_ratio * 100.0
    );
    println!("  Batches: {}", train_batches);
    println!(
        "  Last batch size: {}",
        if train_batch_remainder >= 0 {
            train_batch_remainder
        } else {
            dl.config.batch_size
        }
    );
    println!();
    println!("Test split:");
    println!(
        "  Size: {} ({:.2}%)",
        test_size,
        dl.config.test_ratio * 100.0
    );
    println!("  Batches: {}", test_batches);
    println!(
        "  Last batch size: {}",
        if test_batch_remainder >= 0 {
            test_batch_remainder
        } else {
            dl.config.batch_size
        }
    );
    println!();
    println!("Validation split:");
    println!(
        "  Size: {} ({:.2}%)",
        val_size,
        (1.0 - dl.config.train_ratio - dl.config.test_ratio) * 100.0
    );
    println!("  Batches: {}", val_batches);
    println!(
        "  Last batch size: {}",
        if val_batch_remainder >= 0 {
            val_batch_remainder
        } else {
            dl.config.batch_size
        }
    );
    println!();
    println!("Shuffle: {}", dl.config.shuffle);
    println!("Seed: {:?}", dl.config.shuffle_seed);
}
