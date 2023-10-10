mod utils;

use cudarc::driver::result;
use std::sync::{Arc, RwLock};

fn main() {
    dbg!(utils::structs::GPU::new(0));
}
