use std::collections::HashMap;

use crate::{compute::compute_manager::ComputeTensor, model::instruction::Instruction};

pub struct LayerExecution {
    pub tensors: HashMap<String, ComputeTensor>,
    pub instructions: Vec<Instruction>,
    pub outputs: Vec<String>
}