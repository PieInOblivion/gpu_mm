use std::collections::HashMap;

use crate::{model::instruction::Instruction, tensor::compute_tensor::ComputeTensor};

pub struct LayerExecution {
    pub tensors: HashMap<String, ComputeTensor>,
    pub instructions: Vec<Instruction>,
    pub outputs: Vec<String>
}