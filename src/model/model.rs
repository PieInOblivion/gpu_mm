use super::{layer_desc::LayerDesc, layer_type::LayerType, weight_init::WeightInit};

#[derive(Clone)]
pub struct ModelDesc {
    pub layers: Vec<LayerDesc>,
    pub batch_size: usize,
    pub weight_init: WeightInit,
}

impl ModelDesc {
    pub fn new(batch_size: usize) -> Self {
        Self {
            layers: Vec::new(),
            batch_size,
            weight_init: WeightInit::He
        }
    }

    pub fn new_with(batch_size: usize, weight_init: WeightInit) -> Self {
        Self {
            layers: Vec::new(),
            batch_size,
            weight_init,
        }
    }

    pub fn add_layer(&mut self, layer_type: LayerType) {
        let layer = LayerDesc::new(layer_type);
        self.layers.push(layer);
    }

    pub fn add_layers(&mut self, layer_types: Vec<LayerType>) {
        for layer_type in layer_types.into_iter() {
            self.add_layer(layer_type);
        }
    }

    pub fn total_memory_requirements(&self) -> u64 {
        self.layers.iter()
            .map(|layer| layer.memory_requirements())
            .sum()
    }
}