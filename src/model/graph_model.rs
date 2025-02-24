use std::collections::{HashMap, HashSet};

use crate::{compute::compute_manager::ComputeLayer, dataloader::error::VKMLEngineError};

use super::{layer_desc::LayerDesc, layer_type::LayerType, weight_init::WeightInit};

pub type LayerId = usize;

pub struct GraphModel {
    pub batch_size: usize,
    pub weight_init: WeightInit,
    pub layers: HashMap<LayerId, GraphModelLayer>,
    pub verified: Option<GraphVerifiedData>
}

pub struct GraphModelLayer {
    pub id: LayerId,
    pub layer_desc: LayerDesc,
    pub weight_init: Option<WeightInit>,
    pub compute_layer: Option<ComputeLayer>,

    pub input_layers: Vec<LayerId>,
    pub output_layers: Vec<LayerId>,
}

pub struct GraphVerifiedData {
    pub entry_points: Vec<LayerId>,
    pub exit_points: Vec<LayerId>,
    pub execution_order: Vec<LayerId>,
}

impl GraphModel {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            weight_init: WeightInit::He,
            layers: HashMap::new(),
            verified: None
        }
    }

    pub fn new_with(batch_size: usize, weight_init: WeightInit) -> Self {
        Self {
            batch_size,
            weight_init,
            layers: HashMap::new(),
            verified: None
        }
    }

    pub fn add_layer(&mut self, layer_type: LayerType) -> LayerId {
        let id = self.next_available_id();

        let input_layers = if !self.layers.is_empty() {
            // Find the most recently added layer ID (highest ID that's less than current)
            let prev_id = (0..id).rev()
                .find(|&prev_id| self.layers.contains_key(&prev_id));
            
            match prev_id {
                Some(prev_id) => vec![prev_id],
                None => Vec::new()
            }
        } else {
            Vec::new()
        };

        self.add_layer_with(id, layer_type, input_layers, Vec::new(), None)
    }

    pub fn add_layers(&mut self, layer_types: Vec<LayerType>) -> Vec<LayerId> {
        let mut ids = Vec::new();
        for layer_type in layer_types.into_iter() {
            let id = self.add_layer(layer_type);
            ids.push(id);
        }
        ids
    }

    pub fn add_layer_with(&mut self, id: LayerId, layer_type: LayerType, input_layers: Vec<LayerId>, output_layers: Vec<LayerId>, weight_init: Option<WeightInit>) -> LayerId {
        // Update connections in the related layers
        for &input_id in &input_layers {
            if let Some(input_layer) = self.layers.get_mut(&input_id) {
                if !input_layer.output_layers.contains(&id) {
                    input_layer.output_layers.push(id);
                }
            }
        }
        
        for &output_id in &output_layers {
            if let Some(output_layer) = self.layers.get_mut(&output_id) {
                if !output_layer.input_layers.contains(&id) {
                    output_layer.input_layers.push(id);
                }
            }
        }

        let layer_desc = LayerDesc::new(layer_type);

        let layer = GraphModelLayer {
            id,
            layer_desc,
            weight_init,
            compute_layer: None,
            input_layers,
            output_layers,
        };

        self.layers.insert(id, layer);

        id
    }

    pub fn next_available_id(&self) -> LayerId {
        let mut id = 0;
        while self.layers.contains_key(&id) {
            id += 1;
        }
        id
    }

    pub fn total_memory_requirements(&self) -> u64 {
        self.layers.values()
            .map(|layer| layer.layer_desc.memory_requirements())
            .sum()
    }

    pub fn verify(&mut self) -> Result<(), VKMLEngineError> {
        // Identify input layers (those with LayerType::InputBuffer)
        let input_layer_ids: Vec<LayerId> = self.layers.values()
            .filter(|layer| matches!(layer.layer_desc.layer_type, LayerType::InputBuffer { .. }))
            .map(|layer| layer.id)
            .collect();

        // There should be at least one InputBuffer layer
        if input_layer_ids.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Model must have at least one InputBuffer layer".into()
            ));
        }
    
        // All InputBuffer layers should have no inputs
        let invalid_input_layers: Vec<LayerId> = self.layers.values()
            .filter(|layer| {
                matches!(layer.layer_desc.layer_type, LayerType::InputBuffer { .. }) && 
                !layer.input_layers.is_empty()
            })
            .map(|layer| layer.id)
            .collect();
    
        if !invalid_input_layers.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Input layers cannot have inputs themselves: {:?}", invalid_input_layers)
            ));
        }
    
        // Find exit points (layers with no outputs)
        let exit_points: Vec<LayerId> = self.layers.values()
            .filter(|layer| layer.output_layers.is_empty())
            .map(|layer| layer.id)
            .collect();
    
        if exit_points.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Model has no exit points".into()
            ));
        }
    
        // Verify that all referenced layers exist
        for layer in self.layers.values() {
            for &input_id in &layer.input_layers {
                if !self.layers.contains_key(&input_id) {
                    return Err(VKMLEngineError::VulkanLoadError(
                        format!("Layer {} references non-existent input layer {}", layer.id, input_id)
                    ));
                }
            }
            
            for &output_id in &layer.output_layers {
                if !self.layers.contains_key(&output_id) {
                    return Err(VKMLEngineError::VulkanLoadError(
                        format!("Layer {} references non-existent output layer {}", layer.id, output_id)
                    ));
                }
            }
        }

        // Verify bidirectional consistency of connections
        for layer in self.layers.values() {
            for &output_id in &layer.output_layers {
                if let Some(output_layer) = self.layers.get(&output_id) {
                    if !output_layer.input_layers.contains(&layer.id) {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Connection inconsistency: Layer {} lists {} as output, but {} does not list {} as input",
                                    layer.id, output_id, output_id, layer.id)
                        ));
                    }
                }
            }
            
            for &input_id in &layer.input_layers {
                if let Some(input_layer) = self.layers.get(&input_id) {
                    if !input_layer.output_layers.contains(&layer.id) {
                        return Err(VKMLEngineError::VulkanLoadError(
                            format!("Connection inconsistency: Layer {} lists {} as input, but {} does not list {} as output",
                                    layer.id, input_id, input_id, layer.id)
                        ));
                    }
                }
            }
        }

        // Verify that non-input layers have at least one input connection
        let non_input_layers_without_inputs: Vec<LayerId> = self.layers.values()
            .filter(|layer| {
                !matches!(layer.layer_desc.layer_type, LayerType::InputBuffer { .. }) && 
                layer.input_layers.is_empty()
            })
            .map(|layer| layer.id)
            .collect();

        if !non_input_layers_without_inputs.is_empty() {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Non-input layers without inputs: {:?}", non_input_layers_without_inputs)
            ));
        }
    
        // Detect cycles using a depth-first search
        if self.has_cycle() {
            return Err(VKMLEngineError::VulkanLoadError(
                "Model contains cycles".into()
            ));
        }
    
        // Generate execution order (topological sort)
        let execution_order = self.topological_sort()?;
    
        // Verify that execution order includes all layers
        if execution_order.len() != self.layers.len() {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Execution order has {} layers but model has {} layers", 
                        execution_order.len(), self.layers.len())
            ));
        }
    
        self.verified = Some(GraphVerifiedData {
            entry_points: input_layer_ids,  // Use input layers as entry points
            exit_points,
            execution_order,
        });

        Ok(())
    }

    fn topological_sort(&self) -> Result<Vec<LayerId>, VKMLEngineError> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut temp = HashSet::new();

        // Visit each node
        for &id in self.layers.keys() {
            if !visited.contains(&id) && !temp.contains(&id) {
                self.visit_node(id, &mut visited, &mut temp, &mut result)?;
            }
        }

        // Reverse the result to get the correct execution order
        result.reverse();
        Ok(result)
    }

    fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for &id in self.layers.keys() {
            if !visited.contains(&id) {
                if self.is_cyclic_util(id, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn is_cyclic_util(&self, id: LayerId, visited: &mut HashSet<LayerId>, rec_stack: &mut HashSet<LayerId>) -> bool {
        visited.insert(id);
        rec_stack.insert(id);

        if let Some(layer) = self.layers.get(&id) {
            for &next_id in &layer.output_layers {
                if !visited.contains(&next_id) {
                    if self.is_cyclic_util(next_id, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&next_id) {
                    return true;
                }
            }
        }

        rec_stack.remove(&id);
        false
    }

    fn visit_node(
        &self,
        id: LayerId,
        visited: &mut HashSet<LayerId>,
        temp: &mut HashSet<LayerId>,
        result: &mut Vec<LayerId>,
    ) -> Result<(), VKMLEngineError> {
        // If node is in temp, we have a cycle
        if temp.contains(&id) {
            return Err(VKMLEngineError::VulkanLoadError(
                format!("Cycle detected involving layer {}", id)
            ));
        }

        // If we've already visited this node, skip it
        if visited.contains(&id) {
            return Ok(());
        }

        // Mark as temporarily visited
        temp.insert(id);

        // Visit all neighbors
        if let Some(layer) = self.layers.get(&id) {
            for &next_id in &layer.output_layers {
                self.visit_node(next_id, visited, temp, result)?;
            }
        }

        // Mark as permanently visited and add to result
        temp.remove(&id);
        visited.insert(id);
        result.push(id);

        Ok(())
    }
}