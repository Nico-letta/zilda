use candle_core::{Error, Result, Tensor};
use std::collections::HashMap;

#[allow(dead_code)]
pub struct KVCacheManager {
    pub block_size: usize,
    pub total_blocks: usize,
    pub free_blocks: Vec<usize>,
    pub table_de_pages: HashMap<String, Vec<usize>>,
    pub physical_k_cache: HashMap<(usize, usize), Tensor>,
    pub physical_v_cache: HashMap<(usize, usize), Tensor>,
}

impl Default for KVCacheManager {
    fn default() -> Self {
        let total_blocks = 128;
        Self {
            block_size: 16,
            total_blocks,
            free_blocks: (0..total_blocks).collect(),
            table_de_pages: HashMap::new(),
            physical_k_cache: HashMap::new(),
            physical_v_cache: HashMap::new(),
        }
    }
}

impl KVCacheManager {
    pub fn allocate_slots(&mut self, request_id: &str, num_tokens: usize) -> Result<()> {
        let num_blocks_needed = (num_tokens + self.block_size - 1) / self.block_size;
        if self.free_blocks.len() < num_blocks_needed {
            return Err(Error::Msg(
                "Mémoire insuffisante : plus de blocs libres disponibles".to_string(),
            ));
        }

        let assigned: Vec<usize> = self.free_blocks.drain(0..num_blocks_needed).collect();
        self.table_de_pages.insert(request_id.to_string(), assigned);
        Ok(())
    }

    pub fn free_slots(&mut self, request_id: &str) {
        if let Some(blocks) = self.table_de_pages.remove(request_id) {
            for block_id in &blocks {
                self.physical_k_cache.retain(|&(_, b), _| b != *block_id);
                self.physical_v_cache.retain(|&(_, b), _| b != *block_id);
            }
            self.free_blocks.extend(blocks);
        }
    }

    pub fn get_assigned_blocks(&self, request_id: &str) -> Option<&Vec<usize>> {
        self.table_de_pages.get(request_id)
    }
    #[allow(dead_code)]
    pub fn get_memory_usage(&self) -> f32 {
        let used = self.total_blocks - self.free_blocks.len();
        (used as f32 / self.total_blocks as f32) * 100.0
    }
}