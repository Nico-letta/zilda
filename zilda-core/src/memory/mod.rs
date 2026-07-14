use std::collections::HashMap;


#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CacheBlock {
    pub block_id: usize,
    pub allocated_tokens: usize,
    pub max_capacity: usize,
}

pub struct KVCacheManager {
    pub block_size: usize,
    pub total_blocks: usize,
    pub free_blocks: Vec<usize>,
    pub table_de_pages: HashMap<String, Vec<usize>>,
}

impl KVCacheManager {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        let free_blocks = (0..total_blocks).rev().collect();
        KVCacheManager {
            block_size,
            total_blocks,
            free_blocks,
            table_de_pages: HashMap::new(),
        }
    }

    pub fn allocate_slots(&mut self, request_id: &str, num_tokens: usize) -> Result<(), String> {
        let blocs_requis = (num_tokens + self.block_size - 1) / self.block_size;

        if blocs_requis > self.free_blocks.len() {
            return Err(format!(
                "Hors de mémoire (OOM) : Impossible d'allouer {} blocs. Blocs libres restants : {}",
                blocs_requis, self.free_blocks.len()
            ));
        }

        let mut allocated_ids = Vec::new();
        for _ in 0..blocs_requis {
            if let Some(id) = self.free_blocks.pop() {
                allocated_ids.push(id);
            }
        }

        self.table_de_pages.insert(request_id.to_string(), allocated_ids);
        Ok(())
    }

    pub fn free_slots(&mut self, request_id: &str) {
        if let Some(block_ids) = self.table_de_pages.remove(request_id) {
            for id in block_ids {
                self.free_blocks.push(id);
            }
        }
    }

    pub fn get_memory_usage(&self) -> f32 {
        let blocs_occupes = self.total_blocks - self.free_blocks.len();
        (blocs_occupes as f32 / self.total_blocks as f32) * 100.0
    }
}