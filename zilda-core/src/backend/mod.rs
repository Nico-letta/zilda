pub mod attention;
pub mod block;
pub mod loader;
pub mod moe;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use crate::memory::KVCacheManager;
use self::block::TransformerBlock;

#[derive(Clone, Debug)]
pub struct Config {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_hidden_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
            head_dim: 64,
            vocab_size: 8000,
        }
    }
}

pub struct ZildaMoeBackend {
    pub embed_tokens: Tensor,
    pub lm_head: Tensor,
    pub blocks: Vec<TransformerBlock>,
    pub vram_kv_store: HashMap<usize, (Tensor, Tensor)>,
}

impl ZildaMoeBackend {
    pub fn load(
        _vb: VarBuilder, 
        config: &Config,
        embed_tokens: Tensor,
        lm_head: Tensor,
    ) -> Result<Self> {
        println!("[Model] Initialisation des composants du modèle (config: {:?})", config);
        Ok(Self {
            embed_tokens,
            lm_head,
            blocks: Vec::new(), // Remplir avec la construction des blocs si disponible
            vram_kv_store: HashMap::new(),
        })
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        request_id: &str,
        manager: &KVCacheManager,
    ) -> Result<Tensor> {
        let device = self.embed_tokens.device();
        let input_tensor = Tensor::new(&[token_id], device)?;
    
        // 1. Embedding Lookup -> Shape (1, 1, hidden_size)
        let mut hidden_states = self.embed_tokens.index_select(&input_tensor, 0)?.unsqueeze(0)?;
    
        // 2. Inférence à travers les blocs Transformer
        for block in self.blocks.iter() {
            hidden_states = block.forward(&hidden_states, request_id, manager, &mut self.vram_kv_store)?;
        }
    
        // 3. Projection vers les logits
        // On passe hidden_states de [1, 1, 768] à [1, 768] pour autoriser le matmul avec [768, 8000]
        let hidden_2d = hidden_states.squeeze(0)?;
        let logits = hidden_2d.matmul(&self.lm_head.t()?)?;
    
        Ok(logits) // Renvoie un tenseur de forme [1, 8000]
    }

    pub fn free_request_kv(&mut self, request_id: &str, manager: &KVCacheManager) {
        if let Some(blocks) = manager.get_assigned_blocks(request_id) {
            for &block_id in blocks {
                for layer_idx in 0..100 {
                    let physical_layer_key = block_id * 1000 + layer_idx;
                    self.vram_kv_store.remove(&physical_layer_key);
                }
            }
        }
    }
}