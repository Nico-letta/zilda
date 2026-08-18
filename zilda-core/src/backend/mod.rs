pub mod attention;
pub mod block;
pub mod loader;
pub mod moe;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use self::block::TransformerBlock;
use crate::memory::KVCacheManager;

#[derive(Clone, Debug)]
pub struct Config {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_hidden_layers: 4,
            hidden_size: 768,
            num_attention_heads: 12,
            head_dim: 64,
            vocab_size: 8000,
            max_position_embeddings: 512,
        }
    }
}

pub struct ZildaMoeBackend {
    pub embed_tokens: Tensor,
    pub pos_embeds: Option<Tensor>,
    pub lm_head: Tensor,
    pub ln_f_weight: Tensor,
    pub ln_f_bias: Tensor,
    pub blocks: Vec<TransformerBlock>,
    pub vram_kv_store: HashMap<usize, (Tensor, Tensor)>,
}

impl ZildaMoeBackend {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        println!(
            "[Model] Initialisation du modèle ({} couches, hidden_size: {})...",
            config.num_hidden_layers, config.hidden_size
        );

        // 1. Embedding de tokens
        let embed_tokens = vb
            .pp("embedding")
            .pp("token_embedding")
            .get((config.vocab_size, config.hidden_size), "weight")
            .or_else(|_| vb.get((config.vocab_size, config.hidden_size), "embedding.token_embedding.weight"))?;

        // 2. Embedding de positions (Optionnel)
        let pos_embeds = vb
            .pp("embedding")
            .pp("position_embedding")
            .get((config.max_position_embeddings, config.hidden_size), "weight")
            .or_else(|_| vb.get((config.max_position_embeddings, config.hidden_size), "embedding.position_embedding.weight"))
            .ok();

        // 3. LayerNorm finale (ln_f)
        let ln_f_weight = vb.get(config.hidden_size, "ln_f.weight")
            .or_else(|_| vb.pp("ln_f").get(config.hidden_size, "weight"))?;
        let ln_f_bias = vb.get(config.hidden_size, "ln_f.bias")
            .or_else(|_| vb.pp("ln_f").get(config.hidden_size, "bias"))?;

        // 4. Head de sortie (Réutilisation des poids d'embeddings)
        let lm_head = vb
            .pp("lm_head")
            .get((config.vocab_size, config.hidden_size), "weight")
            .unwrap_or_else(|_| embed_tokens.clone());

        // 5. Blocs Transformer
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("blocks");

        for layer_idx in 0..config.num_hidden_layers {
            let vb_block = vb_layers.pp(layer_idx);
            let block = TransformerBlock::load(vb_block, config, layer_idx)?;
            blocks.push(block);
        }

        println!("[Model] Chargement réussi de {} blocs Transformer.", blocks.len());

        Ok(Self {
            embed_tokens,
            pos_embeds,
            lm_head,
            ln_f_weight,
            ln_f_bias,
            blocks,
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

        // Lookup de l'embedding de token -> Shape [1, 1, hidden_size]
        let mut hidden_states = self.embed_tokens.index_select(&input_tensor, 0)?.unsqueeze(0)?;

        // Passage à travers les 12 blocs Transformer
        for block in self.blocks.iter() {
            hidden_states = block.forward(&hidden_states, request_id, manager, &mut self.vram_kv_store)?;
        }

        // LayerNorm finale avant la projection des logits
        let hidden_2d = hidden_states.squeeze(0)?;
        let norm_hidden = candle_nn::ops::layer_norm(&hidden_2d, &self.ln_f_weight, &self.ln_f_bias, 1e-5)?;

        // Projection vers le vocabulaire -> Shape [1, vocab_size]
        norm_hidden.matmul(&self.lm_head.t()?)
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