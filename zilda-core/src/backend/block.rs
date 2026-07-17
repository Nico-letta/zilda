use candle_core::{Module, Result, Tensor};
use candle_nn::LayerNorm;
use std::collections::HashMap;

use super::attention::MultiHeadAttention;
use super::moe::SparseMoE;
use crate::memory::KVCacheManager;

#[derive(Clone)]
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub moe: SparseMoE,
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
    pub layer_idx: usize,
}

impl TransformerBlock {
    pub fn new(
        attention: MultiHeadAttention, 
        moe: SparseMoE, 
        input_layernorm: LayerNorm, 
        post_attention_layernorm: LayerNorm,
        layer_idx: usize
    ) -> Self {
        Self { attention, moe, input_layernorm, post_attention_layernorm, layer_idx }
    }

    pub fn forward(
        &self, 
        x: &Tensor, 
        request_id: &str,                          // <--- Ajouté
        cache_manager: &KVCacheManager,            // <--- Ajouté
        vram_kv_store: &mut HashMap<usize, (Tensor, Tensor)>
    ) -> Result<Tensor> {
        // 1. Attention avec Normalisation et Résiduel
        let residual = x.clone();
        let hidden_states = self.input_layernorm.forward(x)?;
        
        // On passe les arguments requis à l'attention
        let attn_output = self.attention.forward(
            &hidden_states, 
            self.layer_idx, 
            request_id, 
            cache_manager, 
            vram_kv_store
        )?;
        let hidden_states = residual.add(&attn_output)?;

        // 2. MoE avec Normalisation et Résiduel
        let residual = hidden_states.clone();
        let normalized_hidden = self.post_attention_layernorm.forward(&hidden_states)?;
        let moe_output = self.moe.forward(&normalized_hidden)?;
        let final_output = residual.add(&moe_output)?;

        Ok(final_output)
    }
}