use candle_core::{Result, Tensor};
use std::collections::HashMap;
use crate::memory::KVCacheManager;

#[derive(Clone)] // <--- Corriquet de l'erreur E0277
pub struct MultiHeadAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    out_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        out_proj: Tensor,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        }
    }
    
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        layer_idx: usize,
        request_id: &str,
        cache_manager: &KVCacheManager,
        vram_kv_store: &mut HashMap<usize, (Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // 1. Projections linéaires initiales via matmul
        let q = hidden_states.matmul(&self.q_proj)?;
        let mut k = hidden_states.matmul(&self.k_proj)?;
        let mut v = hidden_states.matmul(&self.v_proj)?;

        // 2. Récupération et mise à jour du KV Cache via la table de pages
        if let Some(allocated_blocks) = cache_manager.table_de_pages.get(request_id) {
            // Pour simplifier l'indexation par couche, on combine l'index du bloc et de la couche
            if let Some(&base_block_id) = allocated_blocks.first() {
                let physical_layer_key = base_block_id * 1000 + layer_idx;
                
                if let Some((past_k, past_v)) = vram_kv_store.get(&physical_layer_key) {
                    k = Tensor::cat(&[past_k, &k], 0)?;
                    v = Tensor::cat(&[past_v, &v], 0)?;
                }
                vram_kv_store.insert(physical_layer_key, (k.clone(), v.clone()));
            }
        }

        // 3. Produit scalaire de l'attention (Scaled Dot-Product)
        let scale = 1.0 / ((self.head_dim as f64).sqrt());
        let scores = q.matmul(&k.t()?)?.affine(scale, 0.0)?;
        let attention_weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let context = attention_weights.matmul(&v)?;

        // 4. Projection de sortie finale
        context.matmul(&self.out_proj)
    }
}