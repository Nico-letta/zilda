use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use crate::memory::KVCacheManager;
use crate::backend::Config;

#[derive(Clone)]
pub struct Attention {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub out_proj: Tensor,
    pub num_heads: usize,
    pub head_dim: usize,
    pub layer_idx: usize,
}

impl Attention {
    pub fn load(vb: VarBuilder, config: &Config, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let q_proj = vb.get((hidden_size, hidden_size), "q_proj.weight")?;
        let k_proj = vb.get((hidden_size, hidden_size), "k_proj.weight")?;
        let v_proj = vb.get((hidden_size, hidden_size), "v_proj.weight")?;
        let out_proj = vb.get((hidden_size, hidden_size), "o_proj.weight")
            .or_else(|_| vb.get((hidden_size, hidden_size), "out_proj.weight"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        request_id: &str,
        cache_manager: &KVCacheManager,
        vram_kv_store: &mut HashMap<usize, (Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        // 1. Projections linéaires initiales
        let q = hidden_states.matmul(&self.q_proj.t()?)?;
        let k = hidden_states.matmul(&self.k_proj.t()?)?;
        let v = hidden_states.matmul(&self.v_proj.t()?)?;

        // Redimensionnement
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let mut k = k.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let mut v = v.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        // 2. Gestion du KV Cache 
        if let Some(allocated_blocks) = cache_manager.table_de_pages.get(request_id) {
            if let Some(&base_block_id) = allocated_blocks.first() {
                let physical_layer_key = base_block_id * 1000 + self.layer_idx;

                if let Some((past_k, past_v)) = vram_kv_store.get(&physical_layer_key) {
                    k = Tensor::cat(&[past_k, &k], 2)?;
                    v = Tensor::cat(&[past_v, &v], 2)?;
                }
                vram_kv_store.insert(physical_layer_key, (k.clone(), v.clone()));
            }
        }

        // 3. Scaled Dot-Product Attention
        let scale = 1.0 / ((self.head_dim as f64).sqrt());
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let attention_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        
        let context = attention_weights.matmul(&v)?;

        // Reprojection vers la dimension cachée 
        let context = context.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        // 4. Projection de sortie finale
        context.matmul(&self.out_proj.t()?)
    }
}