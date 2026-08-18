use candle_core::{D, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use crate::backend::Config;
use crate::memory::KVCacheManager;

#[derive(Clone)]
pub struct MultiHeadAttention {
    pub q_proj: Tensor,
    pub q_bias: Option<Tensor>,
    pub k_proj: Tensor,
    pub k_bias: Option<Tensor>,
    pub v_proj: Tensor,
    pub v_bias: Option<Tensor>,
    pub o_proj: Tensor,
    pub o_bias: Option<Tensor>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub layer_idx: usize,
}

impl MultiHeadAttention {
    pub fn load(vb: VarBuilder, config: &Config, layer_idx: usize) -> Result<Self> {
        let h_size = config.hidden_size;

        let q_proj = vb.get((h_size, h_size), "q_proj.weight")?;
        let q_bias = vb.get(h_size, "q_proj.bias").ok();

        let k_proj = vb.get((h_size, h_size), "k_proj.weight")?;
        let k_bias = vb.get(h_size, "k_proj.bias").ok();

        let v_proj = vb.get((h_size, h_size), "v_proj.weight")?;
        let v_bias = vb.get(h_size, "v_proj.bias").ok();

        let o_proj = vb
            .get((h_size, h_size), "out_proj.weight")
            .or_else(|_| vb.get((h_size, h_size), "o_proj.weight"))?;
        let o_bias = vb
            .get(h_size, "out_proj.bias")
            .or_else(|_| vb.get(h_size, "o_proj.bias"))
            .ok();

        let num_heads = config.num_attention_heads;
        let head_dim = config.head_dim;

        Ok(Self {
            q_proj,
            q_bias,
            k_proj,
            k_bias,
            v_proj,
            v_bias,
            o_proj,
            o_bias,
            num_heads,
            head_dim,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        request_id: &str,
        _manager: &KVCacheManager,
        vram_kv_store: &mut HashMap<String, (Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_dim) = x.dims3()?;
        let x_2d = x.reshape((b_sz * seq_len, hidden_dim))?;
    
        let mut q_2d = x_2d.matmul(&self.q_proj.t()?)?;
        if let Some(ref b) = self.q_bias { q_2d = q_2d.broadcast_add(b)?; }
    
        let mut k_2d = x_2d.matmul(&self.k_proj.t()?)?;
        if let Some(ref b) = self.k_bias { k_2d = k_2d.broadcast_add(b)?; }
    
        let mut v_2d = x_2d.matmul(&self.v_proj.t()?)?;
        if let Some(ref b) = self.v_bias { v_2d = v_2d.broadcast_add(b)?; }
    
        let q = q_2d.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let new_k = k_2d.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let new_v = v_2d.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let cache_key = format!("{}:{}", request_id, self.layer_idx);
        let (k, v) = if let Some((past_k, past_v)) = vram_kv_store.get(&cache_key) {
            let cat_k = Tensor::cat(&[past_k, &new_k], 2)?;
            let cat_v = Tensor::cat(&[past_v, &new_v], 2)?;
            (cat_k, cat_v)
        } else {
            (new_k, new_v)
        };
    
        vram_kv_store.insert(cache_key, (k.clone(), v.clone()));
    
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?;
        let scores = (q.matmul(&k_t)? * scale)?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let context = weights.matmul(&v)?;
    
        let context = context.transpose(1, 2)?.reshape((b_sz * seq_len, hidden_dim))?;
        let mut out_2d = context.matmul(&self.o_proj.t()?)?;
        if let Some(ref b) = self.o_bias {
            out_2d = out_2d.broadcast_add(b)?;
        }
    
        out_2d.reshape((b_sz, seq_len, hidden_dim))
    }
}