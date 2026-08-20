use candle_core::{D, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use crate::backend::Config;
use crate::memory::KVCacheManager;

pub type MultiHeadAttention = CausalSelfAttention;

pub struct CausalSelfAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl CausalSelfAttention {
    pub fn load(vb: VarBuilder, config: &Config, _layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        let q_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let out_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        request_id: &str,
        kv_manager: &mut KVCacheManager,
        pos: usize,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let new_k = self.k_proj.forward(x)?;
        let new_v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let new_k = new_k.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let new_v = new_v.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let assigned_blocks = kv_manager
            .get_assigned_blocks(request_id)
            .ok_or_else(|| candle_core::Error::Msg(format!("Request ID {} non trouvé dans la table de pages", request_id)))?;

        let block_size = kv_manager.block_size;
        let logical_block_idx = pos / block_size;
        let physical_block_id = assigned_blocks[logical_block_idx];

        let key_entry = (layer_idx, physical_block_id);

        let full_k = match kv_manager.physical_k_cache.get(&key_entry) {
            Some(past_k) => Tensor::cat(&[past_k, &new_k], 2)?,
            None => new_k,
        };

        let full_v = match kv_manager.physical_v_cache.get(&key_entry) {
            Some(past_v) => Tensor::cat(&[past_v, &new_v], 2)?,
            None => new_v,
        };

        kv_manager.physical_k_cache.insert(key_entry, full_k.clone());
        kv_manager.physical_v_cache.insert(key_entry, full_v.clone());

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&full_k.transpose(2, 3)?)? * scale)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&full_v)?;

        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&y)
    }
}