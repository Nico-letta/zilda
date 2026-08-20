use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::backend::attention::MultiHeadAttention;
use crate::backend::moe::MoEBlock;
use crate::backend::Config;
use crate::memory::KVCacheManager;

pub struct TransformerBlock {
    pub self_attn: MultiHeadAttention,
    pub ln1_weight: Tensor,
    pub ln1_bias: Tensor,
    pub ln2_weight: Tensor,
    pub ln2_bias: Tensor,
    pub moe: MoEBlock,
    pub layer_idx: usize,
}

impl TransformerBlock {
    pub fn load(vb: VarBuilder, config: &Config, layer_idx: usize) -> Result<Self> {
        let self_attn = MultiHeadAttention::load(vb.pp("attention"), config, layer_idx)?;
        let moe = MoEBlock::load(vb.pp("moe_layer"), config)?;

        let ln1_weight = vb
            .get(config.hidden_size, "ln1.weight")
            .or_else(|_| vb.get(config.hidden_size, "ln_1.weight"))?;
        let ln1_bias = vb
            .get(config.hidden_size, "ln1.bias")
            .or_else(|_| vb.get(config.hidden_size, "ln_1.bias"))?;

        let ln2_weight = vb
            .get(config.hidden_size, "ln2.weight")
            .or_else(|_| vb.get(config.hidden_size, "ln_2.weight"))?;
        let ln2_bias = vb
            .get(config.hidden_size, "ln2.bias")
            .or_else(|_| vb.get(config.hidden_size, "ln_2.bias"))?;

        Ok(Self {
            self_attn,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
            moe,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        request_id: &str,
        kv_manager: &mut KVCacheManager,
        pos: usize,
    ) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_2d = x.reshape((b * s, h))?;
        let norm1_2d = candle_nn::ops::layer_norm(&x_2d, &self.ln1_weight, &self.ln1_bias, 1e-5)?;
        let norm1 = norm1_2d.reshape((b, s, h))?;

        let attn_out = self.self_attn.forward(
            &norm1,
            request_id,
            kv_manager,
            pos,
            self.layer_idx,
        )?;
        let x = x.add(&attn_out)?;

        let (b, s, h) = x.dims3()?;
        let x_2d = x.reshape((b * s, h))?;
        let norm2_2d = candle_nn::ops::layer_norm(&x_2d, &self.ln2_weight, &self.ln2_bias, 1e-5)?;
        let norm2 = norm2_2d.reshape((b, s, h))?;

        let moe_out = self.moe.forward(&norm2)?;
        x.add(&moe_out)
    }
}