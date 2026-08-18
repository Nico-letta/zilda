use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use crate::backend::Config;
use crate::backend::attention::Attention;
use crate::backend::moe::MoEBlock;
use crate::memory::KVCacheManager;

pub struct TransformerBlock {
    pub self_attn: Attention,
    pub moe: MoEBlock,
    pub layer_idx: usize,
}

impl TransformerBlock {
    pub fn load(vb: VarBuilder, config: &Config, layer_idx: usize) -> Result<Self> {
        let self_attn = Attention::load(vb.pp("attention"), config, layer_idx)?;
        let moe = MoEBlock::load(vb.pp("moe_layer"), config)?;

        Ok(Self {
            self_attn,
            moe,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        request_id: &str,
        manager: &KVCacheManager,
        vram_kv_store: &mut HashMap<usize, (Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let attn_out = self.self_attn.forward(hidden_states, request_id, manager, vram_kv_store)?;
        let residual = hidden_states.add(&attn_out)?;

        let moe_out = self.moe.forward(&residual)?;
        residual.add(&moe_out)
    }
}