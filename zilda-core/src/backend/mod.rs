pub mod attention;
pub mod block;
pub mod moe;

use candle_core::{Result, Tensor};
use candle_nn::{embedding, Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;

use crate::backend::block::TransformerBlock;
use crate::memory::KVCacheManager;

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
}

pub struct ZildaModel {
    pub embed_tokens: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub norm_weight: Tensor,
    pub norm_bias: Tensor,
    pub lm_head: Linear,
    #[allow(dead_code)]
    pub config: Config,
}

impl ZildaModel {
    pub fn load(vb: VarBuilder, config: Config) -> Result<Self> {
        let embed_vb = if vb.contains_tensor("embedding.token_embedding.weight") {
            vb.pp("embedding").pp("token_embedding")
        } else if vb.contains_tensor("embed_tokens.weight") {
            vb.pp("embed_tokens")
        } else {
            vb.pp("tok_embeddings")
        };
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, embed_vb)?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        let blocks_vb = if vb.contains_tensor("blocks.0.ln1.weight") {
            vb.pp("blocks")
        } else {
            vb.pp("layers")
        };

        for i in 0..config.num_hidden_layers {
            let block = TransformerBlock::load(blocks_vb.pp(i), &config, i)?;
            blocks.push(block);
        }

        let norm_weight = vb
            .get(config.hidden_size, "ln_f.weight")
            .or_else(|_| vb.get(config.hidden_size, "norm.weight"))?;

        let norm_bias = vb
            .get(config.hidden_size, "ln_f.bias")
            .or_else(|_| vb.get(config.hidden_size, "norm.bias"))
            .unwrap_or_else(|_| Tensor::zeros(config.hidden_size, candle_core::DType::F32, vb.device()).unwrap());

        let lm_head_vb = vb.pp("lm_head");
        let lm_head = candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, lm_head_vb)?;

        Ok(Self {
            embed_tokens,
            blocks,
            norm_weight,
            norm_bias,
            lm_head,
            config,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        request_id: &str,
        kv_manager: &mut KVCacheManager,
        pos: usize,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for block in self.blocks.iter() {
            hidden_states = block.forward(&hidden_states, request_id, kv_manager, pos)?;
        }

        let (b, s, h) = hidden_states.dims3()?;
        let hidden_2d = hidden_states.reshape((b * s, h))?;
        let norm_2d = candle_nn::ops::layer_norm(&hidden_2d, &self.norm_weight, &self.norm_bias, 1e-5)?;
        let norm_out = norm_2d.reshape((b, s, h))?;

        self.lm_head.forward(&norm_out)
    }
}