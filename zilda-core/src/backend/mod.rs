pub mod attention;
pub mod block;
pub mod moe;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use self::block::TransformerBlock;
use crate::memory::KVCacheManager;

pub fn matmul_linear(x: &Tensor, weight_t: &Tensor) -> Result<Tensor> {
    match x.dims() {
        [b, s, h] => {
            let x_2d = x.reshape((*b * *s, *h))?;
            let res_2d = x_2d.matmul(weight_t)?;
            let out_dim = weight_t.dim(1)?;
            res_2d.reshape((*b, *s, out_dim))
        }
        _ => x.matmul(weight_t),
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
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
    pub vram_kv_store: HashMap<String, (Tensor, Tensor)>,
}

pub type Model = ZildaMoeBackend;

impl ZildaMoeBackend {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        println!(
            "[Model] Initialisation du modèle ({} couches, hidden_size: {})...",
            config.num_hidden_layers, config.hidden_size
        );

        let embed_tokens = vb
            .pp("embedding")
            .pp("token_embedding")
            .get((config.vocab_size, config.hidden_size), "weight")
            .or_else(|_| vb.get((config.vocab_size, config.hidden_size), "embedding.token_embedding.weight"))?;

        let pos_embeds = vb
            .pp("embedding")
            .pp("position_embedding")
            .get((config.max_position_embeddings, config.hidden_size), "weight")
            .or_else(|_| vb.get((config.max_position_embeddings, config.hidden_size), "embedding.position_embedding.weight"))
            .ok();

        let ln_f_weight = vb.get(config.hidden_size, "ln_f.weight")
            .or_else(|_| vb.pp("ln_f").get(config.hidden_size, "weight"))?;
        let ln_f_bias = vb.get(config.hidden_size, "ln_f.bias")
            .or_else(|_| vb.pp("ln_f").get(config.hidden_size, "bias"))?;

        let lm_head = vb
            .pp("lm_head")
            .get((config.vocab_size, config.hidden_size), "weight")
            .unwrap_or_else(|_| embed_tokens.clone());

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
        pos: usize,
    ) -> Result<Tensor> {
        let device = self.embed_tokens.device();
        let input_tensor = Tensor::new(&[token_id], device)?;
        let mut hidden_states = self.embed_tokens.index_select(&input_tensor, 0)?.unsqueeze(0)?;

        if let Some(ref pos_embeds) = self.pos_embeds {
            let pos_tensor = pos_embeds.narrow(0, pos, 1)?;
            hidden_states = hidden_states.broadcast_add(&pos_tensor)?;
        }

        for block in self.blocks.iter() {
            hidden_states = block.forward(&hidden_states, request_id, manager, &mut self.vram_kv_store)?;
        }

        let hidden_2d = hidden_states.squeeze(0)?;
        let norm_hidden = candle_nn::ops::layer_norm(&hidden_2d, &self.ln_f_weight, &self.ln_f_bias, 1e-5)?;
        norm_hidden.matmul(&self.lm_head.t()?)
    }

    pub fn free_request_kv(&mut self, request_id: &str, _manager: &KVCacheManager) {
        self.vram_kv_store.retain(|key, _| !key.starts_with(request_id));
    }
}