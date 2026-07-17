use candle_core::safetensors::MmapedSafetensors;
use candle_core::{Device, Tensor};
use candle_nn::{Embedding, LayerNorm};
use std::path::Path;

use super::attention::MultiHeadAttention;
use super::moe::{Expert, SparseMoE};
use super::block::TransformerBlock;
use super::ZildaMoeBackend;

fn create_layernorm(weight: Tensor, eps: f64) -> anyhow::Result<LayerNorm> {
    let bias = Tensor::zeros_like(&weight)?;
    let ln = LayerNorm::new(weight, bias, eps);
    Ok(ln)
}

pub fn load_safetensors_model<P: AsRef<Path>>(
    path: P, 
    device: &Device,
    num_layers: usize, 
    num_experts: usize,
    num_heads: usize,
    head_dim: usize,
    num_experts_per_tok: usize,
) -> anyhow::Result<ZildaMoeBackend> {
    // Chargement via mmap : très efficace, pas d'allocations inutiles
    let tensors = unsafe { MmapedSafetensors::new(path)? };

    // 1. Embedding
    let emb_tensor = tensors.load("embedding.token_embedding.weight", device)?;
    let hidden_size = emb_tensor.dim(1)?;
    let embed_tokens = Embedding::new(emb_tensor, hidden_size);

    // 2. Blocs Transformer
    let mut blocks = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        let q_proj = tensors.load(&format!("blocks.{}.attention.q_proj.weight", layer_idx), device)?;
        let k_proj = tensors.load(&format!("blocks.{}.attention.k_proj.weight", layer_idx), device)?;
        let v_proj = tensors.load(&format!("blocks.{}.attention.v_proj.weight", layer_idx), device)?;
        let out_proj = tensors.load(&format!("blocks.{}.attention.out_proj.weight", layer_idx), device)?;
        
        let attention = MultiHeadAttention::new(q_proj, k_proj, v_proj, out_proj, num_heads, head_dim);

        let gate = tensors.load(&format!("blocks.{}.moe_layer.router.weight", layer_idx), device)?;
        
        let mut experts = Vec::with_capacity(num_experts);
        for exp_idx in 0..num_experts {
            let w1 = tensors.load(&format!("blocks.{}.moe_layer.experts.{}.w1.weight", layer_idx, exp_idx), device)?;
            let w2 = tensors.load(&format!("blocks.{}.moe_layer.experts.{}.w2.weight", layer_idx, exp_idx), device)?;
            experts.push(Expert { w1, w2});
        }
        let moe = SparseMoE::new(gate, experts, num_experts_per_tok);

        let attn_norm_w = tensors.load(&format!("blocks.{}.ln1.weight", layer_idx), device)?;
        let ffn_norm_w = tensors.load(&format!("blocks.{}.ln2.weight", layer_idx), device)?;
        
        let input_layernorm = create_layernorm(attn_norm_w, 1e-5)?;
        let post_attention_layernorm = create_layernorm(ffn_norm_w, 1e-5)?;

        blocks.push(TransformerBlock::new(attention, moe, input_layernorm, post_attention_layernorm, layer_idx));
    }

    // 3. Sortie
    let norm_f_w = tensors.load("ln_f.weight", device)?;
    let norm_f = create_layernorm(norm_f_w, 1e-5)?;
    let lm_head = tensors.load("lm_head.weight", device)?;

    Ok(ZildaMoeBackend::new(embed_tokens, blocks, norm_f, lm_head))
}