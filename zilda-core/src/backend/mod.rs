pub mod attention;
pub mod block;
pub mod loader;
pub mod moe;

use candle_core::{Device, Tensor};
use candle_nn::{Embedding, LayerNorm, Module};
use std::path::Path;
use std::collections::HashMap;
use crate::memory::KVCacheManager;

pub struct ZildaMoeBackend {
    pub embed_tokens: Embedding,
    pub layers: Vec<block::TransformerBlock>,
    pub norm: LayerNorm,
    pub lm_head: Tensor,
    pub vram_kv_store: HashMap<usize, (Tensor, Tensor)>,
}

impl ZildaMoeBackend {
    pub fn new(
        embed_tokens: Embedding,
        layers: Vec<block::TransformerBlock>,
        norm: LayerNorm,
        lm_head: Tensor,
    ) -> Self {
        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vram_kv_store: HashMap::new(),
        }
    }

    /// Point d'entrée appelé par main.rs pour charger le modèle complet
    pub fn load<P: AsRef<Path>>(
        path: P, 
        device: &Device,
        num_layers: usize, 
        num_experts: usize,
        num_heads: usize,
        head_dim: usize,
        num_experts_per_tok: usize,
    ) -> anyhow::Result<Self> {
        // On transmet simplement tous les arguments reçus au loader réel
        loader::load_safetensors_model(
            path, 
            device, 
            num_layers, 
            num_experts, 
            num_heads, 
            head_dim, 
            num_experts_per_tok
        )
    }

    // Dans zilda-core\src\backend\mod.rs

    pub fn forward_token(
        &mut self,
        token_id: u32,
        request_id: &str,
        cache_manager: &KVCacheManager,
    ) -> anyhow::Result<Tensor> {
        let device = self.lm_head.device();
        let input_tensor = Tensor::new(&[token_id], device)?;
        
        // 1. Passage dans la couche d'Embedding [1, hidden_size]
        let x = self.embed_tokens.forward(&input_tensor)?;
        
        // --- FIX : On ajoute la dimension de séquence [1, 1, hidden_size] ---
        let mut x = x.unsqueeze(1)?; 

        // 2. Traitement séquentiel
        for layer in &mut self.layers {
            x = layer.forward(&x, request_id, cache_manager, &mut self.vram_kv_store)?;
        }

        // 3. Normalisation finale
        let final_norm_x = x.apply(&self.norm)?;

        // Note : Ici final_norm_x est en [1, 1, hidden_size]
        // Si ton lm_head attend [hidden_size], il faut reprendre le dernier token
        let logits = final_norm_x.squeeze(1)?.matmul(&self.lm_head)?; 

        Ok(logits)
    }
}