use std::collections::HashMap;
use candle_core::{DType, Device, Result, Tensor}; // 'Module' supprimé car inutilisé
use candle_nn::{linear, Linear, VarMap}; // 'Vocabulary' supprimé car inexistant ici
use crate::memory::KVCacheManager;

pub struct ZildaLinearBackend {
    device: Device,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    pub vram_kv_store: HashMap<usize, (Tensor, Tensor)>,
}

impl ZildaLinearBackend {
    pub fn new(in_dim: usize, out_dim: usize) -> Result<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let q_proj = linear(in_dim, 64, vs.pp("q"))?;
        let k_proj = linear(in_dim, 64, vs.pp("k"))?;
        let v_proj = linear(in_dim, 64, vs.pp("v"))?;
        let out_proj = linear(64, out_dim, vs.pp("out"))?;

        Ok(Self {
            device,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            vram_kv_store: HashMap::new(),
        })
    }
}

impl ZildaLinearBackend {
    pub fn forward_attention(
        &mut self, 
        input: &Tensor, 
        request_id: &str, 
        cache_manager: &KVCacheManager
    ) -> Result<Tensor> {
        let q = input.apply(&self.q_proj)?; // [1, 64]
        let k = input.apply(&self.k_proj)?; // [1, 64]
        let v = input.apply(&self.v_proj)?; // [1, 64]

        if let Some(allocated_blocks) = cache_manager.table_de_pages.get(request_id) {
            if let Some(&current_block_id) = allocated_blocks.first() {
                self.vram_kv_store.insert(current_block_id, (k.clone(), v.clone()));
            }
        }

        let scale = 1.0 / (64.0f64).sqrt();
        let k_t = k.t()?;
        let scores = q.matmul(&k_t)?;
        
        // CORRECTION : Utilisation de la méthode native .affine(multiplier, offset) de Candle
        let scores = scores.affine(scale, 0.0)?; 
        
        let attention_weights = candle_nn::ops::softmax(&scores, 1)?;

        let context = attention_weights.matmul(&v)?;

        context.apply(&self.out_proj)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}