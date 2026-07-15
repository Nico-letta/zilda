use std::collections::HashMap;
use candle_core::{Device, Result, Tensor, Module};
use candle_nn::{Linear, Embedding, LayerNorm};
use crate::memory::KVCacheManager;

struct Expert {
    w1: Linear,
    w2: Linear,
}

pub struct ZildaMoeBackend {
    device: Device,
    token_embedding: Embedding,
    lm_head: Linear,
    // Couches de normalisation manquantes
    ln1: LayerNorm,
    ln2: LayerNorm,
    ln_f: LayerNorm,
    // Attention
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    // Routeur et Experts MoE
    router_weight: Tensor,
    experts: Vec<Expert>,
    // KV Cache actif
    pub vram_kv_store: HashMap<usize, (Tensor, Tensor)>,
}

impl ZildaMoeBackend {
    pub fn new<P: AsRef<std::path::Path>>(weights_path: P) -> Result<Self> {
        let device = Device::Cpu;

        println!("[Backend MUNTU] Chargement des poids depuis {:?}...", weights_path.as_ref());
        let weights = candle_core::safetensors::load(weights_path, &device)?;

        let get_tensor = |name: &str| -> Result<Tensor> {
            weights.get(name).cloned().ok_or_else(|| {
                candle_core::Error::Msg(format!("Poids manquant dans le fichier .safetensors : {}", name))
            })
        };

        // 1. Embedding & LM Head
        let embed_weight = get_tensor("embedding.token_embedding.weight")?;
        let token_embedding = Embedding::new(embed_weight, 768); 

        let lm_head_weight = get_tensor("lm_head.weight")?;
        let lm_head = Linear::new(lm_head_weight, None);

        // 2. Chargement des LayerNorms (Epsilon standard = 1e-5)
        let ln1_weight = get_tensor("blocks.0.ln1.weight")?;
        let ln1_bias = get_tensor("blocks.0.ln1.bias")?;
        let ln1 = LayerNorm::new(ln1_weight, ln1_bias, 1e-5);

        let ln2_weight = get_tensor("blocks.0.ln2.weight")?;
        let ln2_bias = get_tensor("blocks.0.ln2.bias")?;
        let ln2 = LayerNorm::new(ln2_weight, ln2_bias, 1e-5);

        let ln_f_weight = get_tensor("ln_f.weight")?;
        let ln_f_bias = get_tensor("ln_f.bias")?;
        let ln_f = LayerNorm::new(ln_f_weight, ln_f_bias, 1e-5);

        // 3. Attention
        let q_weight = get_tensor("blocks.0.attention.q_proj.weight")?;
        let k_weight = get_tensor("blocks.0.attention.k_proj.weight")?;
        let v_weight = get_tensor("blocks.0.attention.v_proj.weight")?;
        let out_weight = get_tensor("blocks.0.attention.out_proj.weight")?;

        let q_bias = get_tensor("blocks.0.attention.q_proj.bias").ok();
        let k_bias = get_tensor("blocks.0.attention.k_proj.bias").ok();
        let v_bias = get_tensor("blocks.0.attention.v_proj.bias").ok();
        let out_bias = get_tensor("blocks.0.attention.out_proj.bias").ok();

        let q_proj = Linear::new(q_weight, q_bias);
        let k_proj = Linear::new(k_weight, k_bias);
        let v_proj = Linear::new(v_weight, v_bias);
        let out_proj = Linear::new(out_weight, out_bias);

        // 4. Routeur MoE
        let router_weight = get_tensor("blocks.0.moe_layer.router.weight")?;

        // 5. Experts
        let mut experts = Vec::new();
        for i in 0..4 {
            let w1_weight = get_tensor(&format!("blocks.0.moe_layer.experts.{}.w1.weight", i))?;
            let w1_bias = get_tensor(&format!("blocks.0.moe_layer.experts.{}.w1.bias", i)).ok();
            let w2_weight = get_tensor(&format!("blocks.0.moe_layer.experts.{}.w2.weight", i))?;
            let w2_bias = get_tensor(&format!("blocks.0.moe_layer.experts.{}.w2.bias", i)).ok();

            experts.push(Expert {
                w1: Linear::new(w1_weight, w1_bias),
                w2: Linear::new(w2_weight, w2_bias),
            });
        }

        println!("[Backend MUNTU] MoE, LayerNorms et Attention initialisés avec succès.");

        Ok(Self {
            device,
            token_embedding,
            lm_head,
            ln1,
            ln2,
            ln_f,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            router_weight,
            experts,
            vram_kv_store: HashMap::new(),
        })
    }

    fn forward_moe(&self, xs: &Tensor) -> Result<Tensor> {
        let router_logits = xs.matmul(&self.router_weight.t()?)?; 
        let gate_scores = candle_nn::ops::softmax(&router_logits, 1)?;
   
        let gate_scores_vec = gate_scores.to_vec2::<f32>()?[0].clone();
        let (best_expert_idx, &best_score) = gate_scores_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &1.0f32));

        let expert = &self.experts[best_expert_idx];

        let expert_hidden = xs.apply(&expert.w1)?;
        let expert_activated = expert_hidden.gelu()?; 
        let expert_out = expert_activated.apply(&expert.w2)?;

        expert_out.affine(best_score as f64, 0.0)
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        request_id: &str,
        cache_manager: &KVCacheManager,
    ) -> Result<Tensor> {
        // 1. Passage dans l'Embedding [1, 768]
        let input_tensor = Tensor::new(&[token_id], &self.device)?;
        let mut x = self.token_embedding.forward(&input_tensor)?;

        // --- BLOC 0 ---

        // A. Normalisation pré-attention
        let norm_x_attn = x.apply(&self.ln1)?;

        // B. Projections Attention Q, K, V
        let q = norm_x_attn.apply(&self.q_proj)?; 
        let mut k = norm_x_attn.apply(&self.k_proj)?;
        let mut v = norm_x_attn.apply(&self.v_proj)?;

        // C. Concaténation active du KV Cache
        if let Some(allocated_blocks) = cache_manager.table_de_pages.get(request_id) {
            if let Some(&current_block_id) = allocated_blocks.first() {
                if let Some((past_k, past_v)) = self.vram_kv_store.get(&current_block_id) {
                    k = Tensor::cat(&[past_k, &k], 0)?;
                    v = Tensor::cat(&[past_v, &v], 0)?;
                }
                self.vram_kv_store.insert(current_block_id, (k.clone(), v.clone()));
            }
        }

        // D. Attention sur le contexte complet
        let scale = 1.0 / ((q.dim(1)? as f64).sqrt());
        
        // Calcul des scores d'attention
        let scores = q.matmul(&k.t()?)?.affine(scale, 0.0)?;
        
        // Calcul des poids d'attention (Softmax)
        let attention_weights = candle_nn::ops::softmax(&scores, 1)?;
        
        // Calcul du contexte
        let context = attention_weights.matmul(&v)?;

        // Projection de sortie
        let attn_out = context.apply(&self.out_proj)?; 

        // Addition résiduelle (Utilisation explicite de .add pour éviter les erreurs de type)
        x = x.add(&attn_out)?;

        // --- Reste du bloc MoE ---
        let norm_x_moe = x.apply(&self.ln2)?;
        let moe_out = self.forward_moe(&norm_x_moe)?;
        x = (&x + &moe_out)?;

        // Normalisation finale
        let final_norm_x = x.apply(&self.ln_f)?;
        let logits = final_norm_x.apply(&self.lm_head)?; 

        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}