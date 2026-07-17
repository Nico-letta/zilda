use candle_core::{Result, Tensor, D};

#[derive(Clone)]
pub struct Expert {
    pub w1: Tensor,
    pub w2: Tensor,
}

impl Expert {
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Architecture MLP standard : (x * w1) -> activation -> (result * w2)
        // Utilisation de SiLU (Swish) comme activation, standard pour les modèles modernes.
        // Si ton modèle utilise autre chose (ReLU, GeLU), remplace simplement `silu`.
        let x = x.matmul(&self.w1.t()?)?;
        let x = candle_nn::ops::silu(&x)?; 
        x.matmul(&self.w2.t()?)
    }
}

#[derive(Clone)]
pub struct SparseMoE {
    pub gate: Tensor,
    pub experts: Vec<Expert>,
    pub num_experts_per_tok: usize,
}

impl SparseMoE {
    pub fn new(gate: Tensor, experts: Vec<Expert>, num_experts_per_tok: usize) -> Self {
        Self { gate, experts, num_experts_per_tok }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let x_flat = x.reshape((b_sz * seq_len, hidden_size))?;

        // Routage standard via les poids du Gate
        let router_logits = x_flat.matmul(&self.gate.t()?)?;
        let routing_weights = candle_nn::ops::softmax(&router_logits, D::Minus1)?;
        
        // Calcul du Top-K en Rust pur (identique à avant, mais sur tenseur classique)
        let routing_weights_vec = routing_weights.to_vec2::<f32>()?;
        let mut topk_indices_vec = Vec::with_capacity(routing_weights_vec.len());
        let mut topk_weights_vec = Vec::with_capacity(routing_weights_vec.len());

        for weights in routing_weights_vec.iter() {
            let mut indexed_weights: Vec<(u32, f32)> = weights
                .iter()
                .enumerate()
                .map(|(i, &w)| (i as u32, w))
                .collect();
            
            indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut token_indices = Vec::with_capacity(self.num_experts_per_tok);
            let mut token_weights = Vec::with_capacity(self.num_experts_per_tok);
            
            for i in 0..self.num_experts_per_tok {
                if let Some(&(idx, w)) = indexed_weights.get(i) {
                    token_indices.push(idx);
                    token_weights.push(w);
                }
            }
            
            topk_indices_vec.push(token_indices);
            topk_weights_vec.push(token_weights);
        }

        let mut final_output = x_flat.zeros_like()?;

        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let mut token_indices_for_expert = Vec::new();
            let mut weights_for_expert = Vec::new();

            for (token_id, expert_choices) in topk_indices_vec.iter().enumerate() {
                for (k_idx, &chosen_expert) in expert_choices.iter().enumerate() {
                    if chosen_expert as usize == expert_idx {
                        token_indices_for_expert.push(token_id as u32);
                        weights_for_expert.push(topk_weights_vec[token_id][k_idx]);
                    }
                }
            }

            if !token_indices_for_expert.is_empty() {
                let device = x.device();
                let indices_tensor = Tensor::new(token_indices_for_expert.as_slice(), device)?;
                let expert_input = x_flat.index_select(&indices_tensor, 0)?;
                let mut expert_output = expert.forward(&expert_input)?;

                let weights_tensor = Tensor::new(weights_for_expert.as_slice(), device)?.reshape(((), 1))?;
                expert_output = expert_output.broadcast_mul(&weights_tensor)?;
                final_output = final_output.index_add(&indices_tensor, &expert_output, 0)?;
            }
        }

        final_output.reshape((b_sz, seq_len, hidden_size))
    }
}