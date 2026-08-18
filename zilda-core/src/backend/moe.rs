use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;
use crate::backend::Config;

#[derive(Clone)]
pub struct Expert {
    pub w1: Tensor,
    pub w2: Tensor,
}

impl Expert {
    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let w1 = vb.get((hidden_size * 4, hidden_size), "w1.weight")
            .or_else(|_| vb.get((hidden_size * 4, hidden_size), "gate_proj.weight"))?;
        let w2 = vb.get((hidden_size, hidden_size * 4), "w2.weight")
            .or_else(|_| vb.get((hidden_size, hidden_size * 4), "down_proj.weight"))?;

        Ok(Self { w1, w2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.w1.t()?)?;
        let x = candle_nn::ops::silu(&x)?; 
        x.matmul(&self.w2.t()?)
    }
}

#[derive(Clone)]
pub struct MoEBlock {
    pub gate: Tensor,
    pub experts: Vec<Expert>,
    pub num_experts_per_tok: usize,
}

impl MoEBlock {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let num_experts = 4;
        let num_experts_per_tok = 2;

        let gate = vb.get((num_experts, config.hidden_size), "gate.weight")
            .or_else(|_| vb.get((num_experts, config.hidden_size), "router.weight"))
            .or_else(|_| vb.pp("gate").get((num_experts, config.hidden_size), "weight"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");

        for i in 0..num_experts {
            let expert = Expert::load(vb_experts.pp(i), config.hidden_size)?;
            experts.push(expert);
        }

        Ok(Self {
            gate,
            experts,
            num_experts_per_tok,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = x.matmul(&self.gate.t()?)?;
        let weights = candle_nn::ops::softmax(&logits, D::Minus1)?;

        let mut output = Tensor::zeros_like(x)?;
        for (i, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(x)?;
            let weight = weights.narrow(D::Minus1, i, 1)?;
            let weighted_out = expert_out.broadcast_mul(&weight)?;
            output = output.add(&weighted_out)?;
        }
        Ok(output)
    }
}