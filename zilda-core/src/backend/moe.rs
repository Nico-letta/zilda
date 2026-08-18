use candle_core::{D, Result, Tensor};
use candle_nn::VarBuilder;
use crate::backend::Config;

#[derive(Clone)]
pub struct Expert {
    pub w1: Tensor,
    pub w1_bias: Option<Tensor>,
    pub w2: Tensor,
    pub w2_bias: Option<Tensor>,
}

impl Expert {
    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let w1 = vb.get((hidden_size * 4, hidden_size), "w1.weight")?;
        let w1_bias = vb.get(hidden_size * 4, "w1.bias").ok();

        let w2 = vb.get((hidden_size, hidden_size * 4), "w2.weight")?;
        let w2_bias = vb.get(hidden_size, "w2.bias").ok();

        Ok(Self {
            w1,
            w1_bias,
            w2,
            w2_bias,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (x_2d, original_shape) = if x.rank() == 3 {
            let (b, s, h) = x.dims3()?;
            (x.reshape((b * s, h))?, Some((b, s)))
        } else {
            (x.clone(), None)
        };

        let mut h = x_2d.matmul(&self.w1.t()?)?;
        if let Some(ref b) = self.w1_bias {
            h = h.broadcast_add(b)?;
        }

        let act = candle_nn::ops::silu(&h)?;

        let mut out = act.matmul(&self.w2.t()?)?;
        if let Some(ref b) = self.w2_bias {
            out = out.broadcast_add(b)?;
        }

        if let Some((b, s)) = original_shape {
            out.reshape((b, s, self.w2.dim(0)?))
        } else {
            Ok(out)
        }
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

        let gate = vb
            .get((num_experts, config.hidden_size), "router.weight")
            .or_else(|_| vb.get((num_experts, config.hidden_size), "gate.weight"))?;

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
        let (x_2d, original_shape) = if x.rank() == 3 {
            let (b, s, h) = x.dims3()?;
            (x.reshape((b * s, h))?, Some((b, s, h)))
        } else {
            (x.clone(), None)
        };

        let logits = x_2d.matmul(&self.gate.t()?)?;
        let weights = candle_nn::ops::softmax(&logits, D::Minus1)?;

        let mut output = Tensor::zeros_like(&x_2d)?;
        for (i, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(&x_2d)?;
            let weight = weights.narrow(D::Minus1, i, 1)?;
            let weighted_out = expert_out.broadcast_mul(&weight)?;
            output = output.add(&weighted_out)?;
        }

        if let Some((b, s, h)) = original_shape {
            output.reshape((b, s, h))
        } else {
            Ok(output)
        }
    }
}