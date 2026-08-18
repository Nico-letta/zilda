use std::collections::HashSet;
use rand::distr::{weighted::WeightedIndex, Distribution};

pub struct Sampler;

impl Sampler {
    pub fn sample(
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        generated_tokens: &[u32],
    ) -> u32 {
        let mut logits_clone = logits.to_vec();

        if repetition_penalty != 1.0 && !generated_tokens.is_empty() {
            let unique_tokens: HashSet<&u32> = generated_tokens.iter().collect();
            for &&token_id in &unique_tokens {
                let idx = token_id as usize;
                if idx < logits_clone.len() {
                    let logit = logits_clone[idx];
                    logits_clone[idx] = if logit > 0.0 {
                        logit / repetition_penalty
                    } else {
                        logit * repetition_penalty
                    };
                }
            }
        }

        if temperature <= 0.0 {
            return logits_clone
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
        }

        for logit in logits_clone.iter_mut() {
            *logit /= temperature;
        }

        let mut indexed_logits: Vec<(usize, f32)> = logits_clone.into_iter().enumerate().collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_logit = indexed_logits[0].1;
        let exp_logits: Vec<f32> = indexed_logits.iter().map(|(_, l)| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let mut probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

        if top_p > 0.0 && top_p < 1.0 {
            let mut cumulative_prob = 0.0;
            let mut cutoff_idx = probs.len();

            for (i, &p) in probs.iter().enumerate() {
                cumulative_prob += p;
                if cumulative_prob > top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            indexed_logits.truncate(cutoff_idx);
            probs.truncate(cutoff_idx);

            let sum_probs: f32 = probs.iter().sum();
            if sum_probs > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum_probs;
                }
            } else {
                probs = vec![1.0];
                indexed_logits.truncate(1);
            }
        }

        let mut rng = rand::rng();
        if let Ok(dist) = WeightedIndex::new(&probs) {
            let sampled_idx = dist.sample(&mut rng);
            indexed_logits[sampled_idx].0 as u32
        } else {
            indexed_logits[0].0 as u32
        }
    }
}