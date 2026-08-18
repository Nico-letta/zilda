use candle_core::{D, Device, Result, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use std::path::Path;

mod backend;
mod memory;

use backend::{Config, Model};
use memory::KVCacheManager;

fn apply_repetition_penalty(
    logits: &Tensor,
    generated_tokens: &[u32],
    penalty: f32,
) -> Result<Tensor> {
    let is_2d = logits.rank() == 2;
    let logits_1d = if is_2d { logits.squeeze(0)? } else { logits.clone() };
    let mut logits_vec = logits_1d.to_vec1::<f32>()?;

    for &token_id in generated_tokens {
        let idx = token_id as usize;
        if idx < logits_vec.len() {
            if logits_vec[idx] < 0.0 {
                logits_vec[idx] *= penalty;
            } else {
                logits_vec[idx] /= penalty;
            }
        }
    }

    let penalized_1d = Tensor::from_vec(logits_vec, logits_1d.shape(), logits.device())?;
    if is_2d {
        penalized_1d.unsqueeze(0)
    } else {
        Ok(penalized_1d)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let model_path = "../data/muntu_pretrained.safetensors";
    let tokenizer_path = "../data/tokenizer.json";

    println!("[Zilda] Recherche du modèle sur : \"{}\"", model_path);
    if !Path::new(model_path).exists() {
        eprintln!("[Zilda] Erreur : Fichier modèle introuvable à \"{}\"", model_path);
        return Ok(());
    }

    println!("[Loader] Chargement de Safetensors depuis \"{}\"", model_path);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &device)?
    };

    let config = Config {
        hidden_size: 768,
        num_attention_heads: 12,
        head_dim: 64,
        num_hidden_layers: 4,
        vocab_size: 8000,
        max_position_embeddings: 2048,
    };

    let mut model = Model::load(vb, &config)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let kv_manager = KVCacheManager::default();
    let request_id = "req_001";

    let prompt = "Muntu LM";
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let prompt_tokens = encoding.get_ids();
    println!("Prompt tokenisé (IDs) : {:?}", prompt_tokens);

    println!("\n--- Début de la génération ---");
    print!("{}", prompt);
    io::stdout().flush().ok();

    let mut last_logits = None;
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        last_logits = Some(model.forward_token(token_id, request_id, &kv_manager, pos)?);
    }

    let mut current_logits = match last_logits {
        Some(logits) => logits,
        None => return Ok(()),
    };

    let max_new_tokens = 50;
    let prompt_len = prompt_tokens.len();
    let repetition_penalty = 1.25f32;
    let temperature = 0.7f32;

    let mut generated_history = prompt_tokens.to_vec();

    for i in 0..max_new_tokens {
        let pos = prompt_len + i;

        let penalized_logits = apply_repetition_penalty(
            &current_logits,
            &generated_history,
            repetition_penalty,
        )?;

        let scaled_logits = (&penalized_logits / (temperature as f64))?;

        let next_token_id = scaled_logits
            .argmax(D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;

        if next_token_id == 2 {
            break;
        }

        generated_history.push(next_token_id);

        if let Ok(token_str) = tokenizer.decode(&[next_token_id], true) {
            print!("{}", token_str);
            io::stdout().flush().ok();
        }

        current_logits = model.forward_token(next_token_id, request_id, &kv_manager, pos)?;
    }

    model.free_request_kv(request_id, &kv_manager);

    println!("\n\n[Zilda] Génération terminée.");
    Ok(())
}