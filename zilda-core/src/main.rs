use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::io::{self, Write};
use std::path::Path;
use tokenizers::Tokenizer;

mod backend;
mod memory;

use backend::{Config, ZildaModel};
use memory::KVCacheManager;

fn apply_repetition_penalty(
    logits: &Tensor,
    generated_tokens: &[u32],
    penalty: f32,
) -> Result<Tensor> {
    let mut logits_vec = logits.to_vec1::<f32>()?;

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

    Tensor::from_vec(logits_vec, logits.shape(), logits.device())
}

/// Échantillonnage combinant Température, Top-K et Top-P (Nucleus)
fn sample_token(
    logits: &Tensor,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<u32> {
    let logits_vec = logits.to_vec1::<f32>()?;

    // Si température presque nulle, retour sur un Argmax déterministe
    if temperature <= 1e-5 {
        let (max_idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        return Ok(max_idx as u32);
    }

    // 1. Application de la Température
    let mut indexed_logits: Vec<(usize, f32)> = logits_vec
        .into_iter()
        .map(|val| val / temperature)
        .enumerate()
        .collect();

    // Tri décroissant par logit
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 2. Filtrage Top-K (considérer uniquement les K meilleurs tokens)
    if top_k > 0 && top_k < indexed_logits.len() {
        indexed_logits.truncate(top_k);
    }

    // 3. Calcul du Softmax
    let max_logit = indexed_logits[0].1;
    let mut exp_sum = 0.0f32;
    let mut probs: Vec<(usize, f32)> = indexed_logits
        .into_iter()
        .map(|(idx, logit)| {
            let exp = (logit - max_logit).exp();
            exp_sum += exp;
            (idx, exp)
        })
        .collect();

    for item in probs.iter_mut() {
        item.1 /= exp_sum;
    }

    // 4. Filtrage Top-P / Nucleus (conserver la masse de probabilité >= top_p)
    if top_p > 0.0 && top_p < 1.0 {
        let mut cum_sum = 0.0f32;
        let mut cutoff_idx = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            cum_sum += p;
            if cum_sum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        probs.truncate(cutoff_idx);

        // Renormalisation des probabilités restantes
        let new_sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for item in probs.iter_mut() {
            item.1 /= new_sum;
        }
    }

    // 5. Sélection probabiliste
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(123456);
    let random_sample = (nanos % 10_000) as f32 / 10_000.0;

    let mut cumulative = 0.0f32;
    for (idx, prob) in probs.iter() {
        cumulative += prob;
        if random_sample <= cumulative {
            return Ok(*idx as u32);
        }
    }

    Ok(probs.first().map(|(idx, _)| *idx as u32).unwrap_or(0))
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
        num_hidden_layers: 4,
        vocab_size: 8000,
        num_local_experts: 4,
        num_experts_per_tok: 2,
    };

    let model = ZildaModel::load(vb, config)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let request_id = "req_001";
    let prompt = "Muntu LM";
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let prompt_tokens = encoding.get_ids();
    println!("Prompt tokenisé (IDs) : {:?}", prompt_tokens);

    let max_new_tokens = 50;
    let prompt_len = prompt_tokens.len();
    let total_expected_tokens = prompt_len + max_new_tokens;

    let mut kv_manager = KVCacheManager::default();
    kv_manager.allocate_slots(request_id, total_expected_tokens)?;

    println!("\n--- Début de la génération ---");
    print!("{}", prompt);
    io::stdout().flush().ok();

    let mut last_logits = None;
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        let input_tensor = Tensor::new(&[token_id], &device)?.unsqueeze(0)?;
        last_logits = Some(model.forward(&input_tensor, request_id, &mut kv_manager, pos)?);
    }

    let current_logits_3d = match last_logits {
        Some(logits) => logits,
        None => return Ok(()),
    };

    let mut current_logits = current_logits_3d.squeeze(0)?.squeeze(0)?;

    // --- Hyperparamètres d'échantillonnage pour réduire le bruit ---
    let repetition_penalty = 1.15f32;
    let temperature = 0.0f32; // Basse température pour plus de cohérence
    let top_k = 40;            // Limite aux 40 tokens les plus probables
    let top_p = 0.85f32;       // Ne garde que le noyau à 85% de masse de probabilité

    let mut generated_history = prompt_tokens.to_vec();

    for i in 0..max_new_tokens {
        let pos = prompt_len + i;

        let penalized_logits = apply_repetition_penalty(
            &current_logits,
            &generated_history,
            repetition_penalty,
        )?;

        let next_token_id = sample_token(
            &penalized_logits,
            temperature,
            top_k,
            top_p,
        )?;

        if next_token_id == 2 {
            break;
        }

        // Au lieu de décoder token par token dans la boucle :
        generated_history.push(next_token_id);

        // Décodage propre de l'ensemble des nouveaux tokens générés
        if let Ok(full_text) = tokenizer.decode(&generated_history[prompt_len..], true) {
            // Efface la ligne précédente en console et réimprime le texte fluide
            print!("\r{}", full_text);
            io::stdout().flush().ok();
        }

        let input_tensor = Tensor::new(&[next_token_id], &device)?.unsqueeze(0)?;
        let logits_3d = model.forward(&input_tensor, request_id, &mut kv_manager, pos)?;
        current_logits = logits_3d.squeeze(0)?.squeeze(0)?;
    }

    kv_manager.free_slots(request_id);

    println!("\n\n[Zilda] Génération terminée.");
    Ok(())
}