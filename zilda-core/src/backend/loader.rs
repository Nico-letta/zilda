use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use std::path::Path;

use super::{Config, ZildaMoeBackend};

/// Charge les poids du modèle à partir d'un fichier `.safetensors`
pub fn load_safetensors_model<P: AsRef<Path>>(
    path: P,
    device: &Device,
    config: &Config,
) -> Result<ZildaMoeBackend> {
    println!("[Loader] Chargement brut du fichier safetensors : {:?}", path.as_ref());
    let weights = candle_core::safetensors::load(path.as_ref(), device)?;

    println!("[Loader] Recherche des poids d'embedding et de la tête de sortie...");
    
    // Récupération avec le nom exact du checkpoint MUNTU
    let embed_tokens_weight = match weights.get("embedding.token_embedding.weight") {
        Some(w) => w.clone(),
        None => {
            // Fallback générique au cas où
            weights.get("model.embed_tokens.weight")
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg("Embedding weight 'embedding.token_embedding.weight' non trouvé".into()))?
        }
    };

    // Tête de sortie (tied weights avec token_embedding)
    let lm_head_weight = match weights.get("lm_head.weight") {
        Some(w) => w.clone(),
        None => {
            println!("[Loader] Utilisation de 'embedding.token_embedding.weight' pour la tête de sortie (Tied Weights).");
            embed_tokens_weight.clone()
        }
    };

    // Construction du VarBuilder (mmaped_safetensors)
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[path.as_ref().to_path_buf()],
            DType::F32,
            device,
        )?
    };

    println!("[Loader] Initialisation de la structure du modèle...");
    let model = ZildaMoeBackend::load(vb, config, embed_tokens_weight, lm_head_weight)?;

    println!("[Loader] Modèle chargé avec succès sur {:?}", device);
    Ok(model)
}