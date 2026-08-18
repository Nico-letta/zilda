use candle_core::Device;
use tokenizers::Tokenizer;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod backend;
pub mod memory;
pub mod orchestrator;

use crate::backend::{Config, loader::ModelLoader};
use crate::orchestrator::ZildaOrchestrator;

/// Résout dynamiquement le chemin des poids Safetensors.
fn resolve_weights_path() -> PathBuf {
    if let Ok(path) = env::var("MODEL_PATH") {
        return PathBuf::from(path);
    }
    let root_path = PathBuf::from("data/muntu_pretrained.safetensors");
    if root_path.exists() {
        return root_path;
    }
    PathBuf::from("../data/muntu_pretrained.safetensors")
}

/// Résout dynamiquement le chemin du tokenizer JSON.
fn resolve_tokenizer_path() -> PathBuf {
    if let Ok(path) = env::var("TOKENIZER_PATH") {
        return PathBuf::from(path);
    }
    let root_path = PathBuf::from("data/tokenizer.json");
    if root_path.exists() {
        return root_path;
    }
    PathBuf::from("../data/tokenizer.json")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let config = Config::default();

    // 1. Chargement du modèle backend
    let weights_path = resolve_weights_path();
    println!("[Zilda] Recherche du modèle sur : {:?}", weights_path);

    let _backend = Arc::new(Mutex::new(
        ModelLoader::load_from_safetensors(&weights_path, &config, &device)?
    ));

    // 2. Chargement du Tokenizer
    let tokenizer_path = resolve_tokenizer_path();
    println!("[Zilda] Recherche du tokenizer sur : {:?}", tokenizer_path);
    
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Impossible de charger le tokenizer ({:?}): {}", tokenizer_path, e))?;
    let tokenizer = Arc::new(tokenizer);

    // 3. Configuration du KV Cache Manager
    let total_blocks = 128; // Nombre de blocs mémoire alloués
    let block_size = 16;    // Nombre de tokens par bloc

    // 4. Instanciation de l'Orchestrateur
    let (_orchestrator, _rx) = ZildaOrchestrator::new(total_blocks, block_size, tokenizer);

    println!("[Zilda] Orchestrateur initialisé avec succès sur : {:?}", device);

    Ok(())
}