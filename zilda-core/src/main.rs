#![allow(dead_code)]

mod memory;
mod orchestrator;
mod backend;
mod api;

use orchestrator::ZildaOrchestrator;
use backend::ZildaMoeBackend;
use std::sync::Arc;
use tokio::sync::Mutex;

// --- IMPORTS POUR LE BYTE-LEVEL ---
use tokenizers::pre_tokenizers::byte_level::ByteLevel as PreByteLevel;
use tokenizers::decoders::byte_level::ByteLevel as DecByteLevel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("====================================================");
    println!("     SERVEUR DE STREAMING ASYNCHRONE ZILDA          ");
    println!("====================================================");

    println!("[Système] Initialisation du tokenizer BPE MUNTU...");
    let vocab_path = "../data/vocab.json";
    let merges_path = "../data/merges.txt";

    let bpe = tokenizers::models::bpe::BPE::from_file(vocab_path, merges_path)
        .build()
        .map_err(|e| format!("Erreur lors de la lecture des fichiers BPE MUNTU : {}", e))?;

    let mut tokenizer = tokenizers::Tokenizer::new(bpe);

    // Correction ici : utilisation de "with_" et wrapping dans "Some()"
    tokenizer.with_pre_tokenizer(Some(PreByteLevel::default()));
    tokenizer.with_decoder(Some(DecByteLevel::default()));

    let tokenizer = Arc::new(tokenizer);
    println!("[Système] Tokenizer BPE chargé avec succès. Taille du vocabulaire : {}", tokenizer.get_vocab_size(true));

    let total_blocks = 40;
    let block_size = 16;

    let weights_path = "../data/muntu_pretrained.safetensors";
    let backend = Arc::new(Mutex::new(ZildaMoeBackend::new(weights_path)?));
    
    let (orchestrator, rx_queue) = ZildaOrchestrator::new(total_blocks, block_size, Arc::clone(&tokenizer));
    let orchestrator = Arc::new(orchestrator);

    let cache_manager_clone = Arc::clone(&orchestrator.cache_manager);
    let backend_clone = Arc::clone(&backend);
    let tokenizer_clone = Arc::clone(&tokenizer);
    
    tokio::spawn(async move {
        ZildaOrchestrator::run_engine_loop(cache_manager_clone, backend_clone, tokenizer_clone, rx_queue).await;
    });

    let tx_queue_clone = orchestrator.tx_queue.clone();
    println!("[Système] Lancement du service API... ");
    api::start_api_server(9999, tx_queue_clone).await;

    Ok(())
}