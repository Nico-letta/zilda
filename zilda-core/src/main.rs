mod memory;
mod orchestrator;
mod backend;
mod api;

use orchestrator::ZildaOrchestrator;
use backend::ZildaLinearBackend;
use std::sync::Arc;
use tokio::sync::Mutex; // Ajout du Mutex de Tokio pour la gestion de la concurrence

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("====================================================");
    println!("     SERVEUR DE STREAMING ASYNCHRONE ZILDA          ");
    println!("====================================================");

    let total_blocks = 40;
    let block_size = 16;

    // CORRECTION : On encapsule le backend dans un Mutex de Tokio avant de le mettre dans l'Arc
    let backend = Arc::new(Mutex::new(ZildaLinearBackend::new(512, 256)?));
    
    let (orchestrator, rx_queue) = ZildaOrchestrator::new(total_blocks, block_size);
    let orchestrator = Arc::new(orchestrator);

    let cache_manager_clone = Arc::clone(&orchestrator.cache_manager);
    let backend_clone = Arc::clone(&backend);
    
    tokio::spawn(async move {
        ZildaOrchestrator::run_engine_loop(cache_manager_clone, backend_clone, rx_queue).await;
    });

    let tx_queue_clone = orchestrator.tx_queue.clone();
    println!("[Système] Lancement du service API...");
    api::start_api_server(9999, tx_queue_clone).await;

    Ok(())
}