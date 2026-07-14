use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::time::{sleep, Duration};
use crate::memory::KVCacheManager;
use crate::backend::ZildaLinearBackend;
use candle_core::Tensor;

pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub estimated_tokens: usize,
    pub tx_token: mpsc::Sender<String>,
}

pub struct ZildaOrchestrator {
    pub cache_manager: Arc<Mutex<KVCacheManager>>,
    pub tx_queue: mpsc::Sender<InferenceRequest>,
}

impl ZildaOrchestrator {
    pub fn new(total_blocks: usize, block_size: usize) -> (Self, mpsc::Receiver<InferenceRequest>) {
        let manager = KVCacheManager::new(total_blocks, block_size);
        let (tx, rx) = mpsc::channel(100);

        let orchestrator = ZildaOrchestrator {
            cache_manager: Arc::new(Mutex::new(manager)),
            tx_queue: tx,
        };

        (orchestrator, rx)
    }

    pub async fn enqueue_request(&self, request: InferenceRequest) -> Result<(), String> {
        self.tx_queue.send(request).await
            .map_err(|_| "Impossible d'accéder à la file d'attente centrale.".to_string())
    }

    pub async fn run_engine_loop(
        cache_manager: Arc<Mutex<KVCacheManager>>, 
        backend: Arc<Mutex<ZildaLinearBackend>>, // Utilisation d'un Mutex ici car forward_attention est mutable
        mut rx_queue: mpsc::Receiver<InferenceRequest>
    ) {
        println!("[Moteur Core] Boucle d'exécution centralisée Candle démarrée.");
        let mut active_batch: Vec<InferenceRequest> = Vec::new();

        loop {
            while let Ok(req) = rx_queue.try_recv() {
                let mut manager = cache_manager.lock().await;
                if manager.allocate_slots(&req.request_id, req.estimated_tokens).is_ok() {
                    active_batch.push(req);
                } else {
                    let _ = req.tx_token.send("[ERREUR] VRAM saturée (OOM)".to_string()).await;
                }
            }

            if active_batch.is_empty() {
                sleep(Duration::from_millis(50)).await;
                continue;
            }

            let batch_size = active_batch.len();
            println!("\n[Continuous Batching] Calcul de l'attention pour {} requêtes actives...", batch_size);

            let mut finished_requests = Vec::new();
            let mut manager = cache_manager.lock().await;
            let mut model = backend.lock().await;

            for (idx, req) in active_batch.iter_mut().enumerate() {
                // Création d'un état d'entrée simulé [1, 512] conforme au backend
                if let Ok(input_state) = Tensor::randn(0f32, 1f32, (1, 512), model.device()) {
                    
                    // Exécution réelle de la passe d'attention
                    if let Ok(_logits) = model.forward_attention(&input_state, &req.request_id, &manager) {
                        let simulated_token_id = rand::random::<u32>() % 256;
                        
                        let simulated_word = match simulated_token_id % 5 {
                            0 => "Zilda_Core ",
                            1 => "Attention ",
                            2 => "KV_Cache ",
                            3 => "Optimisé ",
                            _ => "Rust_Power ",
                        };

                        let output_token = format!("{}(block_kv) ", simulated_word);

                        if req.tx_token.send(output_token).await.is_err() {
                            finished_requests.push(idx);
                            continue;
                        }

                        if rand::random::<f32>() > 0.88 {
                            let _ = req.tx_token.send("\n[Fin d'attention]".to_string()).await;
                            finished_requests.push(idx);
                        }
                    }
                }
            }

            for idx in finished_requests.into_iter().rev() {
                if idx < active_batch.len() {
                    let completed = active_batch.remove(idx);
                    manager.free_slots(&completed.request_id);
                    println!("[Memory Cache] Blocs physiques libérés pour : {}", completed.request_id);
                }
            }

            sleep(Duration::from_millis(150)).await;
        }
    }
}