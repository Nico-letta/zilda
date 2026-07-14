use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::time::{sleep, Duration};
use crate::memory::KVCacheManager;
use crate::backend::ZildaLinearBackend;
use candle_core::Tensor;
use tokenizers::Tokenizer;

pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub estimated_tokens: usize,
    pub tx_token: mpsc::Sender<String>,
}

// Pour suivre l'état de chaque requête en cours de traitement dans le batch actif
struct ActiveQuery {
    request_id: String,
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    tx_token: mpsc::Sender<String>,
}

pub struct ZildaOrchestrator {
    pub cache_manager: Arc<Mutex<KVCacheManager>>,
    pub tx_queue: mpsc::Sender<InferenceRequest>,
    pub tokenizer: Arc<Tokenizer>,
}

impl ZildaOrchestrator {
    pub fn new(total_blocks: usize, block_size: usize, tokenizer: Arc<Tokenizer>) -> (Self, mpsc::Receiver<InferenceRequest>) {
        let manager = KVCacheManager::new(total_blocks, block_size);
        let (tx, rx) = mpsc::channel(100);

        let orchestrator = ZildaOrchestrator {
            cache_manager: Arc::new(Mutex::new(manager)),
            tx_queue: tx,
            tokenizer,
        };

        (orchestrator, rx)
    }

    pub async fn run_engine_loop(
        cache_manager: Arc<Mutex<KVCacheManager>>, 
        backend: Arc<Mutex<ZildaLinearBackend>>, 
        tokenizer: Arc<Tokenizer>, // On passe aussi le tokenizer à la boucle d'inférence
        mut rx_queue: mpsc::Receiver<InferenceRequest>
    ) {
        println!("[Moteur Core] Boucle d'inférence en Continuous Batching active.");
        let mut active_batch: Vec<ActiveQuery> = Vec::new();

        loop {
            // 1. INJECTION DYNAMIQUE
            while let Ok(req) = rx_queue.try_recv() {
                // Utilisation du vrai tokenizer MUNTU pour encoder le prompt !
                match tokenizer.encode(req.prompt.clone(), true) {
                    Ok(encoding) => {
                        let token_ids = encoding.get_ids().to_vec();
                        let num_tokens = token_ids.len();
                        
                        // Allocation dynamique basée sur la vraie taille du prompt + une marge pour la génération
                        let reserve_size = num_tokens + req.estimated_tokens;
                        let mut manager = cache_manager.lock().await;

                        if manager.allocate_slots(&req.request_id, reserve_size).is_ok() {
                            println!(
                                "[Scheduler] Nouvelle requête intégrée (Prompt: {} tokens, Réservé: {}): {}", 
                                num_tokens, reserve_size, req.request_id
                            );
                            active_batch.push(ActiveQuery {
                                request_id: req.request_id,
                                prompt_tokens: token_ids,
                                generated_tokens: Vec::new(),
                                tx_token: req.tx_token,
                            });
                        } else {
                            let _ = req.tx_token.send("[ERREUR] VRAM saturée (OOM) - Impossible d'allouer le cache".to_string()).await;
                        }
                    }
                    Err(e) => {
                        let _ = req.tx_token.send(format!("[ERREUR] Échec de la tokenisation : {}", e)).await;
                    }
                }
            }

            if active_batch.is_empty() {
                sleep(Duration::from_millis(50)).await;
                continue;
            }

            // 2. ÉTAPE D'ATTENTION (1 token par requête active)
            let mut finished_requests_indices = Vec::new();
            let mut manager = cache_manager.lock().await;
            let mut model = backend.lock().await;

            println!("[Batching] Itération d'attention sur {} requêtes...", active_batch.len());

            for (idx, query) in active_batch.iter_mut().enumerate() {
                // Simulation de l'état caché de l'input [1, 512]
                if let Ok(input_state) = Tensor::randn(0f32, 1f32, (1, 512), model.device()) {
                    
                    if let Ok(_logits) = model.forward_attention(&input_state, &query.request_id, &manager) {
                        // Pour l'instant, on simule la sélection d'un ID de token valide (dans la limite du vocabulaire de 8000)
                        // On évite le token 0 (souvent [PAD] ou [UNK]) pour avoir des caractères visibles
                        let next_token_id = (rand::random::<u32>() % 7900) + 100; 

                        // On décode le token ID en texte brut via le tokenizer de MUNTU !
                        let decoded_word = match tokenizer.decode(&[next_token_id], true) {
                            Ok(text) => text,
                            Err(_) => " ".to_string(),
                        };

                        // Envoi du vrai token décodé au client HTTP
                        if query.tx_token.send(decoded_word).await.is_err() {
                            println!("[Scheduler] Client déconnecté : {}", query.request_id);
                            finished_requests_indices.push(idx);
                            continue;
                        }

                        query.generated_tokens.push(next_token_id);

                        // Arrêt si la génération dépasse la taille allouée ou par probabilité (simulant un EOS)
                        if query.generated_tokens.len() >= 30 || rand::random::<f32>() > 0.90 {
                            let _ = query.tx_token.send("\n[Fin d'attention]".to_string()).await;
                            finished_requests_indices.push(idx);
                        }
                    } else {
                        finished_requests_indices.push(idx);
                    }
                }
            }

            for idx in finished_requests_indices.into_iter().rev() {
                let completed = active_batch.remove(idx);
                manager.free_slots(&completed.request_id);
                println!("[Memory Cache] Requête {} finalisée. Blocs de mémoire libérés.", completed.request_id);
            }

            sleep(Duration::from_millis(100)).await;
        }
    }
}