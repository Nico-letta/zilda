use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::time::{sleep, Duration};
use crate::memory::KVCacheManager;
use crate::backend::ZildaMoeBackend;
use tokenizers::Tokenizer;
use rand::distr::{weighted::WeightedIndex, Distribution};

pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub estimated_tokens: usize,
    pub tx_token: mpsc::Sender<String>,
}

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

    fn sample_next_token(logits: &[f32], temperature: f32) -> u32 {
        if temperature <= 0.0 {
            return logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
        }
    
        let logits: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    
        let max_logit = *logits.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let exp_logits: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

        let mut rng = rand::rng(); 
        let dist = WeightedIndex::new(&probs).unwrap(); 
        dist.sample(&mut rng) as u32
    }

    pub async fn run_engine_loop(
        cache_manager: Arc<Mutex<KVCacheManager>>, 
        backend: Arc<Mutex<ZildaMoeBackend>>,
        tokenizer: Arc<Tokenizer>,
        mut rx_queue: mpsc::Receiver<InferenceRequest>
    ) {
        println!("[Moteur Core] Boucle d'inférence en Continuous Batching active.");
        let mut active_batch: Vec<ActiveQuery> = Vec::new();

        loop {
            while let Ok(req) = rx_queue.try_recv() {
                let tokens = match tokenizer.encode(req.prompt.clone(), true) {
                    Ok(encoding) => encoding.get_ids().to_vec(),
                    Err(_) => vec![],
                };

                let mut manager = cache_manager.lock().await;
                match manager.allocate_slots(&req.request_id, req.estimated_tokens) {
                    Ok(()) => {
                        println!("[Scheduler] Requête {} acceptée et allouée dans le cache. Tokens du prompt : {:?}", req.request_id, tokens);
                        active_batch.push(ActiveQuery {
                            request_id: req.request_id,
                            prompt_tokens: tokens,
                            generated_tokens: Vec::new(),
                            tx_token: req.tx_token,
                        });
                    }
                    Err(err_msg) => {
                        println!("[Scheduler] Rejet de la requête {} : {}", req.request_id, err_msg);
                        let _ = req.tx_token.send(format!("[Erreur] {}", err_msg)).await;
                    }
                }
            }

            if active_batch.is_empty() {
                sleep(Duration::from_millis(50)).await;
                continue;
            }

            let mut finished_requests_indices = Vec::new();
            let mut manager = cache_manager.lock().await;
            let mut model = backend.lock().await;

            for (idx, query) in active_batch.iter_mut().enumerate() {
                
                // Détermination de la phase : Prefill vs Decoding
                let logits_result = if query.generated_tokens.is_empty() {
                    // --- PHASE DE PREFILL ---
                    let mut final_logits = None;
                    let mut err = None;

                    for &token_id in &query.prompt_tokens {
                        match model.forward_token(token_id, &query.request_id, &manager) {
                            Ok(logits) => final_logits = Some(logits),
                            Err(e) => {
                                err = Some(e);
                                break;
                            }
                        }
                    }

                    if let Some(e) = err {
                        Err(e)
                    } else {
                        final_logits.ok_or_else(|| candle_core::Error::Msg("Prompt vide".to_string()))
                    }
                } else {
                    let input_token_id = *query.generated_tokens.last().unwrap();
                    model.forward_token(input_token_id, &query.request_id, &manager)
                };

                match logits_result {
                    Ok(logits) => {
                        if let Ok(logits_vec) = logits.to_vec2::<f32>() {
                            let step_logits = &logits_vec[0];
                            
                            let next_token_id = Self::sample_next_token(step_logits, 0.7);

                            let prev_text = tokenizer.decode(&query.generated_tokens, true).unwrap_or_default();

                            query.generated_tokens.push(next_token_id);

                            let new_text = tokenizer.decode(&query.generated_tokens, true).unwrap_or_default();

                            let decoded_word = if new_text.len() > prev_text.len() {
                                new_text[prev_text.len()..].to_string()
                            } else {
                                String::new()
                            };

                            if !decoded_word.is_empty() {
                                if query.tx_token.send(decoded_word).await.is_err() {
                                    finished_requests_indices.push(idx);
                                    continue;
                                }
                            }

                            if query.generated_tokens.len() >= 30 || next_token_id == 0 {
                                let _ = query.tx_token.send("\n[Fin d'attention]".to_string()).await;
                                finished_requests_indices.push(idx);
                            }
                        }
                    }
                    Err(e) => {
                        let _ = query.tx_token.send(format!("\n[Erreur de calcul : {}]", e)).await;
                        finished_requests_indices.push(idx);
                    }
                }
            }

            for idx in finished_requests_indices.into_iter().rev() {
                let completed = active_batch.remove(idx);
                
                if let Some(blocks) = manager.table_de_pages.get(&completed.request_id) {
                    for &block_id in blocks {
                        model.vram_kv_store.remove(&block_id);
                    }
                }

                manager.free_slots(&completed.request_id);
                println!("[Memory Cache] Requête {} finalisée. Mémoire physique et logique libérée.", completed.request_id);
            }

            sleep(Duration::from_millis(10)).await;
        }
    }
}