pub mod types;
pub mod sampler;
pub mod decoder;

use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::time::{sleep, Duration};
use crate::memory::KVCacheManager;
use crate::backend::ZildaMoeBackend;
use tokenizers::Tokenizer;

pub use types::{InferenceRequest, ActiveQuery};
use sampler::Sampler;
use decoder::StreamDecoder;

pub struct ZildaOrchestrator {
    pub cache_manager: Arc<Mutex<KVCacheManager>>,
    pub tx_queue: mpsc::Sender<InferenceRequest>,
    pub tokenizer: Arc<Tokenizer>,
}

impl ZildaOrchestrator {
    pub fn new(total_blocks: usize, block_size: usize, tokenizer: Arc<Tokenizer>) -> (Self, mpsc::Receiver<InferenceRequest>) {
        let manager = KVCacheManager::new(total_blocks, block_size);
        let (tx, rx) = mpsc::channel(100);

        (
            ZildaOrchestrator {
                cache_manager: Arc::new(Mutex::new(manager)),
                tx_queue: tx,
                tokenizer,
            },
            rx,
        )
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
                let tokens = tokenizer
                    .encode(req.prompt.clone(), true)
                    .map(|e| e.get_ids().to_vec())
                    .unwrap_or_default();

                let mut manager = cache_manager.lock().await;
                if let Ok(()) = manager.allocate_slots(&req.request_id, req.estimated_tokens) {
                    active_batch.push(ActiveQuery::from_request(req, tokens));
                } else {
                    let _ = req.tx_token.send("[Erreur] Cache mémoire saturé".to_string()).await;
                }
            }

            if active_batch.is_empty() {
                sleep(Duration::from_millis(50)).await;
                continue;
            }

            let mut tokens_to_send: Vec<(mpsc::Sender<String>, String, usize)> = Vec::new();
            let mut finished_indices: Vec<usize> = Vec::new();

            {
                let manager = cache_manager.lock().await;
                let mut model = backend.lock().await;

                for (idx, query) in active_batch.iter_mut().enumerate() {
                    let logits_result = query.step_forward(&mut model, &manager);

                    match logits_result {
                        Ok(logits) => {
                            if let Ok(logits_vec) = logits.to_vec2::<f32>() {
                                let temp = if query.temperature <= 0.0 { 0.7 } else { query.temperature };
                                let top_p = query.top_p;
                                let rep_pen = query.repetition_penalty;

                                let next_token = Sampler::sample(
                                    &logits_vec[0],
                                    temp,
                                    top_p,
                                    rep_pen,
                                    &query.generated_tokens,
                                );

                                query.generated_tokens.push(next_token);

                                let delta_text = StreamDecoder::decode_next(&tokenizer, &query.generated_tokens);
                                
                                let text_payload = if delta_text.is_empty() {
                                    format!("[Token_{}]", next_token)
                                } else {
                                    delta_text
                                };

                                tokens_to_send.push((query.tx_token.clone(), text_payload, idx));

                                let num_generated = query.generated_tokens.len().saturating_sub(query.prompt_tokens.len());

                                if next_token == 0 || next_token == 2 || num_generated >= 50 {
                                    tokens_to_send.push((query.tx_token.clone(), "\n[Fin d'attention]".to_string(), idx));
                                    finished_indices.push(idx);
                                }
                            }
                        }
                        Err(e) => {
                            tokens_to_send.push((query.tx_token.clone(), format!("\n[Erreur calcul : {}]", e), idx));
                            finished_indices.push(idx);
                        }
                    }
                }
            } 
            for (tx, text, idx) in tokens_to_send {
                if tx.send(text).await.is_err() && !finished_indices.contains(&idx) {
                    finished_indices.push(idx);
                }
            }

            if !finished_indices.is_empty() {
                finished_indices.sort_unstable();
                finished_indices.dedup();

                let mut manager = cache_manager.lock().await;
                let mut model = backend.lock().await;

                for idx in finished_indices.into_iter().rev() {
                    if idx < active_batch.len() {
                        let completed = active_batch.remove(idx);
                        model.free_request_kv(&completed.request_id, &manager);
                        manager.free_slots(&completed.request_id);
                    }
                }
            }

            sleep(Duration::from_millis(10)).await;
        }
    }
}