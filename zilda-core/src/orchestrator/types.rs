use tokio::sync::mpsc;
use candle_core::Tensor;
use crate::memory::KVCacheManager;
use crate::backend::ZildaMoeBackend;

pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    pub estimated_tokens: usize,
    pub tx_token: mpsc::Sender<String>,

    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

pub struct ActiveQuery {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub tx_token: mpsc::Sender<String>,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl ActiveQuery {
    /// Construit une requête active avec les valeurs par défaut pour le sampling
    pub fn from_request(req: InferenceRequest, prompt_tokens: Vec<u32>) -> Self {
        Self {
            request_id: req.request_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            tx_token: req.tx_token,
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            repetition_penalty: req.repetition_penalty.unwrap_or(1.15),
        }
    }

    /// Exécute l'étape de forward : 
    /// - Passe tous les tokens du prompt (Prefill) si aucun token n'a encore été généré.
    /// - Passe uniquement le dernier token généré (Decode) sinon.
    pub fn step_forward(
        &mut self,
        model: &mut ZildaMoeBackend,
        manager: &KVCacheManager,
    ) -> Result<Tensor, candle_core::Error> {
        if self.generated_tokens.is_empty() {
            let mut final_logits = None;
            for &token_id in &self.prompt_tokens {
                final_logits = Some(model.forward_token(token_id, &self.request_id, manager)?);
            }
            final_logits.ok_or_else(|| candle_core::Error::Msg("Prompt vide".into()))
        } else {
            let input_token_id = *self.generated_tokens.last().unwrap();
            model.forward_token(input_token_id, &self.request_id, manager)
        }
    }
}