use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::post,
    Json, Router,
};
use futures_util::stream::Stream;
use serde::Deserialize;
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::mpsc;
use crate::orchestrator::InferenceRequest;

#[derive(Deserialize)]
pub struct ApiPromptRequest {
    pub prompt: String,
    #[serde(default)]
    pub _stream: Option<bool>,
    // Ajout des paramètres pour piloter notre Sampler à chaud
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

pub struct ApiState {
    pub tx_queue: mpsc::Sender<InferenceRequest>,
}

pub async fn start_api_server(port: u16, tx_queue: mpsc::Sender<InferenceRequest>) {
    let shared_state = Arc::new(ApiState { tx_queue });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completion))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .unwrap();
    
    println!("[API Axum] Serveur prêt pour le streaming sur http://127.0.0.1:{}", port);
    axum::serve(listener, app).await.unwrap();
}

async fn handle_chat_completion(
    State(state): State<Arc<ApiState>>,
    Json(payload): Json<ApiPromptRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    if payload.prompt.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Le prompt est vide.".to_string()));
    }

    let (tx_token, mut rx_token) = mpsc::channel::<String>(50);
    let request_id = format!("REQ_HTTP_{}", rand::random::<u32>());

    // On passe directement les options à l'orchestrateur
    let internal_request = InferenceRequest {
        request_id,
        prompt: payload.prompt,
        estimated_tokens: 32,
        tx_token,
        temperature: payload.temperature,
        top_p: payload.top_p,
        repetition_penalty: payload.repetition_penalty,
    };

    if state.tx_queue.send(internal_request).await.is_err() {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, "Moteur d'inférence saturé.".to_string()));
    }

    let mystream = async_stream::stream! {
        while let Some(token) = rx_token.recv().await {
            yield Ok(Event::default().data(token));
        }
    };

    Ok(Sse::new(mystream).keep_alive(axum::response::sse::KeepAlive::new().interval(Duration::from_secs(1))))
}