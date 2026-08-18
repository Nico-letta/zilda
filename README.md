This is an asynchronous inference engine written in **Rust**, optimized for executing **Mixture of Experts (MoE)** models. It relies on the `candle-core` framework for native execution without Python overhead. The current target model is **MUNTU LM** (`muntu_pretrained.safetensors`).



### Technical Architecture



*   **Asynchronous Runtime:** Based on `tokio` for handling non-blocking requests and token streaming.

*   **Zero-Copy Loading:** Uses `MmapedSafetensors` to load weights directly into RAM, minimizing initialization latency.

*   **Streaming Pipeline:** Atomic inference per token (Embed -> Attention -> MoE -> Projection).

*   **Orchestrator:** Asynchronous queue for request management, isolated from the mathematical execution backend. *(Implemented, pending re-integration.)*

*   **KV Cache:** Per-request K/V storage in the attention layers; block-based memory manager (`KVCacheManager`) for multi-request allocation.

*   **Tokenizer:** External MUNTU tokenizer loaded via the Hugging Face `tokenizers` crate (`tokenizer.json`).



### Software Stack



*   **Language:** Rust

*   **ML Framework:** `candle-core` (Hugging Face).

*   **API Server:** `tokio` (async runtime), `axum`.

*   **Serialization:** `safetensors`.



### Current State (Proof of Concept)



Zilda is currently a functional **PoC**. The active entry point is a **CLI inference loop** (`main.rs`): load weights, run a token-by-token forward pass, and stream output to stdout. The HTTP server and orchestrator modules exist but are not wired into the binary yet.



**Validated Features:**

*   Weight loading via Mmap.

*   Token-by-token forward pass (Transformer blocks: LayerNorm, Attention, MoE).

*   Position embeddings and basic K/V cache in attention.

*   CLI text generation with temperature and repetition penalty.

*   External MUNTU tokenizer integration.



**Pending Re-integration:**

*   HTTP/API server (`axum`, SSE streaming on `/v1/chat/completions`).

*   Asynchronous pipeline, request queue, and continuous batching (orchestrator).



**In Progress:**

*   Mathematical validation of the complete `forward pass` (coherent logits).

*   MoE routing refinement (current implementation uses soft routing across all experts).



### Roadmap

Status key: `[x]` done · `[ ]` pending · *(partial)* started but incomplete · *(in progress)* actively being worked on.

### Completed

*   [x] **Shape resolution:** Input/output dimensions on attention and MoE layers.
*   [x] **Forward pass structure:** Transformer blocks (LayerNorm, Attention, MoE) with position embeddings.
*   [x] **Basic K/V cache:** Per-request key/value storage in attention layers.

### Phase 1: Consolidation (Short-term)

*   [ ] **Mathematical validation:** Verification of the complete `forward pass` (coherent logits). *(in progress)*
*   [ ] **MoE routing:** Top-k expert selection instead of soft routing across all experts.
*   [ ] **Unit tests:** Isolation of Expert `forward` calculations.
*   [ ] **Server re-integration:** Reconnect the API and orchestrator to the active backend.

### Phase 2: Auto-configuration (Mid-term)

*   [ ] **Auto-Discovery Schema:** Removal of hard-coded parameters (`num_layers`, `num_heads`, etc.). The engine must inspect tensor metadata at load time to auto-configure.
*   [ ] **External configuration:** Implementation of a `zilda_config.toml` file to configure the engine without recompilation.
*   [ ] **KV Cache:** Finalize block-based memory management via `KVCacheManager` for long sequences and concurrent requests. *(partial: allocator implemented, not wired to CLI)*

### Phase 3: Performance & Production (Long-term)

*   [ ] **GPU Acceleration:** CUDA/FlashAttention integration via `candle`.
*   [ ] **Batching:** Activate continuous batching. *(partial: orchestrator code present)*
*   [ ] **Quantification:** Support for Q4/Q8 formats.
*   [ ] **Language bindings:** Complete `zilda-python` (PyO3) and `zilda-js` integrations. *(partial: PyO3 stub, JS skeleton)*


