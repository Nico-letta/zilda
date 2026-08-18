This is an asynchronous inference engine written in **Rust**, optimized for executing **Mixture of Experts (MoE)** models. It relies on the `candle-core` framework for native execution without Python overhead.

### Technical Architecture

*   **Asynchronous Runtime:** Based on `tokio` for handling non-blocking requests and token streaming.
*   **Zero-Copy Loading:** Uses `MmapedSafetensors` to load weights directly into RAM, minimizing initialization latency.
*   **Streaming Pipeline:** Atomic inference per token (Embed -> Attention -> MoE -> Projection).
*   **Orchestrator:** Asynchronous queue for request management, isolated from the mathematical execution backend.
*   **Tokenizer:** Custom BPE (MUNTU) tokenizer optimized for byte-level processing.

### Software Stack

*   **Language:** Rust
*   **ML Framework:** `candle-core` (Hugging Face).
*   **API Server:** `tokio` (async runtime), `axum`.
*   **Serialization:** `safetensors`.

### Current State (Proof of Concept)

Zilda is currently a functional **PoC**. The basic infrastructure is validated, but the mathematical logic (tensor alignment) is undergoing debugging.

**Validated Features:**
*   Operational HTTP/API server.
*   Asynchronous pipeline and request queue.
*   Weight loading via Mmap.
*   Integrated BPE tokenizer.

### Roadmap

### Phase 1: Consolidation (Short-term)
*   **Shape resolution:** Correcting input/output dimensions on attention and MoE layers.
*   **Mathematical validation:** Verification of the complete `forward pass` (coherent logits).
*   **Unit tests:** Isolation of Expert `forward` calculations.

### Phase 2: Auto-configuration (Mid-term)
*   **Auto-Discovery Schema:** Removal of hard-coded parameters (`num_layers`, `num_heads`, etc.). The engine must inspect tensor metadata at load time to auto-configure.
*   **External configuration:** Implementation of a `zilda_config.toml` file to configure the engine without recompilation.
*   **KV Cache:** Memory management optimization for long sequences.

### Phase 3: Performance & Production (Long-term)
*   **GPU Acceleration:** CUDA/FlashAttention integration via `candle`.
*   **Batching:** Processing multiple simultaneous requests.
*   **Quantification:** Support for Q4/Q8 formats.
