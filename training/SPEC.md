# Rebuilding `adapter_final.safetensors` — the Synapse

**Status:** the original build script was never missing. It is `src/bin/train_adapter.rs`
in `SplatRagBench-master`. It is **Rust on `candle`**, not Python — which is why every
Python-shaped search for a trainer came back empty.

Verbatim copies of every file referenced below are in `original_source/`.

---

## 0. Identity check

The shipped artifact contains exactly two tensors:

```
adapter.linear.weight   F32  [896, 128]
adapter.linear.bias     F32  [896]
```

`896*128*4 + 896*4 + header = 462,512 bytes` — the exact file size.

Those names are a fingerprint. `train_adapter.rs` builds the adapter under
`vb.pp("adapter")` and `adapter.rs` builds its `Linear` under `vb.pp("linear")`,
so `VarMap::save` emits `adapter.linear.{weight,bias}`. Nothing else in the repo
produces that naming. This is the right script.

Recovered copy md5: `b8118021c7c27948565c4f322b5f6e04`
(identical in `SplatRagBench-master/` and `ontological-inversion/`).

---

## 1. What the model is

One linear layer. That is the entire Synapse.

```rust
pub const TOPOLOGICAL_CENTROID_DIM: usize   = 64;
pub const TOPOLOGICAL_COVARIANCE_DIM: usize = 64;
pub const TOTAL_INPUT_DIM: usize            = 128;

pub struct SplatAdapter { proj: Linear }   // linear(128, hidden_size), bias=true
```

No hidden layer, no activation, no normalization, no dropout.
`hidden_size` is read from the loaded LLM at runtime; for Qwen2.5-0.5B it is 896.

---

## 2. The input vector (128-d) — this is the part that gets rebuilt wrong

The 128 is **not** a 128-d embedding. It is two 64-d halves concatenated.

### 2a. Embedding backend

`EmbeddingModel::new(_model_repo, use_gpu)` — **note the underscore.** The model-repo
argument is deliberately ignored. `config.rs` defaults `nomic_model_repo` to
`sentence-transformers/all-MiniLM-L6-v2`, and that value is **dead code**.

The real model is hardcoded in the Python sidecar `src/nomic_daemon.py`:

```python
model_id = 'nomic-ai/nomic-embed-text-v1.5'
```

Rust spawns `./venv/bin/python3 src/nomic_daemon.py` and talks to it over
stdin/stdout as JSON lines. The daemon returns, per text: `pooled`, `token_embeddings`,
`tokens`. Pooling is attention-masked mean pooling.

nomic-embed-text-v1.5 *is* a Matryoshka model, so the 64-d truncation below is a
legitimate Matryoshka cut, not arbitrary slicing. **If you swap in a non-Matryoshka
encoder, truncation stops being meaningful and the adapter will not match.**

### 2b. First half — μ (centroid), 64-d

```
pooled = daemon.embed_document(text)     # 768-d
pooled.truncate(64)                      # Matryoshka cut
L2-normalize(pooled)                     # ‖μ‖ = 1
```

### 2c. Token matrix for PCA

```
token_embeddings[i].truncate(64)         # 64-d each
# NOT normalized — pooled is normalized, tokens deliberately are not
```

### 2d. PCA → the Gaussian

Let `X` be the `n × 64` token matrix, `n` = token count.

```
centered = X - μ                # centered on the POOLED mean, not the token mean
cov      = centeredᵀ · centered / (n - 1)     # 64 × 64
eigen    = SymmetricEigen(cov)
sort eigenvalues descending → λ1 ≥ λ2 ≥ λ3 …, each clamped .max(1e-6)

u_vec      = eigenvector(λ1)              # unit, 64-d
anisotropy = λ1 / (λ2 + 1e-9)
sigma_iso  = (λ1 · λ2 · λ3)^(1/3) then √  =  (λ1·λ2·λ3)^(1/6)
```

Two non-obvious things:

- **Centering uses the pooled mean, not the column mean of the tokens.** That is not
  textbook PCA. The code comments acknowledge it. Reproduce it as written.
- `cov` is 64×64 but rank ≤ n−1. For short texts (n < 64) it is rank-deficient and
  λ3 collapses to the `1e-6` clamp, which drags `sigma_iso` down hard. This is
  load-bearing behavior, not a bug to fix.

**Fallback when `n ≤ 2`:**
```
principal_axis = normalize(μ)   (or all-ones normalized if ‖μ‖ = 0)
sigma_iso      = 0.5
anisotropy     = 1.0
```

### 2e. Second half — variance, 64-d

```
magnitude    = sigma_iso * anisotropy
variance_vec = u_vec * magnitude
input_geo    = concat(μ, variance_vec)     # [1, 128]
```

### ⚠ Scale asymmetry — read this before debugging a bad rebuild

`‖μ‖ = 1` exactly (L2-normalized). `‖variance_vec‖ = sigma_iso * anisotropy`, and
`anisotropy = λ1/λ2` is unbounded — the `SemanticGaussian` doc comment itself says
`>100 = extreme needle`.

So the two halves of the input can differ in scale by two orders of magnitude, with
no normalization between them. The trained adapter absorbed that asymmetry into its
weights. If a rebuild normalizes the halves "to be clean," it produces a different
adapter that will not reproduce the flip.

---

## 3. The target (896-d)

```rust
let encoding = tokenizer.encode(text, true)?;        // add_special_tokens = true
let raw_embed = qwen_model.embed(&token_tensor)?;    // [1, T, 896]
let target = raw_embed.mean(1)?.detach();            // [1, 896]
```

**The target is the token-embedding layer, mean-pooled over the sequence.** Not a
mid-stack hidden state. Not layer 4. The LLM is frozen — it is loaded outside the
`VarMap`, and `.detach()` cuts the graph.

Base model: `HF_MODEL_REPO`, defaulting to `Qwen/Qwen2.5-0.5B` (hidden 896). The script
warns but proceeds if hidden_size ≠ 896.

> **Known mismatch, carried forward on purpose:** the downstream `ontological-inversion`
> repo injects this adapter's output into the **layer-4 residual** at α ≈ 0.15–0.30.
> The adapter was never fit to layer-4 activations. Same dimensionality, different
> distribution. The layer-4 choice is an empirical downstream finding; nothing in this
> training script justifies it. Do not present the two as one coherent derivation.

---

## 4. Training loop

```rust
let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;

for epoch in 0..20 {
    for (input_geo, target) in &training_data {     // ← one pair at a time
        let predicted = adapter.forward(input_geo)?;
        let diff = (predicted - target)?;
        let loss = diff.sqr()?.mean_all()?;         // MSE
        opt.backward_step(&loss)?;
    }
    if avg_loss < 0.001 { break; }                  // early stop
}
varmap.save("adapter_final.safetensors")?;
```

| | |
|---|---|
| Loss | MSE, `diff.sqr().mean_all()` |
| Optimizer | `AdamW::new_lr(1e-3)` — candle defaults for β, ε, weight decay |
| Epochs | 20, early stop at avg loss < 0.001 |
| **Batch size** | **1.** The loop steps per pair. The `batch_size = 32` in Phase 1 is the *shaping* batch only and never reaches the optimizer. |
| Data order | `all_entries.sort_by_key(|k| k.0)` — sorted by memory id, **no shuffling**, same order every epoch |
| Init | candle `linear()` default |

Batch-1 AdamW at lr 1e-3 with fixed data order is noisy SGD. The result is
order-dependent. That is one reason from-scratch reconstructions kept missing it.

---

## 5. Training data — the actual blocker

```rust
let base_path = if Path::new("mindstream_geometry.bin").exists() { "mindstream" }
                else { "bench_memory" };
let mem_sys = MemorySystem::new(base_path, &manifest_path)?;
// manifest: id (u64) -> text (String)
```

Every manifest entry becomes one training pair. Shaped in batches of 32; a failed
batch is skipped with a warning, and a text that tokenizes empty is skipped.

**The corpus that produced the shipped file is not in the repo.** What ships is a
smoke-test stub:

```
manifest.bin              71 bytes   →  "The sky is neon green."
mindstream_semantics.bin 576 bytes
mindstream_geometry.bin   48 bytes
```

One sentence. `adapter_final.safetensors` was not trained on that.

**Consequence: byte-identical reproduction is impossible from this repo.** A
functionally equivalent adapter is entirely reproducible — the pipeline above is
complete — but it needs a corpus, and the original one is missing. Do not let a
rebuild claim it "reproduced the adapter" when it trained on different data.

---

## 6. Rebuild checklist

1. Stand up `nomic_daemon.py` with `nomic-ai/nomic-embed-text-v1.5`, `trust_remote_code=True`.
2. Implement the Shaper exactly as §2d — pooled-mean centering, the `1e-6` clamps,
   `(λ1λ2λ3)^(1/6)`, the `n ≤ 2` fallback.
3. Build the 128-d input as `concat(normalize(truncate(pooled,64)), u_vec * σ·a)`.
   Do not normalize across the concat.
4. Freeze Qwen2.5-0.5B; target = `embed(tokens).mean(1).detach()`.
5. `Linear(128→896)` with bias, MSE, AdamW lr 1e-3, batch 1, 20 epochs, id-sorted order.
6. Save via `VarMap::save` so tensor names come out `adapter.linear.{weight,bias}`.
7. Verify: shapes `[896,128]` / `[896]`, file 462,512 bytes.
8. Then, and only then, test the flip downstream at α 0.15–0.30, injecting at layer 4 —
   remembering §3's caveat that this is not the space it was trained on.

---

## 7. Source paths

Live repo (external drive, re-enumerates letters — mount by label, not `/dev/sdX`):

```
/run/media/ruffianl/ghost_team/02_projects/legacy-splatrag/splatrag/
  SplatRagBench-master/SplatRagBench-master/
    src/bin/train_adapter.rs      ← the trainer
    src/adapter.rs                ← SplatAdapter
    src/ingest/shaper.rs          ← Gaussian construction
    src/embeddings.rs             ← 64-d truncation, daemon IPC
    src/physics/gaussian.rs       ← SemanticGaussian, random_orthogonal
    src/nomic_daemon.py           ← hardcoded nomic-embed-text-v1.5
    src/config.rs                 ← contains the dead MiniLM default
    adapter_final.safetensors
```

Downstream consumer (does **not** contain a trainer — verified by grep for
`torch.optim|AdamW|.backward()|def train`, zero hits):

```
/run/media/ruffianl/ghost_team/02_projects/ghost_team/active_projects/ontological-inversion/
```

Note that repo's `PROVENANCE.md` describes the input as a "128-d nomic Matryoshka
concept code." That is wrong in two ways: the nomic cut is **64**-d, and the other 64
dims are scaled PCA variance, not embedding. Trust this spec over that line.
