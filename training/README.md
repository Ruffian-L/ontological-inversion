# How the Synapse was built

`adapter_final.safetensors` in the repo root is the trained map that turns text
into a steering direction. For a long time its build script was thought to be
missing — every search for it looked for Python. It is **Rust on `candle`**:
`src/bin/train_adapter.rs` in `SplatRagBench-master`. It is here, verbatim,
along with every file it depends on.

```
SPEC.md                        the complete rebuild spec, verified against source
original_source/
  train_adapter.rs             the trainer
  adapter.rs                   SplatAdapter — one Linear(128 → hidden), bias, nothing else
  shaper.rs                    the Gaussian: PCA, sigma_iso, anisotropy
  embeddings.rs                the 64-d Matryoshka cut and the daemon IPC
  gaussian.rs                  SemanticGaussian
  nomic_daemon.py              the Python sidecar; hardcodes nomic-embed-text-v1.5
  config.rs                    contains a dead MiniLM default — see below
```

## The model is one linear layer

That is the entire Synapse. No hidden layer, no activation, no normalization,
no dropout.

```rust
pub const TOPOLOGICAL_CENTROID_DIM: usize   = 64;
pub const TOPOLOGICAL_COVARIANCE_DIM: usize = 64;
pub const TOTAL_INPUT_DIM: usize            = 128;

pub struct SplatAdapter { proj: Linear }   // linear(128, hidden_size), bias = true
```

Shipped artifact: `adapter.linear.weight [896, 128]`, `adapter.linear.bias [896]`,
md5 `b8118021c7c27948565c4f322b5f6e04`. Those tensor names are a fingerprint —
`vb.pp("adapter")` then `vb.pp("linear")` is what produces them, and nothing else
in the source tree does.

## Four things a rebuild gets wrong

**1. The 128 is two halves, not an embedding.** First 64: the pooled nomic vector
Matryoshka-cut to 64 and L2-normalised, so `‖mu‖ = 1` exactly. Second 64: the PCA
principal axis of the concept's token cloud, scaled by `sigma_iso × anisotropy`.
`anisotropy = λ₁/λ₂` is unbounded — the code's own comment calls `>100` an
extreme needle — so the two halves can differ in length by two orders of
magnitude, with no normalization between them. **The trained weights absorbed
that asymmetry.** Normalising the halves "to be clean" yields a different adapter
that will not reproduce the flip.

**2. The encoder in `config.rs` is dead code.** `EmbeddingModel::new(_model_repo, …)`
— note the underscore. The argument is deliberately ignored, so the
`all-MiniLM-L6-v2` default never applies. The real encoder is hardcoded in
`nomic_daemon.py` as `nomic-ai/nomic-embed-text-v1.5`. That matters beyond
bookkeeping: nomic-v1.5 is a Matryoshka model, which is what makes the 64-d
truncation a legitimate cut rather than arbitrary slicing. **Swap in a
non-Matryoshka encoder and truncation stops meaning anything.**

**3. The PCA is not textbook, on purpose.** Centering uses the *pooled* mean, not
the token column mean. Eigenvalues are clamped at `1e-6`, and `sigma_iso =
(λ₁λ₂λ₃)^(1/6)`. For short texts the covariance is rank-deficient and λ₃ hits the
clamp, dragging `sigma_iso` down hard. That is load-bearing behaviour, not a bug
to fix. There is an `n ≤ 2` fallback. Reproduce all of it as written.

**4. The target is the token-embedding layer, mean-pooled.**

```rust
let raw_embed = qwen_model.embed(&token_tensor)?;   // [1, T, 896]
let target    = raw_embed.mean(1)?.detach();        // [1, 896]
```

Not a mid-stack hidden state, not layer 4. The LLM is frozen — loaded outside the
`VarMap`, and `.detach()` cuts the graph.

That last point is worth stating plainly rather than smoothing over: **this repo
injects at the layer-4 residual, and the adapter was never fit to layer-4
activations.** Same dimensionality, different distribution. Layer 4 is an
empirical downstream finding — the sweeps found it and it reproduces — but
nothing in this training script derives it. The two should not be presented as
one coherent derivation.

## Training loop

```rust
let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;

for epoch in 0..20 {
    for (input_geo, target) in &training_data {     // one pair at a time
        let predicted = adapter.forward(input_geo)?;
        let loss = (predicted - target)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
    }
    if avg_loss < 0.001 { break; }
}
varmap.save("adapter_final.safetensors")?;
```

MSE, AdamW at `1e-3` with candle defaults, **batch size 1** — the `batch_size = 32`
elsewhere is shaping only and never reaches the optimizer — 20 epochs, early stop
at `0.001`, data sorted by memory id with **no shuffling**, same order every epoch.
Batch-1 AdamW at that learning rate with a fixed order is noisy SGD, so the result
is order-dependent. That is one reason from-scratch reconstructions kept missing.

## What cannot be reproduced

The corpus that produced the shipped file is gone. What ships in
`SplatRagBench-master` is a smoke-test stub: `manifest.bin` is 71 bytes and
contains one sentence, *"The sky is neon green."* The adapter was obviously not
trained on that.

So: a **functionally equivalent** adapter is fully reproducible from `SPEC.md` —
the pipeline is complete and nothing is guessed. A **byte-identical** one is not.
A rebuild trained on a different corpus should not report that it "reproduced the
adapter."
