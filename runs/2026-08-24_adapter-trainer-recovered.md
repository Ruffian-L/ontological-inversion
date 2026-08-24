# Run card — the Synapse trainer was never missing

**Ran by:** Claude (Opus 5) with Jason   ·   **Date:** 2026-08-24   ·   **Tree:** ontological-inversion @ working
**Verdict:** PASS (recovery + spec)   ·   No new steering claim.

## 1. What we asked

Where did `adapter_final.safetensors` come from, and can somebody rebuild it
from what is in this repo?

## 2. What we ran

No model runs. A source recovery against the original tree on the `ghost_team`
drive, plus a hash check of the artifact this repo ships.

## 3. How we ran it (copy-paste reproducible)

```
md5sum adapter_final.safetensors
# b8118021c7c27948565c4f322b5f6e04

# same file, recovered independently from the original repo:
md5sum SplatRagBench-master/SplatRagBench-master/adapter_final.safetensors
# b8118021c7c27948565c4f322b5f6e04
```

## 4. What we expected

That the trainer was lost, and a rebuild would have to be inferred from the
tensor shapes.

## 5. What actually happened

**The build script was never missing.** It is `src/bin/train_adapter.rs` in
`SplatRagBench-master` — **Rust on `candle`**. Every prior search for it looked
for Python (`torch.optim`, `AdamW`, `.backward()`, `def train`), which is why it
kept coming back empty. Its tensor naming is a fingerprint: `vb.pp("adapter")`
then `vb.pp("linear")` is the only thing in the tree that emits
`adapter.linear.{weight,bias}`, which is exactly what the shipped file contains.

Four things a rebuild would have gotten silently wrong, all now written down in
[`training/SPEC.md`](../training/SPEC.md):

1. **The 128-d input is not an embedding.** It is `concat(mu₆₄, shape₆₄)` — an
   L2-normalised Matryoshka cut of the pooled nomic vector, and the PCA
   principal axis of the token cloud scaled by `sigma_iso × anisotropy`. The
   halves are deliberately unnormalised against each other and can differ in
   length by 100×. The weights absorbed that.
2. **`config.rs`'s encoder name is dead code.** `EmbeddingModel::new(_model_repo, …)`
   ignores the argument, so the `all-MiniLM-L6-v2` default never applies.
   `nomic_daemon.py` hardcodes `nomic-ai/nomic-embed-text-v1.5` — which is what
   makes the 64-d cut a real Matryoshka cut rather than arbitrary slicing.
3. **The PCA is non-textbook on purpose** — centered on the pooled mean, `1e-6`
   eigenvalue clamps, `sigma_iso = (λ₁λ₂λ₃)^(1/6)`, and an `n ≤ 2` fallback.
   Load-bearing, not a bug.
4. **The target is `embed(tokens).mean(1).detach()`** — the token-embedding
   layer, mean-pooled. Not layer 4.

Point 4 is a mismatch this repo carries and should carry openly: **we inject at
the layer-4 residual, and the adapter was never fit to layer-4 activations.**
Same dimensionality, different distribution. Layer 4 is an empirical finding —
the sweeps found it, it reproduces, the controls hold — but nothing in the
training script derives it, and the two should not be told as one story.

`PROVENANCE.md` previously described the input as a "128-d nomic (Matryoshka)
concept code." That was wrong in two ways — the nomic cut is 64-d, and the other
64 dims are scaled PCA variance, not embedding. Corrected in place, with the
correction dated rather than the old line deleted.

**What still cannot be reproduced byte-for-byte:** the training corpus is gone.
What ships in `SplatRagBench-master` is a 71-byte smoke stub containing one
sentence, *"The sky is neon green."* A functionally equivalent adapter is fully
reproducible from the spec; an identical one is not. A rebuild trained on other
data should not report that it "reproduced the adapter."

## 6. The scoreboard — the climb

| Attempt | What we tried | Result |
|---|---|---|
| 2026-06-24 | reconstruct the antipode from scratch, 11 ways | none reproduced the flip |
| 2026-06-24 | recover the trained adapter and inject its direction | flip reproduced, α≈0.2 |
| **2026-08-24** | **find how that adapter was built** | **found: `train_adapter.rs`, Rust on candle; spec written; 4 rebuild traps documented** |

> Read this as the climb: the artifact worked before anyone knew how it was
> made, and now the recipe is on the shelf next to it.

## 7. What this does not claim

Nothing about the steering result changes. No run was re-scored, no band moved,
no metric improved. This is provenance and reproducibility only.
