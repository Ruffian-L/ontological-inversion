# Provenance

This effect was not invented for this repo — it was **rediscovered and reproduced**. The thread:

- **Origin (Nov 2025, Grok/Gemini sessions, "SplatRAG/Niodoo"):** synthetic concepts
  (`worbglob`, `Glub-Tub` = "magma-eating hamster") were injected into a small model and steered
  with a signed "gain." Negative gain was observed to flip a concept to a *structured opposite*
  rather than noise — named **Ontological Inversion / The Anti-Splat**. The original logged
  outputs include *"a type of appliance used to heat water in a fireplace,"* *"a 'wooden stick'
  or 'firebrick',"* and *"not an animal… a shelter."*

- **Mechanism (recovered from source):** `niodoo/src/physics/steering.rs` (inverse-distance
  logit-bias force field), `antigravity.rs` (Coulomb charge model), `ontological_inversion.rs`
  (`householder_reflect` / `reflection_commutator`), `genesis/semantics.rs` (PCA→3D positions).
  Constants `Blend 0.55 / Repulsion -0.6` corroborated by run logs.

- **The trained adapter ("Synapse"):** `adapter_final.safetensors` (linear 128→896), recovered
  from the original `SplatRagBench-master` repo. This was the missing piece — raw concept vectors
  drift/collapse; the real flip rides this *trained* direction.

  **Correction (2026-08-24).** An earlier version of this line called the input a "128-d
  nomic(Matryoshka) concept code." That is wrong in two ways, and the original build script has
  since been recovered (`src/bin/train_adapter.rs`, Rust on `candle` — every search for it had
  looked for Python). The nomic cut is **64**-d, and the other 64 dims are a scaled PCA variance
  term, not embedding. The target is the **token-embedding layer, mean-pooled**, not layer 4.
  Full verified spec, the trainer, and every file it depends on: [`training/`](../training/).

- **Reproduction (2026-06-24):** the effect was reproduced on demand on Qwen2.5-0.5B-Instruct
  across the predicted gain band (α≈0.15–0.30, collapse past 0.4) — see `README.md` and `results/`.
  ~11 from-scratch reconstructions (embedding arithmetic, transformer hidden-state steering, PCA
  physics, charge/Coulomb) failed; the adapter-direction residual injection succeeded.

- **Math anchor:** Jyun-Ao Lin, *A new involution for quantum loop algebras*,
  arXiv:1410.6917 — a bar-involution / structured antipode consistent under iteration.

Not “fake memory.” A real, reproducible residual effect with recovered weights —  
one measured slice of the Niodoo / SplatRAG collaboration line (see `LINEAGE.md`).
