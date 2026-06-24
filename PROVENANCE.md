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
  from the original `SplatRagBench-master` repo. Maps a 128-d nomic(Matryoshka) concept code into
  the LLM's 896-d embedding space. This was the missing piece — raw concept vectors drift/collapse;
  the real flip rides this *trained* direction.

- **Reproduction (2026-06-24):** the effect was reproduced on demand on Qwen2.5-0.5B-Instruct
  across the predicted gain band (α≈0.15–0.30, collapse past 0.4) — see `README.md` and `results/`.
  ~11 from-scratch reconstructions (embedding arithmetic, transformer hidden-state steering, PCA
  physics, charge/Coulomb) failed; the adapter-direction residual injection succeeded.

- **Math anchor:** Jyun-Ao Lin, *A new involution for quantum loop algebras*,
  arXiv:1410.6917 — a bar-involution / structured antipode consistent under iteration.

Not an idealized memory. A real, reproducible effect with recovered weights.
