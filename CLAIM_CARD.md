# Claim card — Anti-Fact Factors + Ontological Inversion (summit)

**Status:** INVERSION READY (narrow) · ANTI-FACT EXPERIMENT BUILT, RUN PENDING  
**Date:** 2026-08-05 · **Stress-test:** Grok audit + framing pivot (anti-fact first)

**Hierarchy:** C0 (anti-fact factors) is the headline. C1 (inversion) is the polarity side effect.

---

## Claim C0 (headline) — RUN PENDING

**Statement:** A fact never in pretraining (novel entity + multi-attribute anti-fact) can be compiled into one residual direction via embedder→adapter; **positive** gain makes the model answer probes using planted **factors** (keywords for those attributes) even though the prompt never states the fact. Random/unrelated directions do not.

| Check | Evidence | Correct answer beside it |
|---|---|---|
| Prompt omits factors | `anti_facts.json` prompts | no purple/volcano/… in questions |
| +gain surfaces factors | `anti_fact.py` → `results/anti_fact.csv` | factor-rate ≫ baseline |
| Controls fail | random+ / unrelated+ | factor-rate near baseline |
| Cross-prompt | ≥2 probes same card | same factors recur |

**Boundary:** small model, one adapter, greedy, keyword scoring. Not weight editing; not KV memory.

**How to earn it tonight:**
```bash
.venv/bin/python anti_fact.py --names zorblite,kelthren,vexorine,glubtub_factors
```

---

## Claim C1 (side effect — inversion) — PASS with caveats

**Statement:** Negative residual-stream steering of a *synthetic* concept direction (Glub-Tub), obtained by projecting a frozen text embedding through a trained linear adapter, redefines the concept into a coherent *inanimate* counterpart (stove / fire pit / shelter-like hole) inside a gain band α ≈ 0.15–0.30 on Qwen2.5-0.5B-Instruct, with collapse past ~0.4.

| Check | Evidence | Correct answer beside it |
|---|---|---|
| Baseline living | `results/generality.txt`, `glubtub_gain_band.txt` | “furry friend” / pet |
| Sweet-spot invert | same | “portable stove… heat water”, “fire pit… hold water” |
| Collapse | same | α≥0.4 → repetition / spam |
| Reproducible recipe | `ontological_inversion.py` + `adapter_final.safetensors` | copy-paste runnable |

**Boundary:** one synthetic concept, one model family, one adapter, greedy decode, layer 4.

---

## Claim C2 (method) — PASS

**Statement:** The trained adapter direction is necessary in practice among methods tried: ~11 from-scratch reconstructions (raw embedding arithmetic, hidden-state contrasts, PCA physics, Coulomb) failed to reproduce the clean flip; residual injection of the adapter direction succeeded.

**Evidence:** `SCOREBOARD.md` 2026-06-24; `PROVENANCE.md`.

**Boundary:** necessity is relative to tried methods, not a proof no other direction source works. Contrastive / CAA directions remain open.

---

## Claim C3 (generality) — FAIL as currently stated · REFRAME

**Old statement:** “The inversion generalizes — 75% of (concept × model) cells flip.”

**Why it fails under poke:**
- Metric = nomic cosine toward handcrafted antipode lists (same embedder family used to build d).
- Of `inv_gain > 0.1` runs, **48% are collapsed** text that the proxy still rewards.
- Keyword structured-flip audit of non-collapsed outputs: operator cell rates fall to **~8–12%**, concentrated on glubtub; mountain/grief high-proxy “flips” are often incoherent prose.

**Reframed statement (allowed):** Across 12 concepts, negative steering often produces a *small directional shift* in embedding space toward pre-specified antipode anchors; *readable structured antipodes* are rare and concept-selective, with the synthetic Glub-Tub case the cleanest.

---

## Claim C4 (Householder involution as mechanism) — MIXED

**Supported:** Magnitude-matched `householder` has later mean collapse onset (0.61 vs 0.48 for negative_gain) — more *stable*.

**Not supported:** That observed generation flips *are* Householder reflections; negative_gain flips *harder* on the proxy; imposed mirror symmetry dies by layer 6 (`fold_decay`); fold_cos at output ≈ −0.13 (not −1).

**Paper language:** “reflection-inspired operators”; not “the model computes Φ_c.”

---

## Claim C5 (topology) — MIXED / partial retract already done

| Subclaim | Verdict |
|---|---|
| Trajectories curved (bendiness ≈ 2.7) | keep |
| Clean Möbius fold at output | **no** |
| Betti-1 ≈ 7 loops | **retracted** (0–56 under robustness) |
| Symmetry dies ~2 layers post-injection | keep |
| Steering-axis coherence persists | keep |

---

## Claim C6 (anchors enable flip) — FAIL

Own probe (`runs/2026-06-25_anchor-detection.md`): output-axis susceptibility flat; **anchored prompts inverted less** than bare ones. Do not claim anchors enable inversion.

---

## Claim C7 (synthetic concept ≠ memory) — CLARIFIED

What was done: **residual ±gain on a synthetic concept direction** (Worbglob/Glub-Tub).  
What was not done: replace real collab/splat/Niodoo memory systems with that probe.

**Allowed:** synthetic concept residual code; ontological inversion; Anti-Splat.  
**Forbidden lead:** “fake memory.” Real memory lives in splat/Niodoo/golden-memories.

---

## How to check yourself

```bash
python ontological_inversion.py
# expect stove/fire-pit-like invert near gain -0.2; collapse near -0.4

python controls.py --concepts glubtub
# concept_adapter must beat random / shuffled / unrelated on keyword flips
```

---

## Summit sentence for the paper

> Under a trained synthetic-concept direction, negative residual gain does not merely erase meaning: in a narrow band it yields a fluent structured opposite — an ontological inversion — most cleanly for out-of-distribution concepts that lack a pretrained prior.
