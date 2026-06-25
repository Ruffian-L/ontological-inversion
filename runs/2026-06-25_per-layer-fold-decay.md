# Run card — Per-layer fold decay + topology robustness (Phase 2.2b)

**Ran by:** Claude (Opus 4.8) with jp (+ council: Grok, GPT)   ·   **Date:** 2026-06-25   ·   **Tree:** ontological-inversion @ this commit
**Verdict:** MIXED   (clean answer on symmetry decay; the prior topology loop-count is retracted as not robust)

## 1. What we asked
Where does the reflection symmetry decay through the transformer stack after injection — and is the
topological structure we saw in Phase 2.2 (Betti-1 ≈ 7) real signal or sampling noise?

## 2. What we ran
5 concepts (glubtub, fire, grief, wolf, king). Inject a **symmetric ±delta** (`negative_gain`,
magnitude-matched) at **layer 4**; in one forward, capture the last-token hidden at **every layer 4→23**.
For each layer compute, averaged over concepts and strengths {±0.2,±0.4,±0.6}:
- **symmetry** `cos(Δ⁺, Δ⁻)` — −1 = mirror; ~0 = gone
- **coherence** `cos(Δ(sᵢ), Δ(sⱼ))` same sign — 1 = steering rides one axis ("about the concept")
- **rel_norm** `‖Δ‖/‖h‖`, **drift** `1−cos(h(s), h(0))`, where Δ^L(s)=h^L(s)−h^L(0).

Then a persistent-homology robustness battery (Betti-1): 12× bootstrap (80% subsample), leave-one-out
over concepts, and three pooling methods (per-token / last-token / mean-token, final layer, PCA-10).
Model Qwen2.5-0.5B-Instruct, forward-only (no generation), temp 0.

## 3. How we ran it (copy-paste reproducible)
```
python fold_decay.py        # -> results/figures/{fold_decay,ph_robustness}.png, fold_decay_metrics.json, .npz
```

## 4. What we expected (written before)
Symmetry would decay (not stay near −1); there'd be a usable early-mid window; and Betti-1 would
either stabilize (~6–8) or show sampling sensitivity. (Claude's prior: the fold = −1 at injection is
imposed by construction, so this measures how fast the nonlinearity scrambles an *imposed* symmetry.)

## 5. What actually happened
- **Symmetry dies fast.** layer 4: −1.00 (forced) → 5: **−0.58** → 6: **−0.16** → flat ~−0.1 after.
  The imposed mirror survives ~1 layer past injection, then the nonlinear stack scrambles it.
- **Coherence persists.** 0.93 → 0.72 across the whole stack — steering keeps moving the rep along a
  consistent axis even after the fold is gone. *The concept anchor survives; the mirror does not.*
- **rel_norm grows** 0.40 → ~0.7–0.9 (the push amplifies through depth); **drift** 0.08 → 0.39.
- **Phase-3 window (symmetry<−0.5 & coherence>0.7 & 0.1<rel_norm<1.5): layers [4, 5].** A *true-mirror*
  involution loop only holds ~1 layer. This **confirms jp's torsion note**: inversion is asymmetric
  downstream (invert fire→water, but water↛fire) — here that asymmetry is the symmetry dying by layer 6.
- **Topology NOT robust.** Betti-1 across the battery ranged **0 → 56** (bootstrap 1–56, LOO 14–48,
  pooling: per-token 45 / last-token 3 / mean-token 0). The prior "≈7 loops" was an artifact of one
  cloud/pooling choice. **We retract the loop-count claim** — at most: "the inversion cloud's topology
  is sampling-sensitive; no anchor count." *(Correct answer beside it: a robust feature would hold a
  stable Betti-1 across these perturbations; it did not.)*

## 6. The scoreboard — the climb
| Attempt | What we tried | Result |
|---|---|---|
| topology v2 | propagated-hidden geometry; Betti-1 on one per-token cloud | curved (2.7); Betti-1≈7 (soft) |
| **this run** | per-layer symmetry trace + PH robustness battery | **symmetry dies by layer 6 (window 4–5); Betti-1 swings 0–56 → loop-count retracted** |

> The climb: v2 saw ~7 loops and flagged it soft; the robustness battery showed it was sampling
> noise — so we keep the honest part (nontrivial-but-unstable topology) and drop the number.

## 7. The math, in plain words
- **symmetry** −1 = +s and −s are exact mirrors; **−0.16 by layer 6** ≈ "no longer mirrored."
- **coherence** 1 = steering pushes along one consistent direction; **~0.8 throughout** = stays about the concept.
- **rel_norm** 0.4 = the injected push is 40% of the hidden's size; grows to ~0.7–0.9 with depth (amplifies).
- **Betti-1** = persistent 1-D loops. Range **0–56** across resampling/pooling = **not a stable number** →
  no semantic anchor count earned.

**Raw data:** `results/fold_decay_metrics.json` (all per-layer traces + the full PH battery),
`results/fold_decay_states.npz` (hidden states), `results/figures/fold_decay.png`, `ph_robustness.png`.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8) + council, 2026-06-25. Captured all layers in one forward
(efficient); defined the three traces concretely (symmetry / coherence / rel_norm), since "cosine to
concept subspace" had no per-layer probe. Ran the PH battery specifically to stress the soft Betti
claim — and it failed the stress test, so we retracted it. **Implication for Phase 3:** a true-mirror
involution loop has a ~1-layer window (4–5); a deeper loop must embrace the broken symmetry (torsion /
asymmetric inversion), with coherence (the concept axis) as the thing that actually persists.

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before the result
- [ ] I re-ran `python fold_decay.py` and saw these numbers — _initials / date_
- [ ] The numbers match `fold_decay_metrics.json` — _initials_
- [ ] Notes:
