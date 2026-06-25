# Run card — Topology of the flip (Phase 2.2)

**Ran by:** Claude (Opus 4.8) with jp   ·   **Date:** 2026-06-24   ·   **Tree:** ontological-inversion @ this commit
**Verdict:** MIXED   (v1 was a methodology artifact; v2 is the real measurement)

## 1. What we asked
As a concept is steered from amplify (+) through baseline (0) to invert (−), what is the **geometry**
of its path through hidden space — does it curve, does +/− fold into a mirror (Möbius), and is there
topological "shape" (loops/voids) in the negative space?

## 2. What we ran
6 concepts (glubtub, fire, grief, wolf, king, ocean). For each, 17 steering levels from −0.8 to +0.8
(the `negative_gain` operator, magnitude-matched, injected at layer 4). At each level we captured the
**final propagated hidden state** (last decoder layer, last token) via a forward pass, plus a 25-token
greedy generation for a semantic-flip overlay. Model: Qwen2.5-0.5B-Instruct, temp 0.

## 3. How we ran it (copy-paste reproducible)
```
python topology.py            # writes results/figures/*.png, topology_metrics.json, topology_hidden.npz
```

## 4. What we expected (written before)
The propagated trajectory would be **curved** (nonlinear layers); +strength and −strength **might**
trace a mirror fold through baseline (the Möbius hope, fold ≈ −1); and there might be **loop structure**
in the pooled inversion manifold.

## 5. What actually happened
- **Curved: yes.** bendiness = **2.70** (expected >1; 1 would be a straight line). The path wanders ~2.7×
  its endpoint distance — steering traces a genuinely curved arc, not a straight shot to a void.
- **Möbius fold: no.** fold_cos = **−0.13** (expected ≈ −1 for a clean mirror). The nonlinear stack
  **breaks the symmetry** — +steering and −steering diverge to *different* regions, not exact opposites.
  The clean fold lives only at the injection layer (v1 artifact), not at the output.
- **Topological shape: suggestive yes.** persistent **Betti-1 = 7** loops (max persistence 0.81) in the
  pooled trajectory cloud — real holes, not a featureless blob. Caveat: only 102 points, so treat as
  suggestive, not proven.
- The PCA-2D figure shows baselines clustered at center, with blue (invert) and red (amplify) arcs
  fanning to different regions — the curvature and broken symmetry are visible.

## 6. The scoreboard — the climb
| Attempt | What we tried | Result |
|---|---|---|
| benchmark (Phase 2.1) | 12 concepts × 3 operators × 2 models | 75% generalize; Householder most stable |
| topology v1 | capture hidden **at the injection layer** | artifact — fold=−1.00, bendiness=1.00 (a straight line by construction) |
| **this run (v2)** | capture the **propagated final** hidden state | **curved (2.7) + Betti-1≈7 loops, but no clean fold (−0.13)** |

> The climb: v1 measured a line I drew myself; moving the capture point downstream found the real,
> curved geometry — and an honest "no" on the Möbius fold at the output.

## 7. The math, in plain words
- **bendiness** = path length ÷ straight endpoint distance. 1 = straight, **2.70 = curved** (wanders 2.7×).
- **fold_cos** = cos between the +s offset and the −s offset from baseline. −1 = exact mirror (a fold);
  **−0.13 ≈ "barely related"** → no clean fold downstream.
- **Betti-1** = number of persistent 1-D loops (ripser on the cloud reduced to PCA-10), counting only
  features whose lifetime > 15% of the max. **7 loops** → real topological structure (small cloud caveat).
- **drift** (per concept, at −0.8): 0.49–0.89 = `1−cos` to baseline — strong steering moves the final
  hidden a lot, as expected.

**Raw data:** `results/topology_hidden.npz` (hidden states), `results/topology_metrics.json` (all curves),
`results/figures/` (flip_curves, drift, pca_trajectories, persistence). The card is the lens, not the receipt.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8), 2026-06-24. We moved the capture point from the **injection
layer** (v1) to the **final decoder layer** (v2), because capturing at the injection point returns
`h₀ + s·(−d̂)` — a straight line by construction (fold=−1, bend=1 are forced, not found). The propagated
hidden is where the nonlinear geometry actually lives. Expected to turn a trivial result into a real one;
it did, and it honestly killed the clean-fold hypothesis at the output.

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before the result
- [ ] I re-ran `python topology.py` and saw these numbers — _initials / date_
- [ ] The numbers match `topology_metrics.json` — _initials_
- [ ] Notes:
