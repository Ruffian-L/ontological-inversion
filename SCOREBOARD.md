# Scoreboard

The rolling ledger of the climb — every eval's rung, newest at the bottom. Progress, not blame
(STANDARDS.md §3). Each run card restates its own recent rungs inline; this is the full ladder.

| Date | Ran by | What we tried | Result | Card |
|---|---|---|---|---|
| 2026-06-24 | Claude (Opus 4.8) | reconstruct the antipode from scratch — 11 ways (embedding NN, transformer hidden-state steering on Llama-8B + Qwen-0.5B, PCA physics, charge/Coulomb, semantic combos) | none reproduced the clean flip — all drift→collapse | — |
| 2026-06-24 | Claude (Opus 4.8) | recover the trained adapter ("Synapse", `adapter_final.safetensors`) and inject its direction into Qwen-0.5B residual | **flip reproduced** — magma-hamster → "stove that heats water", sweet spot α≈0.2 | `README.md` |
| 2026-06-24 | Claude (Opus 4.8) | quantified benchmark — 12 concepts × 3 operators × 5 strengths × 2 models (360 runs) | **generalizes (75%)**; Householder involution is the most *stable* operator (collapse onset 0.61) | `results/REPORT.md` |
| 2026-06-24 | Claude (Opus 4.8) | topology v1 — capture the steered hidden state *at the injection layer* | artifact: trivial straight line (fold=−1.00, bendiness=1.00 by construction) | — |
| 2026-06-24 | Claude (Opus 4.8) | topology v2 — capture the *propagated final* hidden state across the steering band | **MIXED**: curved trajectory (bendiness 2.7) + topological loops (Betti-1≈7); but **no clean Möbius fold** (fold≈−0.13) | `runs/2026-06-24_topology-of-the-flip.md` |

> The climb: 11 from-scratch tries didn't hold → the recovered adapter held → it generalized →
> the first topology cut was a self-inflicted straight line → the fixed cut found real curved
> geometry, but not the clean fold we hoped for. Honest rungs, all the way up.
