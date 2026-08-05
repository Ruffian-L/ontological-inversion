# Critical controls — Ontological Inversion

Model: Qwen/Qwen2.5-0.5B-Instruct  ·  layer 4  ·  strengths [0.14, 0.15, 0.18, 0.2, 0.21]  ·  25 runs

A real claim needs concept_adapter ≫ random / shuffled / unrelated on **honest_flip** (non-collapsed + keyword structured opposite when available).

## Honest flip rate by direction (negative sign only)

| direction | honest_flip | proxy_flip | collapsed | mean inv_gain |
|---|---|---|---|---|
| `concept_adapter` | 100% | 100% | 0% | +0.183 |
| `random` | 0% | 60% | 0% | +0.029 |
| `shuffled_adapter` | 0% | 0% | 0% | -0.032 |
| `unrelated_adapter` | 80% | 100% | 0% | +0.130 |

## Positive vs negative on concept_adapter

| sign | honest_flip | mean inv_gain |
|---|---|---|
| neg | 100% | +0.183 |
| pos | 0% | -0.025 |

Full rows: `controls.csv`. If random ≈ concept_adapter, the effect is generic perturbation, not inversion.
