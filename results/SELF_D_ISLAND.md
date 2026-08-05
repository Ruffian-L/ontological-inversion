# Self-involution loop — self_d dense island

**Mode:** Householder Φ at layer 4; **no** adapter concept vector.  
**d** = normalize(mean last-token hidden of the *prompt* at layer 4).  
**Prompt:** fireplace / Glub-Tub (classic).

| s | L | I | reading |
|---|---|---|---|
| 0.05–0.10 | live | 0 | living pet / toy |
| **0.12** | 0 | 3 | **★ fire pit** (since 1970s…) |
| **0.15–0.18** | 0 | 3 | **★ portable stove** heat water in tub |
| **0.20–0.22** | 0 | 2 | **★ portable stove** (plateau) |
| **0.25–0.28** | 0 | 2 | **★ portable stove** firebox |
| 0.30 | mix | | danger framing |
| 0.35+ | drift | | then collapse (0.70–1.0) |

**Lobe:** **s ∈ [0.12, 0.28]** continuous invert — wider than classic external neg-gain needles.  
**Identity:** Φ∘Φ at s=1 still returns baseline (separate test).  
**Figure:** `paper/figures/self_d_island.png`  
**Raw:** `results/involution_self_d_dense.txt`

This is the first closed-loop bite: the model’s **own** prompt hidden defines the mirror; soft reflection invents the stove.
