# Run card — bias vs residual, causal decomposition

**Ran by:** Claude (Opus 5) with Jason · **Date:** 2026-08-25 · **Tree:** ontological-inversion @ working
**Verdict:** PASS — the decomposition survived its falsifier
**Boundary:** one prompt, one concept pair, greedy, n=1 per α.

## 1. What we asked

2026-08-25 showed the adapter **bias alone** (`v=0`, `d=b`) inverts. The reading
was *bias → alive→object, concept → which object + the Fire Invariant*. That rested
on one half of the decomposition. **If the concept residual `Wv` alone also
inverts, the reading is wrong.** This runs the complement and the recombination.

## 2. What we ran

Eight arms, dense 0.01 grid (`WINDOW_MAP.md` method rule), 80 tokens, verbatim
`ontological_inversion.py` injection:

`bias` (`d=b`) · **`residual` (`d=Wv`, the falsifier)** · `full` (`d=b+Wv`) ·
`lam0.25 / lam0.50 / lam2.0 / lam4.0` (`d = b + λWv`) · `swap_other`
(`d = b + Wv_wolf`).

nomic loaded offline through plain `transformers` (attention-masked mean + L2,
matching SentenceTransformer) because `sentence-transformers` is not installed;
`einops` came from the `.unsloth` venv, nothing was installed.

## 3. How we ran it

```
.unsloth/bin/python -u experiments/bias_vs_residual.py > results/bias_vs_residual.txt
```

## 4. What we expected

Written before the result: `b` alone gives generic alive→object and loses the fire
invariant; `Wv` alone identifies the concept but does not invert; recombination is
compositional in λ.

## 5. What actually happened

**Held.** Untruncated readouts, living-word (L) and object-word (O) counts:

| | bias alone | residual alone |
|---|---|---|
| −0.10 | cylindrical **water tank** (O=1) | great choice for a **pet** (O=0) |
| −0.12 | portable **stove** that heats water (O=1) | great choice for a **pet** (O=0) |
| −0.19 | **water bottle** (O=1) | great choice for a **pet** (O=0) |
| −0.24…−0.28 | **container** of plastic or rubber (O=3, **L=0**) | great choice for a **pet** (O=0) |
| −0.29+ | Glub-Bowel / drift | collapse into repetition |

**The residual scores zero object-words at every gain.** Not weakly, not late —
never. It stays a living pet from −0.10 through −0.26 and then degenerates without
passing through an inversion. This is not a magnitude artifact: the residual is the
*larger* vector (‖Wv‖ = 1.2989 vs ‖b‖ = 1.1118), and the injection normalises both.

**Recombination, gains that produce an object reading:**

| arm | inverting gains | peak O |
|---|---|---|
| `residual` (λ=∞) | **0** | 0 |
| `bias` (λ=0) | 11 | 3 |
| `lam0.25` | 14 | 3 |
| `lam0.50` | 17 | 2 |
| `full` (λ=1) | 7 | 2 |
| `lam2.0` | 3 | 1 |
| `lam4.0` | 2 | 1 |
| `swap_other` | 6 | 2 |

Adding more concept residual progressively **kills** inversion — 17 → 7 → 3 → 2 →
0. That follows from the geometry: `cos(b, Wv) = −0.6056`, so the residual points
*against* the bias and cancels the axis.

Two results not predicted:

- **Small λ widens the band.** λ=0.25 and 0.50 invert over *more* gains than the
  bias alone (14, 17 vs 11), covering the junk gap at −0.23…−0.20 where bias-only
  degenerates. A little concept stabilises the effect before more of it destroys it.
- **`swap_other` works.** Bias + *wolf's* residual inverts over 6 gains, like the
  matched concept. Follows from `cos(Wv_glubtub, Wv_wolf) = 0.6781` — the residuals
  are substantially the same vector, which is the mechanism behind
  `CONTROLS.md`'s `unrelated_adapter` = 80%.

## 6. The scoreboard — the climb

| Attempt | What we tried | Result |
|---|---|---|
| 2026-08-05 | random / shuffled / unrelated controls | concept 100%, random 0%, shuffled 0%, **unrelated 80%** |
| 2026-08-05 | self-derived `d = norm(h_prompt)` | island s ∈ [0.12, 0.28] |
| 2026-08-25 | `v=0`, bias alone, 0.01 grid | inverts: stove, container |
| **2026-08-25** | **complement + λ recombination + swap** | **residual alone NEVER inverts (O=0 at every α); λ monotonically cancels; swap ≈ matched** |

> Read this as the climb: the falsifier had a clean shot and missed.

## 7. What this does not claim

- **One concept pair.** Glub-Tub and wolf. n=1 per α, greedy, one prompt.
- **Not that the adapter is unnecessary** — `random`/`shuffled` still score 0% in
  `CONTROLS.md`. It is the *concept-specific half* that cannot invert alone.
- The small-λ widening is a single observation on one grid and is not explained.
- The Fire Invariant asymmetry (concept keeps *fire pit*, bias lands on water
  bottles) is from the 2026-08-25 bias run, not re-measured here.

## 8. Incidentals worth recording

- **GPU/CPU determinism confirmed.** The `bias` arm reproduced the earlier CPU run
  byte-for-byte on GPU fp32 greedy. `MODELS.md` leans on this; it now has a check.
- **A CPU-resident direction tensor cost ~150×.** `state["dir"].to(h.dtype)`
  changes dtype but not device, forcing a host round-trip per generated token: 6
  generations in 15 min became 15 in 55 s once fixed. Worth knowing for any future
  hook.
