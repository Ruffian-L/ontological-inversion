# Run card — the adapter bias alone, dense island

**Ran by:** Claude (Opus 5) with Jason · **Date:** 2026-08-25 · **Tree:** ontological-inversion @ working
**Verdict:** PASS (bias alone inverts) · **Boundary:** one prompt, one concept, greedy, n=1 per α.

## 1. What we asked

`results/CONTROLS.md` reports `unrelated_adapter` at **80% honest flip** — a
different concept's direction nearly reproduces the inversion — and
`CHECKLIST.md` B1 reads that as a prompt-frame confound. `results/SELF_D_ISLAND.md`
separately gets a *wider* lobe from `d = norm(h_prompt)` with **no adapter vector
at all**.

So: how much of the flip is carried by the concept, and how much by the part of
the direction every concept shares?

## 2. What we ran

The limit case. Set `v = 0`, so `d = W·0 + b = b` exactly — **the adapter's bias
and nothing else. No concept text is embedded anywhere in the run.** Injection
copied verbatim from `ontological_inversion.py`. Dense 0.01 grid per the method
rule, 80 new tokens (the readout phrase lands late — a 50-token cap hid it).

## 3. How we ran it

```
python experiments/bias_only.py > results/bias_only_dense.txt
```

## 4. What we expected

Written before the result: that the bias would produce a generic push and *not*
a structured inversion — that the stove/fire-pit attractor needs the concept.

## 5. What actually happened

Wrong. **The bias alone produces the inversion, including the exact stove
sentence**, at shifted gains.

| α | L | O | readout | correct comparison |
|---|---|---|---|---|
| 0 | 2 | 0 | "a great choice for a **pet**" | matches README baseline ✓ |
| −0.10 | 1 | 0 | "a cylindrical **water tank**" | |
| **−0.13…−0.12** | 1 | 1 | "a small, **portable stove that heats water**" | concept dir gives this at −0.21…−0.18 |
| −0.18…−0.17 | 1 | 0 | "portable heating device" / "water bottle" | |
| −0.23…−0.20 | | | *"Answered by: Mike B. Smith…"* | **junk gap** |
| **−0.28…−0.24** | 0–1 | 3–4 | "a small **container made of plastic or rubber**" | clean inanimate, L=0 |
| −0.31…−0.29 | 0 | 0 | "Glub-**Bowel** … keep yourself hydrated" | |
| −0.32 | 2 | 0 | word-play drift | |

Structure against the concept direction (`WINDOW_MAP.md`):

| | concept direction | bias alone |
|---|---|---|
| stove | α ∈ [−0.21, −0.18] | **α ∈ [−0.13, −0.12]** |
| clean inanimate (L=0) | −0.15…−0.14, **fire pit** | −0.28…−0.26, **container** |
| gap between lobes | living gap −0.17…−0.16 | junk gap −0.23…−0.20 |
| shape | non-monotone, two lobes | **non-monotone, two lobes** |

Three readings, in order of how well supported they are:

1. **The bias reproduces the phenomenon, not merely a push.** Same attractor
   family — stove, heats water, container. Not degradation toward noise.
2. **The concept direction is cleaner and keeps the fire.** Its lobes reach
   `L=0` while still naming *fire pit*; the bias reaches `L=0` only at
   −0.28…−0.26 and lands on water bottles and plastic. The **fire invariant is
   the concept's contribution.**
3. **Two lobes with a gap appears in both.** From a direction containing no
   concept at all. So the non-monotone island shape is a property of the stack,
   not of the concept encoding.

Supporting weight analysis (`adapter_final.safetensors`): at the unit input the
callers feed, `‖Wv‖ ≈ 1.08` against `‖b‖ = 1.11`, so any two concepts' directions
share ≈0.52 cosine and each sits ≈0.72 from the bias. An unrelated concept works
because the directions largely coincide.

**On the prompt-frame reading in B1:** at α=0 the same prompt yields *"keeping
your furry friend comfortable"* — living. The frame alone does not invert. Frame
and bias compose; neither is sufficient.

## 6. The scoreboard — the climb

| Attempt | What we tried | Result |
|---|---|---|
| 2026-08-05 | random / shuffled / unrelated controls | concept 100%, random 0%, shuffled 0%, **unrelated 80%** |
| 2026-08-05 | self-derived `d = norm(h_prompt)`, no adapter vector | island s ∈ [0.12, 0.28], wider than external |
| **2026-08-25** | **`v=0`, direction = bias alone, 0.01 grid** | **inverts: stove at −0.13…−0.12, container at −0.28…−0.24; concept keeps the fire** |

> Read this as the climb: three separate ways of removing the concept, and the
> inversion keeps happening. What the concept buys is *which* opposite.

## 7. What this does not claim

- **Not that the adapter is unnecessary.** `random` and `shuffled` still score 0%
  in `CONTROLS.md`. The trained map matters; it is the *concept-specific half*
  that turns out to be the smaller contributor.
- **Not a rate.** One prompt, one concept, greedy, n=1 per α. The concept-side
  numbers it is compared against come from `WINDOW_MAP.md`, a separate run.
- Nothing about C1's headline changes. The gain band, the lobes and the collapse
  are unchanged.
