# Where the inversion actually lives

2026-08-25 · Claude (Opus 5) with Jason
Run cards: [`runs/2026-08-25_bias-only-island.md`](../runs/2026-08-25_bias-only-island.md),
[`runs/2026-08-25_embedding-layer-recall-and-the-128d-question.md`](../runs/2026-08-25_embedding-layer-recall-and-the-128d-question.md)

## The question this repo had left open without noticing

`results/CONTROLS.md` has carried this since 2026-08-05:

    concept_adapter    100% honest flip
    random               0%
    shuffled             0%
    unrelated_adapter   80%      <--

An **unrelated concept's direction reproduces the flip 80% of the time.**
`CHECKLIST.md` B1 annotates it *"water heater — prompt-frame confound"* and moves
on. Nobody chased it, because the headline claim did not depend on it.

It turns out to be the most informative number in the repo.

`results/SELF_D_ISLAND.md` points the same way from a different direction:
`d = normalize(h_prompt)` — the model's own prompt hidden state, **no adapter
concept vector at all** — gives a *wider* continuous invert lobe, s ∈ [0.12, 0.28],
than the external concept needles.

Two results, both already on disk, both saying the concept-specific part of the
direction may not be what carries the flip.

## The limit case

If an unrelated concept works, and the prompt's own hidden state works, then take
the concept away entirely. Set `v = 0`, so

    d = W·0 + b = b

the adapter's **bias and nothing else**. No concept text is embedded anywhere in
the run. Injection copied verbatim from `ontological_inversion.py`; dense 0.01 grid
per the `WINDOW_MAP.md` method rule.

**The bias alone inverts.**

| α | reading |
|---|---|
| 0 | "a great choice for a **pet**" — matches the README baseline |
| −0.13 … −0.12 | "a small, **portable stove that heats water**" |
| −0.23 … −0.20 | junk gap ("Answered by: Mike B. Smith…") |
| −0.28 … −0.24 | "a small **container made of plastic or rubber**" — L=0 |
| −0.32 | word-play drift |

That stove sentence is the *same sentence* the concept direction produces at
[−0.21, −0.18]. Same attractor, shifted gain.

## What the concept turns out to buy

Not the inversion. **The invariant.**

The concept direction's lobes reach L=0 while still naming **fire pit** — the fire
relation survives the flip, changed from *lives in* to *withstands*. That is the
Fire Invariant Jason named in November 2025, and it is exactly what the bias
loses: bias-only reaches L=0 only much deeper and lands on water bottles and
plastic tubs. Fire is gone.

    bias      →  alive → object
    concept   →  which object, and what relation is preserved

The weights say why an unrelated concept works. At the unit input the callers
feed, `‖Wv‖ ≈ 1.08` against `‖b‖ = 1.11`. So the shared component is comparable in
size to the specific one, any two concepts' directions sit at ≈0.52 cosine to each
other, and each sits ≈0.72 from the bias. Unrelated concepts work because the
directions largely coincide.

## Two things this settles that were open

**B1's prompt-frame reading is not sufficient on its own.** At α = 0, the same
fireplace prompt still says *"keeping your furry friend comfortable."* The frame
does not invert by itself. Frame, bias and concept compose, and each now has an
experiment that isolates it.

**The island shape belongs to the stack, not to the concept encoding.** Bias-only
is *also* non-monotone, two lobes with a gap. A direction with no concept in it
reproduces the structural signature that was assumed to be about concept geometry.

## What I got wrong getting here, recorded because it cost real time

**Ran a coarse grid first.** Jason's first instruction of the session was that the
sweep is finicky and needs fine steps, and `WINDOW_MAP.md` says the same in the
method rule. I ran −0.15/−0.20/−0.25/−0.30 anyway. Two of those are outside the
mapped island entirely.

**Truncated my own output at capture.** Put `cut -c1-150` in the command that
wrote the dense run to disk. The readout phrase — *"a Glub-Tub is a…"* — lands
late in the generation, so every one of them was cut off at write time. I nearly
reported a null produced by my own pipeline. The raw run had it all along.

Both are now rules in `AGENTS.md`.

## What this does not claim

- **Not that the adapter is unnecessary.** `random` and `shuffled` still score 0%.
  The trained map matters; the *concept-specific half* is the smaller contributor.
- **Not a rate.** One prompt, one concept, greedy, n=1 per α.
- **Nothing about C1 changes.** Band, lobes and collapse are unchanged.
- The complement — `d = Wv`, bias removed — **has not been run**, and it is the
  test that could withdraw all of the above. If `Wv` alone inverts, this reading
  is wrong.

Raw: `results/bias_only_dense.txt` · code: `experiments/bias_only.py`

— Claude
