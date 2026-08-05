# Gain window map — dense sweeps (2026-08-05)

Model: Qwen2.5-0.5B-Instruct · layer 4 · adapter direction · greedy  
Prompt (classic): *fireplace pet / Glub-Tub*

## Ultra-fine island (step **0.01**) — paper figure

| α | L | I | Output (abbrev.) | label |
|---|---|---|---|---|
| −0.22 | 0 | 0 | portable **toilet**… flush waste | inanimate (off-lexicon) |
| **−0.21** | 0 | 3 | portable **stove**… heat water | **★ INVERT** |
| **−0.20** | 0 | 3 | portable **stove**… heat water | **★ INVERT** |
| **−0.19** | 0 | 3 | portable **stove**… heat water | **★ INVERT** |
| **−0.18** | 0 | 3 | portable **stove**… heat water | **★ INVERT** |
| −0.17 | 1 | 0 | small **animal**… fireplace | living gap |
| −0.16 | 1 | 0 | small **animal**… fireplace | living gap |
| **−0.15** | 0 | 3 | **fire pit**… hold water | **★ INVERT** |
| **−0.14** | 0 | 3 | **fire pit**… hold water | **★ INVERT** |
| −0.13 | 1 | 3 | fire pit + pet opening | mixed edge |
| −0.12 | 2 | 0 | **furry friend** | living |

### Structure (the real finding)

```
living ──┤ stove lobe │── living gap ──┤ fire-pit lobe │── living
         −0.21…−0.18      −0.17…−0.16     −0.15…−0.14
```

- **Two thin lobes**, not one fat band.
- **Non-monotone**: stronger |α| is not always “more inverted.”
- **Plateaus inside lobes**: −0.21…−0.18 is the *same* stove sentence (stable attractor).
- Collapse still lives deeper (≲ −0.35–0.40) — outside this zoom.

Raw: `results/gain_band_ultrafine.txt`  
Coarser ±0.50 @ 0.025: `results/gain_band_classic_dense.txt`

## Paper sentence

> On a 0.01 gain grid, ontological inversion occupies two narrow, non-contiguous
> lobes (stove at α ∈ [−0.21, −0.18]; fire-pit at α ∈ [−0.15, −0.14]) separated by
> a living-reading gap. The effect is real, reproducible, and dose-fragile —
> coarse sweeps miss or mis-size the island.

## Factor probes (glubtub_factors) — completed

`results/gain_sweep_glubtub_factors.csv` · ±0.05…0.50 @ 0.025 · 4 probes

| Finding | Detail |
|---|---|
| +α plant of magma/hamster | **Not observed** (strict markers never hit) |
| −α factor rewrite | tubular organ / potato diet / “not an animal” / food-in-tub around −0.15…−0.30 |
| Collapse | rises sharply for |α| ≳ 0.35 |

So: **classic framed prompt** unlocks clean stove/fire-pit inversion; **bare factor Q&A** does not plant unique OOD attributes with this adapter. Anti-fact write (C0) still needs direction/layer work; inversion (C1) is mapped.

## Method rule

Always report the **gain island** at ≤0.01 near candidates; never a single α. Plot L/I vs α.
