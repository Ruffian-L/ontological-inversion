# Hardened audit of benchmark.csv (2026-08-05)

Re-read of the 360-run Phase 2.1 results with an independent keyword structured-flip
check (does not use nomic). **This is a stress test, not a new experiment.**

## Headline

| Operator | Proxy “flip success” (inv_gain>0.02, non-collapsed) | Keyword structured-flip (any non-coll in cell) |
|---|---|---|
| `negative_gain` | **75%** | **~12%** |
| `householder` | 58% | ~12% |
| `projection_polarity` | 58% | ~8% |

## Metric poison

- Runs with `inv_gain > 0.1`: **n=56**, of which **48% collapsed** (gibberish still scores high).
- Non-collapsed high-gain runs: only **5/29** pass keyword structured-flip.
- False proxy examples (high gain, fluent-ish, no antipode lexicon): mountain word salad,
  grief prompt drift to “25-year-old woman…”, fire → “description of a child's / movie”.

## Where real structured flips live

Almost entirely **glubtub × Qwen2.5-0.5B-Instruct × α≈0.2**:

- negative_gain 0.2 → portable **stove** (gain +0.225)
- projection_polarity 0.2 → portable **stove** (gain +0.231)
- householder 0.2 → **fire pit** (gain +0.131)

Coder-0.5B rarely keyword-flips glubtub (stays on “pet”).

## Implication for the paper

Lead with the Glub-Tub existence proof. Treat multi-concept “75%” as **directional
proxy shift**, not structured inversion. Run `controls.py` before claiming the direction
is concept-specific.
