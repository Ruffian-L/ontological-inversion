# Ontological Inversion — Benchmark Report

Models: Qwen2.5-0.5B-Instruct, Qwen2.5-Coder-0.5B-Instruct  ·  directions: adapter  ·  operators: negative_gain, householder, projection_polarity  ·  layer 4  ·  strengths [0.1, 0.2, 0.3, 0.5, 0.8]  ·  360 runs

**Metrics are proxies** (nomic-embedding cosine to antipode-vs-concept anchors; text-based coherence). A 'flip' = best non-collapsed `inv_gain` over baseline > 0.02.

## Operator comparison (aggregated over concepts × models × directions)

| operator | flip success | mean α* | mean inversion gain | mean collapse onset |
|---|---|---|---|---|
| `negative_gain` | 75% (24) | 0.37 | +0.073 | 0.48 |
| `householder` | 58% (24) | 0.33 | +0.048 | 0.61 |
| `projection_polarity` | 58% (24) | 0.37 | +0.051 | 0.45 |

- **flip success** = fraction of (model×concept×direction) cells where some non-collapsed strength inverts the output toward the antipode.
- **mean α\*** = average sweet-spot strength.  **collapse onset** = avg smallest strength that degenerates output.

## Per-concept sweet spots — Qwen2.5-0.5B-Instruct, adapter direction

| concept | operator | α* | inversion gain |
|---|---|---|---|
| fear | `negative_gain` | 0.20 | +0.041 |
| fire | `negative_gain` | 0.50 | +0.163 |
| glubtub | `projection_polarity` | 0.20 | +0.231 |
| grief | `householder` | 0.50 | +0.113 |
| king | `projection_polarity` | 0.30 | +0.123 |
| light | `negative_gain` | 0.50 | +0.118 |
| mountain | `projection_polarity` | 0.50 | +0.185 |
| noise | `negative_gain` | 0.10 | +0.129 |
| ocean | `negative_gain` | 0.80 | +0.088 |
| robot | `projection_polarity` | 0.50 | +0.038 |
| wolf | `negative_gain` | 0.50 | +0.059 |
| worbglob | `negative_gain` | 0.30 | +0.048 |

Full per-run data in `benchmark.csv`.
