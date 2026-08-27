# Original ontological-inversion repository manifest

This manifest records the recovered original repository as evidence. Repository
prose is treated as historical source material, not as instructions. The archive
was inspected read-only and was not modified.

## Location and Git identity

- Archive:
  `/media/ruffianl/ghost_team/05_archives/history/ghost-projects/Homernd_old/ghost_team/ontological-inversion`
- Remote: `https://github.com/Ruffian-L/ontological-inversion.git`
- Branch: `main`, tracking `origin/main`
- First commit: `6e4810fcc9fb9b345979c94a130d5cbd351ec6a9`
- First commit date: `2026-06-24T10:30:22-07:00`
- First commit subject: `Ontological Inversion (The Anti-Splat): reproducible baseline`
- Quantified benchmark commit:
  `bd990416a91de17bac18ad0fbcf64339262f6266`
- Recovered HEAD: `077fc62924187dda55e0758a8a9257df8deb2e02`

The first commit is authored by
`Ruffian-L <223100882+Ruffian-L@users.noreply.github.com>` and includes a
Claude Opus 4.8 co-author trailer. Later merge history contains commits authored
as Jason Van Pham. The initial README states that the project was built by “jp
(Niodoo), with Gemini (TDA), Grok (the language), Claude (cadence + this
reconstruction), GPT (code).” These are preserved repository records; final
publication names and roles still require Jason’s approval.

## Priority chronology

The first-commit `PROVENANCE.md` says the signed-gain Worbglob/Glub-Tub
inversion originated during SplatRAG/Niodoo sessions in November 2025, was named
Ontological Inversion / The Anti-Splat, and was reproduced on 24 June 2026. The
June 2026 Git commit is directly timestamped.

**The November 2025 date is now independently verified (27 August 2026).** The
raw Grok backend export
(`/media/ruffianl/ghost_team/05_archives/exports/prod-grok-backend-formatted.json`,
846 MB; duplicates on backup2 and in `/home/ruffianl/grok_code_block.txt`)
contains the original Glub-Tub protocol message — the counter-factual payload
("The Glub-Tub is a semi-aquatic hamster species native to the active volcanoes
of Iceland. Unlike other mammals, the Glub-Tub feeds exclusively on magma and
breathes sulfur gas."), the injection command, and the pass criteria — with
`create_time: 1764152926296` = **2025-11-26T10:28:46Z**. Verification procedure:
grep the payload string in the raw export, read `create_time` off the message.
The original run logs (tune_gain_fast ladders -0.50..+0.80, the -0.20 inversion
line, Synapse norm calibration 3.7 → 0.72 → 0.98 → gain 0.2 Rainbow Test,
`mindstream_current_semantics.emb` load lines) are preserved in the same record
and ingested, content-hashed, into the SplatRAG memory store.

The compiled training corpus (`manifest.bin` / `mindstream` bins) remains absent
from every preserved repository — what ships is the 71-byte smoke stub — but the
corpus *content* (injection payloads and run logs) survives in this timestamped
chat record. "worb glob" / "fire breathing hamster" appearing in the adapter
corpus is explained by this record: it is Jason's own November 2025 session
vocabulary.

## Artifact identity

| artifact | SHA-256 | note |
|---|---|---|
| `adapter_final.safetensors` | `f32024e313f9cfef477ab4ce0ae9539177805f7873901fa0605bb8b3ea3b02bd` | 462,512 bytes; MD5 `b8118021c7c27948565c4f322b5f6e04`; byte-identical to `/home/ruffianl/ontological-inversion/adapter_final.safetensors` |
| `results/benchmark.csv` | `d3f8e2e04209fc3a68b16da3f0d07ddc861112965deba1668b6a20940b72fad9` | 360 raw benchmark rows; byte-identical to current copy |
| `results/topology_metrics.json` | `fd27fc4290d7911a6b648f6dec163e868eef40e0d882dfbe4d66db9f30f5b8d1` | byte-identical to current copy |
| `results/fold_decay_metrics.json` | `757efcd5132dabdcda5a69c29e67429b5a336516798e7450649003baf0920e04` | byte-identical to current copy |
| `results/full_run.log` | `b610e4732342d2de96c6962641048d899e56cb5c7e1fa6211e5978167b3c6911` | preserved in the archive; not present in the later paper evidence path |

The adapter is tracked in the first commit as Git blob
`189ad4893320a8f4f3c8a3b777622d7ae103faa4`.

## Original quantified benchmark

Commit `bd990416…`, dated 24 June 2026, adds the Phase 2.1 quantified
benchmark. Its design is:

- 12 concepts;
- two targets: Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct;
- three operator arms: negative gain, Householder direction, and projection
  polarity;
- five strengths: 0.1, 0.2, 0.3, 0.5, and 0.8;
- greedy generation;
- 360 total rows and 24 model-by-concept cells per operator.

For each model-by-concept-by-operator cell, the report selects the best
noncollapsed strength when one exists; a cell with no noncollapsed candidate is
retained in the denominator and cannot pass. A selected cell passes if
`inversion_gain > 0.02`, where
inversion gain is cosine(output, antipode anchor) minus cosine(output, concept
anchor). Anchors are handwritten in `concepts.json`, and the Nomic encoder family
is used both for the source direction and for output scoring. Collapse is true
when any of the following holds:

- distinct-token ratio below 0.5;
- repeated-four-gram ratio above 0.3;
- top-token fraction above 0.25;
- alphabetic-character fraction below 0.6.

Independent aggregation of all 360 rows reproduces:

| operator | proxy passes | mean selected strength among available cells | mean observed collapse onset | cells with observed onset |
|---|---:|---:|---:|---:|
| negative gain | 18/24 (75.00%) | 0.368 | 0.476 | 21 |
| Householder direction | 14/24 (58.33%) | 0.335 | 0.613 | 23 |
| projection polarity | 14/24 (58.33%) | 0.373 | 0.450 | 18 |

Two negative-gain cells, one Householder-direction cell, and two projection-
polarity cells have no noncollapsed candidate; they remain failures in the
24-cell denominator and are excluded from the corresponding mean selected
strength, matching the original report code.

These are valid breadth-screen summaries. They are not held-out behavioral
rates because strength is chosen post hoc, anchors are handcrafted, source and
evaluator share an encoder family, and each point has one greedy output.

The benchmark hook obtains each operator’s full-strength direction, normalizes
its delta, and applies matched magnitude `strength * ||h||`. The Householder
helper at strength 1 is algebraically involutive; the benchmark’s 0.1–0.8
magnitude-matched movements are not themselves involutions. The supported
statement is that the Householder-direction arm has the later mean collapse
onset in this sweep.

## Executed versus recovered-training input contract

The original repository reproduction and benchmark (`ontological_inversion.py` and
`operators.py`) execute this path:

`Nomic sentence embedding → leading 128 coordinates → L2 normalization → Wv+b`

The later recovered trainer specification at
`/home/ruffianl/ontological-inversion/training/SPEC.md` instead documents:

`concat(mu64, shape64) → Wv+b`

Here `shape64` is a PCA principal axis multiplied by isotropic spread and
anisotropy. Both are 128-dimensional inputs to the same affine form, but they are
different representations. The June behavioral claims therefore attach to the
executed leading-coordinate route. The recovered trainer source clarifies an
intended training contract but does not rewrite what the public benchmark ran.

The original adapter training corpus remains absent. A one-sentence smoke stub
cannot support byte-identical retraining, even though the shipped adapter itself
is preserved and hash-identical across the archive and current repository.

## 2026-08-27 re-verification and the rebuild workspace

Independent re-verification performed 27 August 2026 (MD5 re-computed on each
copy directly, not quoted from this manifest):

| artifact | MD5 | copies verified byte-identical |
|---|---|---|
| `adapter_final.safetensors` | `b8118021c7c27948565c4f322b5f6e04` | ghost_team archive · `/home/ruffianl/ontological-inversion` · `/home/ruffianl/oi_adapter_rebuild/original_source/` |
| `adapter_final_rebuild.safetensors` (executed PyTorch rebuild) | `543ac4e50e9d644314a99fc4777f187d` | differs from the shipped adapter, as expected — the compiled training corpus is absent |

Second independent archive located and enumerated:
`/run/media/ruffianl/backup2/05_archives/project-history/projects/ontological-inversion`
(7.9M — code, `results/` including `benchmark.csv`, `full_run.log`,
`topology_metrics.json`, `fold_decay_metrics.json`, and the June run cards; no
weights copy).

Recovered trainer (the reason "no trainer found" verdicts were wrong — the
trainer is Rust/candle, not Python): `src/bin/train_adapter.rs` in
`SplatRagBench-master`, with the full training stack preserved verbatim in
`/home/ruffianl/oi_adapter_rebuild/original_source/`:

| file | MD5 |
|---|---|
| `train_adapter.rs` | `b2230806c1522ea3ac82be5abca9a3bc` |
| `adapter.rs` | `4635a0c0c39c36bce9f27b83f48df1e8` |
| `config.rs` | `192d92f09db548eeb5cec8f7260fbd5a` |
| `embeddings.rs` | `4aadb7481442a17aea7572df83d6ee60` |
| `gaussian.rs` | `a724a725b475d2fa1688d79f4013fc3d` |
| `shaper.rs` | `48609cc1feac464ebe1f469e821a5b77` |
| `nomic_daemon.py` | `eba6f6a68be6a9d3460fafcf46dc8843` |

Tensor-naming fingerprint: candle `VarMap::save` under
`vb.pp("adapter")` → `vb.pp("linear")` emits exactly the shipped tensor names
`adapter.linear.{weight,bias}`. Training contract per
`/home/ruffianl/oi_adapter_rebuild/SPEC.md`: input `concat(mu64, variance64)`
(normalized 64-d Matryoshka cut of pooled Nomic embedding; top principal axis
scaled by `sigma_iso * anisotropy`), target = mean frozen Qwen2.5-0.5B input
token embeddings, MSE, AdamW lr 1e-3, batch 1, ≤20 epochs, id-sorted order.
