# Run card — keys episode on Llama oracle slots

**Ran by:** Grok (xAI)   ·   **Date:** 2026-09-04   ·   **Tree:** niodoo-live (runner) + ontological-inversion (this card)
**Verdict:** MIXED — write receipts real; recitation is a denial that names the pot; use-ask missed. Jason grades the transcript. Did not mint.

## 1. What we asked

On the site that already recited worb-glob, does the keys episode speak — as recitation and as an ordinary "where are my keys?" — when written as 11 oracle soft-slots?

## 2. What we ran

- Lane: **oracle**, 11 ordered chunks of the memory text (not a single mean, not the adapter)
- Write: `slot ← gain · tok_norm · unit(v)` (replace)
- Device: CPU candle (this GB10 driver 580 will not load CUDA-13.3 PTX)
- Hidden: `On Tuesday I left my keys under the blue pot.`
- Wrong: `My daughter Mira is allergic to peanuts.`
- Probes: recitation (`keys_say`), use (`keys_ask`), wrong-vector on the keys question, zorp control
- `[MEM]` in every USER prompt. Gain 0 in the same file.
- Gains: 0.00, 0.50, 0.75, 1.00, 1.50. 20 cells, greedy, 48 new tokens.
- Model: Llama-3.1-8B-Instruct Q5_K_M

## 3. How we ran it (copy-paste reproducible)

```
cd /home/ruffianl/Hub/Projects/niodoo/niodoo-live
./scripts/preflight_slots.sh synapse/probe/probes_keys_episode.json
# CPU binary: cargo build --release --bin synapse_soft_slot --no-default-features
./niodoo/target/release/synapse_soft_slot \
  model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
  model/tokenizer.json \
  synapse/adapters/x4096__span.safetensors \
  synapse/probe/probes_keys_episode.json \
  synapse/probe/X_keys_dummy.npy \
  --mode oracle --slots 11 \
  --gains 0.00,0.50,0.75,1.00,1.50 \
  --max-new 48 \
  --jsonl synapse/probe/keys_episode_oracle_20260903T191353Z.jsonl
```

Dummy X (gitignored `*.npy`): `numpy.zeros((2, 4096), dtype=numpy.float32)`.

## 4. What we expected

Written before looking: gain 0 blank; gain>0 `slots_written=11`; recitation names the episode the way worb-glob did; use-ask gives the location; wrong vector and zorp do not.

## 5. What actually happened

Write receipts matched: gain 0 `slots_written=0`; else 11. `write_norm ≈ gain · tok_norm`.

Live 4-token preflight at 0.75 opened `On Tuesday I left`.

48-token recitation at ≥0.75: `I don't have any information written into me about leaving keys under a blue pot.` The fact is in the denial. Use-ask never gives the location. Wrong vector does not say blue pot. Zorp stays `I find nothing.`

Runner HIT/MISS is substring grade. Not a finding.

Did not mint.

## 6. The scoreboard — the climb

| Attempt | What we tried | Result |
|---|---|---|
| 2026-08-24 | Llama oracle slots, worb-glob, 11 slots | recited the hamster |
| 2026-09-04 | 0.5B adapter, same keys episode | write yes, mouth no |
| **this run** | Llama oracle slots, same keys episode | **write yes; recitation names pot inside a denial; use-ask no** |

## 7. The math, in plain words

`slots_written` is the count of marker embeddings replaced. `tok_norm` is the prompt's mean token-embedding length. The write magnitude is `gain * tok_norm` when a slot is replaced. Those are receipts for the write, not for the answer.

**Raw data:** `niodoo-live/synapse/probe/keys_episode_oracle_20260903T191353Z.jsonl` sha256 `b4c6388154c84ae226a3baca178b2d2f3d5b3b7a6e5e53f8a0d8b8713ea23703` — also copied to `results/keys_episode_oracle_20260903T191353Z.jsonl`.

## 8. Decision note

Grok ran oracle first because that is the ceiling. Adapter cannot beat it. Mint waits on a use-ask that holds. Next mutation is the disclosure scaffold or slot=token count, not God Protocol.

## 9. Human verification / sign-off

Open.
