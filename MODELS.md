# Models — trust the bytes

The HF landscape carries poisoned uploads (incl. from well-known names), so this file records the
**exact, verified** artifacts every result was produced with, and how to run touching only audited
bytes (STANDARDS.md §4).

## Regime A pinned artifacts (Nomic to Qwen2.5-0.5B) — all verified == official HF `main` on 2026-06-25

| Role | Source (official org) | Pinned revision | Format / exec |
|---|---|---|---|
| Generator (default) | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | safetensors, **no code** |
| Generator (original) | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | safetensors, **no code** |
| Embedder weights | `nomic-ai/nomic-embed-text-v1.5` | `e9b6763023c676ca8431644204f50c2b100d9aab` | safetensors |
| Embedder **code** (`trust_remote_code`) | `nomic-ai/nomic-bert-2048` | `7710840340a098cfb869c4f65e87cf2b1b70caca` | the executed `.py` |
| Trained adapter ("Synapse") | `adapter_final.safetensors` (in-repo) | committed bytes | safetensors |

Pinned revisions are wired into the scripts via `modelpin.py` (`revision=rev(id)`).

## Regime B pinned artifacts (Qwen3 to Llama ordered slots)

Regime A and Regime B share nothing but the house rules. These are the endpoints
every E10 and span result was produced with, frozen in the preregistration
(`regime_b/preregistration/`, dated 2026-08-26, before any output existed).

| Role | Artifact | Hash / revision | Format / exec |
|---|---|---|---|
| Generator | `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf` | sha256 `14e10feba0c82a55…` (5,732,992,416 bytes) | GGUF, **no code** |
| Generator tokenizer | `tokenizer.json` | ships with the GGUF checkout | data-only |
| Retriever | `Qwen3-Embedding-8B-Q8_0.gguf`, 4096-d | sha256 `d20ddc71e8a5c434…` | GGUF, served locally |
| Retriever endpoint | `http://127.0.0.1:8081/v1/embeddings` | local only, never remote | OpenAI-shaped |
| Trained bridge | `regime_b/adapters/x4096__span.safetensors` | sha256 `f689bbf0a919bfd4…` (65 MB, in-repo) | safetensors |
| Frozen scorer | `regime_b/scorer/synapse_score_e10.py` | sha256 `c1e782900a43761a…` | evaluator-only rubric |
| Injection runner | `regime_b/runner/synapse_soft_slot.rs` | sha256 `d9a0d369287d1f57…` | Rust on `candle` |

**Not Q4.** `OPEN_ITEMS.md` notes a `Qwen3-Embedding-8B-Q4_K_M.gguf` is also on
disk and that no matched Q4 replication is claimed. Every published Regime-B
number used **Q8_0**. Do not substitute.

### Bridge construction

Closed-form ridge, no gradient descent, no target-model loss:

| Parameter | Value | Source |
|---|---|---|
| regularizer `lam` | `1.0` | `regime_b/bridge/synapse_train_span.py` |
| split seed | `1729` | same |
| split | 90/10, honoring text-group boundaries so spans of one text cannot straddle | same |
| sources | `chunks,spans` — 32,000 chunks + 32,000 spans | preregistration |
| bias | train-set mean of Y (not fit) | same |
| solve | `torch.linalg.solve` in float64, cast to float32 | same |

### Decode contract for E10

Greedy, at most 32 new tokens, one decode per cell, 14 ordered memory slots,
injection at the token-embedding input before block 0. Gains 0.750 / 0.825 /
0.900, frozen before the run, all reported.

## Why each is low-risk
- **Qwen** loads as `model.safetensors` — data-only, no pickle, **no code execution**. Worst case is
  behaviorally-poisoned *weights*, not RCE.
- **nomic-embed-v1.5** weights are safetensors; its `auto_map` pulls modeling code from
  `nomic-ai/nomic-bert-2048`. That `.py` (`modeling_hf_nomic_bert.py`, 2556 lines) was **statically
  audited** at the pinned commit: no `subprocess/os.system/eval/exec/socket/urllib/requests/pickle`,
  no network calls — standard BERT-variant architecture; imports are torch/numpy/einops/transformers/
  safetensors only. (Static review of *these commits*; strong, not a formal proof.)

## Safe-run mode (touch only audited local bytes)
After the models are cached (first run), pin to offline so a poisoned upstream re-upload can never be
fetched mid-run:
```bash
export HF_HUB_OFFLINE=1     # use ONLY the local, audited cache — no network fetches
export TRANSFORMERS_OFFLINE=1
python ontological_inversion.py
```
To re-verify the cache matches official before trusting it again:
```bash
for id in Qwen/Qwen2.5-0.5B-Instruct nomic-ai/nomic-embed-text-v1.5 nomic-ai/nomic-bert-2048; do
  echo "$id $(curl -s https://huggingface.co/api/models/$id | python3 -c 'import sys,json;print(json.load(sys.stdin)["sha"])')"
done   # compare against the SHAs above
```

## Honest caveats
- Pinning gives **reproducibility + protection against re-uploads**; it does **not** prove the pinned
  commit was never compromised (only the static audit + official-match speak to that).
- `code_revision` for `trust_remote_code` isn't cleanly forwarded by SentenceTransformer; the real guard
  for the executed code is the **audited cache + offline mode** above.
- Decode is greedy (temp 0), so given pinned weights the runs are deterministic.
