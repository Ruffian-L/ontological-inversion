# Models — trust the bytes

The HF landscape carries poisoned uploads (incl. from well-known names), so this file records the
**exact, verified** artifacts every result was produced with, and how to run touching only audited
bytes (STANDARDS.md §4).

## Pinned artifacts (all verified == official HF `main` on 2026-06-25)

| Role | Source (official org) | Pinned revision | Format / exec |
|---|---|---|---|
| Generator (default) | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | safetensors, **no code** |
| Generator (original) | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | safetensors, **no code** |
| Embedder weights | `nomic-ai/nomic-embed-text-v1.5` | `e9b6763023c676ca8431644204f50c2b100d9aab` | safetensors |
| Embedder **code** (`trust_remote_code`) | `nomic-ai/nomic-bert-2048` | `7710840340a098cfb869c4f65e87cf2b1b70caca` | the executed `.py` |
| Trained adapter ("Synapse") | `adapter_final.safetensors` (in-repo) | committed bytes | safetensors |

Pinned revisions are wired into the scripts via `modelpin.py` (`revision=rev(id)`).

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
