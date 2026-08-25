# Run card — memory-support calibration start

**Ran by:** Codex   ·   **Date:** 2026-08-25   ·   **Tree:** ontological-inversion @ c308640 + backend patch
**Verdict:** MIXED — startup/smoke PASS; 15,840-cell calibration RUNNING

## 1. What we asked
Can the frozen calibration plan execute reproducibly on the local GB10 and write resumable receipts?

## 2. What we ran
Two one-cell disclosed smokes at matched arm, gain 0, seed 104729, then the complete locked calibration plan. Every model prompt says this is an evaluation of reading an injected memory and rewards correct abstention.

## 3. How we ran it (copy-paste reproducible)
```
uv venv --python /home/ruffianl/.local/bin/python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -u -m memory_support.hf_backend results/memory_support_calibration_plan.jsonl --out results/memory_support_calibration_responses.jsonl
```

Live log:
```
.venv/bin/python -u -m memory_support.watch results/memory_support_calibration_responses.jsonl --last 3
```

## 4. What we expected
The gain-zero reconstruction cell should not recover “Ilya Venn maintains the copper greenhouse in Oris Bay.” The backend should load pinned Qwen on CUDA, emit a cell receipt, and remain alive for the full plan.

## 5. What actually happened
Both gain-zero smokes omitted the correct source and produced the same unrelated scenario beginning “Inject 4 bytes at address 0x12345678…”. That is a failed memory read and expected negative control, not a selector failure. After the attention-mask correction, Qwen loaded on CUDA/bfloat16; PID 126493 emitted its first append-only receipt and continued running.

## 6. The scoreboard — the climb
| Attempt | What we tried | Result |
|---|---|---|
| bare launcher | `/usr/bin/python3` (3.14) | failed before model load: NumPy absent |
| smoke 1 | CUDA backend, one gain-zero cell | receipt PASS; attention-mask warning |
| smoke 2 | explicit attention mask | receipt PASS; warning removed |
| **calibration** | all 15,840 locked cells | **RUNNING; no result yet** |

> Infrastructure passing does not imply that any gain reads the memory.

## 7. The math, in plain words
Two of two gain-zero smokes failed source reconstruction, as a blank-effective cell should. The full denominator remains 15,840; one receipt at launch is progress, not an endpoint.

**Raw data:** `results/memory_support_smoke_responses.jsonl`, `results/memory_support_smoke_attention_responses.jsonl`, `results/memory_support_calibration_responses.jsonl`.

## 8. Decision note (provenance)
**Decided by / on:** Jason requested execution on 2026-08-25. Codex chose the existing Python 3.12 + `uv` route because the invoked system Python was 3.14 without dependencies. CUDA/bfloat16 uses the available GB10; seeds, prompts, gains, and model revision did not change.

## 9. Human verification / sign-off
- [x] Evaluation disclosure is present in every locked prompt
- [x] Correct gain-zero expectation is printed beside the smoke output
- [ ] Full calibration completed and receipt count is 15,840
- [ ] Numbers checked after blinded judging
