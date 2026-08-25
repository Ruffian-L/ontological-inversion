# Frozen memory-support selector pilot

This harness tests whether a preregistered rule can find a stable, content-specific,
low-hallucination reading region before seeing downstream answers. It uses
**memory-support score**: the metrics describe observable support supplied by an
injected memory, not an internal model state.

## Freeze order

1. Generate the probe manifest exactly once, before any gain outputs exist.
2. Review the generated calibration plan, then run it.
3. Add blinded semantic-judge scores with arm and gain hidden; calibrate and freeze
   thresholds and the judge model in `config.json`.
4. Commit the frozen config and lock hash. Only then will `runner.py` plan held-out
   cells.
5. Run held-out cells in fresh processes/contexts and score all attempts, including
   abstentions and controls.

```bash
python3 -m memory_support.protocol
python3 -m memory_support.runner --split calibration --out results/memory_support_calibration_plan.jsonl --dry-run
python3 -m unittest tests.test_memory_support_selector
```

The calibration plan contains 12 clusters × 8 probes × 165 arm/gain/seed cells =
15,840 generation cells (10,368 primary-selection and 5,472 audit-arm cells).
`hf_backend.py` is deliberately separate:
planning cannot accidentally start model inference. Review the plan, then run:

```bash
python3 -u -m memory_support.hf_backend results/memory_support_calibration_plan.jsonl \
  --out results/memory_support_calibration_responses.jsonl
```

Watch it without truncating the receipt:

```bash
tail -f results/memory_support_calibration_responses.jsonl
```

`score.py` supplies deterministic token/entity, unsupported-answer, and degeneration
components. Held-out selection is not valid until a blinded semantic judge, the
calibration-derived thresholds, interval/bootstrap unit (memory cluster), and
manual-audit sample are frozen. The selector ranks eligible plateaus by their
worst gate margin and abstains if no three adjacent gains pass.

`analyze.py` computes matched-minus-strongest-control specificity and interval
gates. `freeze.py` applies the preregistered calibration quantile rule and writes a
hashed threshold receipt. It does not silently unlock held-out execution: the
operator must review the receipt, freeze the judge identity, change config status,
and commit that boundary first. After review, `activate.py` records the frozen
selector hash, judge identity, and new config hash; held-out planning verifies all
three receipts.

`endpoints.py` bootstraps by whole held-out memory cluster and makes the primary
automatic-minus-frozen-global comparison with abstentions scored as zero utility.
It also reports unsupported answers, matched-control separation, abstention,
manual-heuristic, text-oracle, and explicitly labelled post-hoc oracle ceilings.

`judge.py prepare` emits opaque judge jobs containing source and response but no
arm, gain, seed, split, or cluster label. It also selects the manual-audit sample by
a deterministic hash before scores return. `judge.py merge` refuses a partial
fidelity judgment set.

The current build creates infrastructure and a dry-run plan only. It produces no
scientific result and makes no claim that the adapter can carry episodic memory.
