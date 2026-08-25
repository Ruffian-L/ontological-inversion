# Run card — frozen memory-support harness build

**Ran by:** Codex   ·   **Date:** 2026-08-25   ·   **Tree:** ontological-inversion, pre-commit worktree
**Verdict:** PASS (infrastructure only; no model result)

## 1. What we asked
Can the proposed memory-support experiment be made into an immutable calibration/held-out test rather than a best-output scan?

## 2. What we ran
Generated 24 synthetic, entity-disjoint clusters (12/12 split), two fixed prompt forms for four probe families, and a calibration plan over nine gains, three seeds, four primary arms, and three audit arms. Ran four deterministic unit tests.

## 3. How we ran it (copy-paste reproducible)
```
python3 -m memory_support.protocol
python3 -m memory_support.runner --split calibration --out results/memory_support_calibration_plan.jsonl --dry-run
python3 -m unittest tests.test_memory_support_selector
```

## 4. What we expected
The manifest would contain 192 probes; held-out planning would remain locked; the calibration plan would enumerate every precommitted cell; selector tests would pass.

## 5. What actually happened
The manifest contains 192 probes with SHA-256 `7d407d178f82dddeecf47ec84ad8f33d6e1f02b839b58338bcda9cdc5243b327`. The dry plan contains 15,840 cells. Held-out requires `config.status=frozen`. Five unit tests pass. No model was loaded and no output was scored.

## 6. The scoreboard — the climb
| Attempt | What we tried | Result |
|---|---|---|
| prior gain island | selected the best observed surface output | post-hoc and probe-loaded; not a held-out selector |
| **this build** | immutable probes → calibration → frozen gates → held-out plateau selection | **harness PASS; scientific test not run** |

> This rung proves the test is enumerable and locked, not that residual memory is readable.

## 7. The math, in plain words
12 calibration clusters × 8 probes × 165 arm/gain/seed combinations = 15,840 planned outputs. A gain is eligible only inside at least three adjacent points clearing all six gates; plateau rank uses its weakest margin.

The frozen global gain is the calibration gain with the best mean worst-gate
margin. Held-out automatic-minus-global utility is bootstrapped by whole memory
cluster, and an abstention contributes zero utility and zero unsupported answers.

**Raw data:** `memory_support/locked_probes.jsonl`, `results/memory_support_calibration_plan.jsonl`.

## 8. Decision note (provenance)
**Decided by / on:** Jason specified the frozen selector; Codex implemented it on 2026-08-25. The coarse nine-gain grid is deliberately not densified on test memories, overriding the repo's inversion-island dense-sweep rule for this separate preregistered memory-selection question.

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before any model run (none occurred)
- [ ] I reviewed the calibration plan and judge choice — _initials / date_
- [ ] I re-ran the model campaign — _initials / date_
