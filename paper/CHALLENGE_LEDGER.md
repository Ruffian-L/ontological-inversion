# Challenge ledger

This file records nulls, retractions, instrument failures, and open challenges.
Each entry is scoped to the intervention actually run. Nothing here deletes a
dated result in paper/DISCOVERY_LEDGER.md.

## C1 — Final post-normalization writes do not produce content-sensitive recall

Roughly 300 generations yield zero recalls or partials, and an unmatched control
shares the same digit artifact. This rejects that late additive write in the
tested Llama setup. It does not challenge input-slot reconstruction or layer-4
Qwen steering.

Receipt: research_logs/2026-08-24_synapse-recall-wrong-end-of-the-stack.md.

## C2 — Pooled sentence-to-ordered-slot mapping is structurally weak

The pooled map reaches centered held-out cosine 0.238 and unrelated generation.
Text-first fragmentation reaches 0.615 and span-aware training 0.647. The null
belongs to the pooled construction, not to linear cross-model transport as a
class.

Receipt: research_logs/2026-08-24_soft-slot-llama-spoke-the-buried-fact.md.

## C3 — vacated

The adapter-corpus contamination claim for Worb-glob is out. It was a token
grep, not a paper result. Do not restore it.

## C4 — E10 rejects full conjunctive recall for one clean-nonce protocol

The aligned Q8 ridge arm scores 0/9 at gains 0.75, 0.825, and 0.90. Reverse,
cross-memory, random, and dimension-shuffle arms score 0/9. Oracle scores 3/9 at
0.825. The “pisp that glows” output is a partial attribute and fails the frozen
penguin-plus-glow rule.

This does not overturn D1, D4–D9, or cross-model writing as a class. It shows
that this fixed bridge does not deliver full corpus-clean conjunctive recall on
these three memories.

Receipts: preregistrations/2026-08-26_E10_verified_novel_adapter_join.md;
synapse/probe/e10_summary.json, SHA-256 6d34b951…;
synapse/probe/e10_adapter.jsonl, SHA-256 5a63d29b….

## C5 — E2 is invalid for native source-width claims

The harness used target width 896, did not load Qwen3 or the intended adapter,
and made widths at or above 896 identical except seed. Its 17,760 rows are not
evidence for or against a native encoder-width boundary.

Receipts: synapse/probe/e2_raw.jsonl; harness inspection and correction log.

## C6 — Bounded knowledge is withdrawn

E7’s primary decline difference is +0.126 with 95% cluster-bootstrap interval
[+0.080,+0.177], below the frozen 0.25 criterion. Memory broadly reduces
declines for supported and unsupported questions. The remaining descriptive
claim is an answer-propensity shift, not calibrated confidence.

Receipts: preregistrations/2026-08-25_E7_base_rate.md;
synapse/probe/e7_raw.jsonl, SHA-256 77caf18d….

## C7 — Stable Betti-1 and output-anchor claims are retracted

The earlier Betti-1 estimate near seven varies from 0 to 56 under robustness
choices, and the output-space anchor probe is inconclusive. Per-layer cosine,
coherence, relative norm, and drift measurements remain.

Receipts: /home/ruffianl/ontological-inversion/results/HARDENED_AUDIT.md;
/home/ruffianl/ontological-inversion/runs/2026-06-25_anchor-detection.md.

## C8 — Concept specificity is not established for the steering direction

The unrelated adapted direction produces object readings in 4/5 matched cells.
Random and coordinate-shuffled directions produce 0/5, so a learned directional
effect remains. The challenge narrows specificity, not steering existence.

Receipt: /home/ruffianl/ontological-inversion/results/CONTROLS.md.

## C9 — E6 is preregistered and unrun

The name-frequency by redundancy factorial has no completed outcome. It supports
no manuscript result until the preregistered intervention is actually executed.

Receipt: preregistrations/2026-08-25_E6_name_frequency_redundancy.md.

## C10 — Original benchmark fractions are proxy-screen summaries

The 18/24 and 14/24 cells select the best noncollapsed point from five strengths
when one exists; cells with none remain failures. They use Nomic cosine against
handwritten anchors. The same encoder family
supplies the source representation and the evaluator. The numbers are valid
descriptive breadth evidence for the original screen, not held-out behavioral
success rates.

Receipt: archived results/benchmark.csv and benchmark.py at commit bd990416…;
paper/ORIGINAL_REPO_MANIFEST.md.

## C11 — Executed and recovered-training input contracts differ

The June repository reproduction and benchmark normalize the leading 128 coordinates
of a Nomic sentence embedding before applying the adapter. The later recovered
trainer specification instead describes concat(mu64, shape64). Both are 128-
dimensional and feed the same affine artifact, but they must not be collapsed.
The June behavior remains an executed result; its mechanistic description is
the leading-coordinate route, and byte-identical retraining is still blocked by
the missing original corpus.

Receipts: archived ontological_inversion.py and operators.py; current
/home/ruffianl/ontological-inversion/training/SPEC.md; manifest hash record.
