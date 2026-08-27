# Frozen memory-support selector

The old gain-island workflow can answer “where did one interesting generation
appear?” It cannot by itself answer the harder prospective question: can a frozen
rule identify a safe reading region for an unseen memory before downstream answers
are visible? This build makes those different questions mechanically distinct.

The corpus has 24 whole memory clusters, split 12/12 with different synthetic
entities and source chunks. Every cluster has two fixed forms of reconstruction,
entailed, deliberately unsupported, and contextual-use probes. Every prompt tells
the model it is being evaluated on reading an injected memory. The manifest is
hashed before gain output exists.

Primary arms are matched memory, wrong memory, norm-matched random direction, and
blank injection. Coordinate permutation, reversed slot order, and text-in-prompt
oracle are audit arms. The frozen global gain, manual heuristic, and post-hoc oracle
are analysis baselines rather than hidden selection inputs.

Selection is gate-first. Fidelity, utility, specificity, unsupported answering,
degeneration, and seed/prompt consistency retain separate intervals. Specificity
is the matched lower bound minus the strongest control upper bound. At least three
adjacent gains must pass. Plateaus are ranked by their worst single gate margin;
the midpoint is chosen and ties favor the smallest absolute gain. No qualifying
plateau means abstention.

Calibration also freezes one global gain by mean worst-gate margin. Held-out
reporting compares automatic selection against that global point first, counting
abstentions, and bootstraps whole clusters. Manual gain, text oracle, and post-hoc
best gain remain separately labelled baselines rather than inputs to selection.

The current `config.json` is intentionally `calibration_required`. The held-out
planner refuses to enumerate held-out cells until calibration thresholds, semantic
judge identity, judge prompt, and audit boundary are reviewed and frozen. This is
not security against a malicious operator; it is a visible procedural tripwire
against accidentally looking.

What did not happen matters: no model loaded, no responses generated, no judge was
chosen, no thresholds frozen, and no claim was made that the recovered Synapse can
carry episodic memory. The next mutation is calibration-only execution after Jason
reviews the plan and compute size.
