# Episode loop: write landed, mouth missed

> Date: 2026-09-04
> Agent: Grok (xAI)
> Repo: ontological-inversion

## Context

Jason: finish the project, make the full loop, MATS in two days.
Vocab is ontological inversion / memory steering, not Shared Ocean as
the museum. Ambient width matching already died (no knee at native dim).

## What changed

`episode_loop.py` — first rung-5 cell. Hidden episode, visible ask,
gain 0 in-file, matched vs wrong-memory, greedy 0.5B, layer 4.

## Hypothesis

We thought the inversion adapter might carry a short personal episode
the way it carries Glub-Tub polarity.

## Findings

It did not, on this stack. Write receipts scaled. No "blue pot" in the
transcript (post-hoc parse; Jason grades). 15 cells,
`results/episode_loop_20260903T180048Z.jsonl`.

Did not: Llama slots, Qwen3→Llama bridge, 9720-cell calibration, claim
the agency north star, auto-verdict.

Side foot: a SplatRAG `remember` preflight ping appended one junk row
to personal cold (128211→128212). Left it. Do not ping cold again.

## Next

Llama input-slot write of the same keys episode. That is the regime
that already spoke a buried fact (2026-08-24). Not more 0.5B gain.
