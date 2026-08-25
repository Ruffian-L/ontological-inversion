# Run card — why 128-d was the right size, and what the embedding layer does

**Ran by:** Claude (Opus 5) with Jason   ·   **Date:** 2026-08-25   ·   **Tree:** niodoo-live (companion), ontological-inversion @ working
**Verdict:** PASS (companion result)   ·   No steering claim in this repo changes.

## 1. What we asked

Two questions this repo has an interest in, answered in a companion line of work
on Llama-3.1-8B:

1. Is the Synapse's 128-d input undersized, or is it the right size for what it
   does?
2. The training target is the token-embedding layer (`training/SPEC.md` §3) while
   this repo injects at the layer-4 residual. Does anything happen if you inject
   where the adapter was actually fit?

## 2. What we ran

The companion work writes a concept's direction into **soft token slots at the
embedding layer** of Llama-3.1-8B — the site `SPEC.md` names as the training
target — and measures whether the content comes back out in generation. Full
detail in `niodoo-live/research_logs/2026-08-24_soft-slot-llama-spoke-the-buried-fact.md`
and `2026-08-25_reviewer2-nearest-neighbour-and-contamination.md`.

## 3. What we expected

That 128-d would prove too small and the original Synapse would look
under-provisioned.

## 4. What actually happened

**It is not undersized. It is sized for a different job.** Truncating the same
encoder's vectors to leading Matryoshka slices — same encoder, same data, same
site, so width is the only variable — gives a clean monotone curve with **no knee
at the model's native width**. Width is a plain capacity axis; there is no
"native shape" effect.

But the two things a vector can carry come apart:

> **128 dimensions preserve 94% of the pairwise similarity structure and 50% of
> the token-level reconstruction signal.**

(Pearson r of pairwise similarity against the full-width vectors: 0.936 at 128-d,
0.836 even at 32-d.)

Semantic similarity — *is this concept about fire?* — survives truncation almost
perfectly. Token-level reconstruction — *which exact word* — does not.

**This is why 128-d is right for this repo.** Ontological inversion is a
similarity-scale operation: push the residual along a concept axis, sweep a
signed gain, watch a living thing become an object that keeps its function. It
needs the concept's *direction*, not its *spelling*. The 462 KB adapter is
correctly sized for that. Asking the same vector to reproduce exact wording is a
different budget, and that is the one that needs width.

**On the injection site.** Writing at the final post-norm — one matmul from the
logits — produces gain-scaled disturbance with **no content**: the model reports
noticing something, and an *unmatched control leaks the identical artifact*.
Nothing downstream is left to read a vector there. At the embedding layer, with
32 blocks downstream, content comes back. Layer 4 sits between the two.

That is a negative result worth having next to §3's flagged mismatch: it does not
say layer 4 is wrong — layer 4 reproduces here and survives the controls — but it
shows the site is not a free parameter, and it gives the shape of the failure
when you get it wrong.

## 5. The scoreboard — the climb

| Attempt | What we tried | Result |
|---|---|---|
| 2026-06-24 | recover the trained adapter, inject at layer-4 residual | flip reproduced, α≈0.2 |
| 2026-08-24 | find how that adapter was built | `train_adapter.rs`; target is the token-embedding layer, not layer 4 |
| 2026-08-24 | inject at the final post-norm | **content-free**: gain-scaled disturbance, control leaks the same artifact |
| 2026-08-24 | inject at the token-embedding layer | content returns; capacity governed by tokens-per-slot |
| **2026-08-25** | **truncate input width, same encoder, same site** | **no knee at native width; 128-d keeps 94% of similarity and 50% of reconstruction** |

> Read this as the climb: the adapter was never too small, it was sized for
> steering — and the site it was fit to turns out to do something the site we
> inject at does not.

## 6. What this does not claim

- **Nothing about the inversion result changes.** No run re-scored, no band
  moved, no metric re-derived. This is a companion measurement on a different
  model in a different repo.
- **The width curve is one encoder's truncation curve**, not a universal one. A
  different encoder at equal width is untested — including this repo's own
  `concat(mu₆₄, shape₆₄)`, whose shape half a pooled embedding does not carry.
- **The companion line has its own open items** — norm-matched random-direction
  controls, decoding variance, blind grading, n>3 concepts — enumerated in its
  research log and not claimed here.
- One contaminated probe string was found in the companion corpus and its
  affected results are quarantined there. It does not touch this repo.
