# Changelog

This is a research repo. Not production unless Jason says so.

Pairing: every action here gets a **why**. Hypothesis form:

- We made this change. We think X will happen.
- Later: X did not happen, yet we found Y. Next we mutate Z.
- We mutated Z. Results matched. LFG.

Keep this file short. Longer writeups go in [`research_logs/`](research_logs/)
(one subject, date + title) and structured per-experiment evidence goes in
[`runs/`](runs/) as run cards. The climb is [`SCOREBOARD.md`](SCOREBOARD.md);
the narrative is [`RESEARCH_LOG.md`](RESEARCH_LOG.md). Agent contract:
[`AGENTS.md`](AGENTS.md).

> Entries dated before 2026-08-24 were **reconstructed on 2026-08-25** from
> `SCOREBOARD.md` and `RESEARCH_LOG.md`, because this file did not exist until
> then. They are accurate to those sources but were not written contemporaneously.
> Anything from 2026-08-24 onward was written in the turn the work happened.

## 2026-08-25 — memory-support calibration launched on GB10

We did: created ignored `.venv/` with Python 3.12 and the repository's 69 pinned
packages after bare Python 3.14 failed on missing NumPy. Mutated `hf_backend.py`
to select CUDA/bfloat16, pass an explicit attention mask, cache directions, and
resume from append-only receipts. Two disclosed one-cell gain-zero smokes completed;
both correctly provided no source-memory readback and instead produced the same
unsupported code-memory scenario. Launched all 15,840 calibration cells as PID
126493; first receipt landed. Did not inspect held-out, freeze gates, judge outputs,
merge results, train, or claim readability.

Added `memory_support/watch.py` after the raw progress log proved unreadable to a
human; it streams the full memory, probe, arm, gain, seed, and model generation
from the append-only receipts without touching the running process.

We think: the environment and receipt path now hold; whether the injected direction
supports memory remains open. Gain-zero failure is an expected negative control.

Next: let calibration finish, then blind the fidelity judge jobs before scoring.

## 2026-08-25 — frozen memory-support selector harness built

We did: added `memory_support/`, a 24-cluster (12 calibration / 12 held-out)
preregistered pilot with 192 probes, nine gains, three seeds, matched/wrong/random/
blank controls, audit arms, blinded-judge slots, specificity/safety intervals, and
worst-margin three-point plateau selection. Added cluster-bootstrap endpoints for
automatic selection versus the calibration-frozen global gain, including
abstentions and labelled oracle baselines. Locked the probe manifest before any
gain outputs and dry-planned 15,840 calibration cells. Five unit tests pass. Did
not load a model, generate responses, freeze thresholds, inspect held-out plans,
or claim memory readability.

We think: separating immutable planning, generation, scoring, calibration freeze,
and held-out unlock makes post-hoc best-output selection mechanically difficult.

Next: review the calibration plan and judge choice, then run calibration only.

## 2026-08-25 (E8) — the concept vector cannot invert on its own

We did: ran the complement of the bias-only result. `d = Wv` with the bias removed
(the falsifier), plus `d = b + λWv` for λ ∈ {0.25, 0.5, 1, 2, 4}, plus a swap using
an unrelated concept's residual. Eight arms, dense 0.01 grid, 80 tokens. nomic
loaded offline through plain transformers; nothing installed.

**Residual alone never inverts.** Zero object-words at *every* gain from −0.10 to
−0.26 — it says "great choice for a pet" throughout, then collapses into repetition
without ever passing through an inversion. Not a magnitude artifact: ‖Wv‖ = 1.2989
against ‖b‖ = 1.1118, so the residual is the *larger* vector and the injection
normalises both.

λ recombination cancels monotonically — inverting gains go 17 → 7 → 3 → 2 → 0 as
the concept share rises. Did not re-measure the Fire Invariant asymmetry; one
concept pair; n=1 per α.

We think: **the bias is the inversion axis, and the concept-specific half cannot
invert at all on its own.** The cancellation follows from the geometry —
cos(b, Wv) = −0.6056, so the residual points *against* the bias and progressively
destroys the axis. The reading from earlier today survives in its strong form.

Two results we did not predict. Small λ (0.25–0.50) **widens** the band rather than
narrowing it, covering the junk gap at −0.23…−0.20 where bias-only degenerates — a
little concept stabilises the effect before more of it kills it. And **swap works**:
bias plus *wolf's* residual inverts like the matched concept, because
cos(Wv_glubtub, Wv_wolf) = 0.6781. That is the mechanism behind `CONTROLS.md`'s
`unrelated_adapter` = 80%, which has sat unexplained since 2026-08-05.

Also: GPU/CPU greedy determinism confirmed byte-for-byte (`MODELS.md` leans on it),
and a CPU-resident direction tensor in the hook cost ~150× — dtype-only `.to()`
does not move devices.

Next: E2, the native-width straddle across the four local models whose hidden size
the encoder can cross from both sides.

## 2026-08-25 — the bias carries the inversion; the concept carries the invariant

We did: ran the limit case behind `unrelated_adapter`=80%. Set `v = 0` so the
injected direction is the adapter's **bias alone** — no concept text embedded
anywhere in the run. Dense 0.01 grid per the `WINDOW_MAP.md` method rule, 80 new
tokens (a 50-token cap had hidden the readout phrase). Also measured the adapter
geometry directly, and wrote `AGENTS.md`, which this repo did not have.

Bias alone inverts: "a small, portable **stove that heats water**" at
α ∈ [−0.13, −0.12] — the same sentence the concept direction gives at
[−0.21, −0.18] — and clean inanimate "**container made of plastic or rubber**"
(L=0) at α ∈ [−0.28, −0.24], with a junk gap at [−0.23, −0.20]. Baseline
reproduces the README ("a great choice for a pet"). Did not run the complement
(`Wv` alone). n=1 per α, one prompt, greedy.

We think: **the bias carries alive→object; the concept carries which object, and
the Fire Invariant.** The concept's lobes reach L=0 while still naming *fire pit*;
the bias reaches L=0 only much deeper and lands on water bottles and plastic. The
weights agree — at the unit input the callers feed, ‖Wv‖ ≈ 1.08 against
‖b‖ = 1.11, so any two concepts' directions share ≈0.52 cosine, which is why an
unrelated concept works. B1's prompt-frame reading is not sufficient alone: at
α = 0 the same prompt is still living.

Also: the two-lobes-with-a-gap shape appears in a direction containing no concept
at all, so **the island shape is a property of the stack, not of the concept
encoding.**

Next: `d = Wv` with the bias removed, and `d = b + λWv` recombination. If `Wv`
alone inverts, this reading is wrong and gets withdrawn.

## 2026-08-25 — retrieval-grade compression is not recall-grade

We did: companion measurement on Llama-3.1-8B. Truncated the same encoder's
vectors to leading Matryoshka slices — encoder, data and site all held fixed, so
width is the only variable — after validating the cut (pairwise-similarity
correlation against full width: r=0.936 at 128-d, 0.836 at 32-d).

We think: **there is no nativeness effect.** The curve is monotone with no knee at
the model's native width, so width is a plain capacity axis. And the two things a
vector carries come apart: **128 dimensions preserve 94% of pairwise similarity
structure and 50% of token-level reconstruction.** That is why 462 KB is the right
size for this repo — inversion is a similarity-scale operation needing a concept's
direction, not its spelling. Only recitation needs width.

Second finding, against `training/SPEC.md` §3's flagged mismatch: injecting at the
final post-norm is content-free — gain-scaled disturbance, and an unmatched control
leaks the identical artifact. At the token-embedding layer, the site the adapter was
actually fit to, content returns. Layer 4 sits between. Layer 4 is not wrong — it
reproduces and survives the controls — but the site is not a free parameter.

Next: the causal decomposition (above), and rank curves for steering vs
reconstruction to test whether low-rank similarity predicts steering ability.

## 2026-08-24 — the Synapse trainer, recovered; README rewritten

We did: found the build script for `adapter_final.safetensors`, long assumed lost.
It is `src/bin/train_adapter.rs` in `SplatRagBench-master`, **Rust on `candle`** —
every prior search had been shaped like Python and returned nothing. Added it plus
every source file it depends on and a verified rebuild spec under `training/`. The
shipped adapter hashes identically to the original tree's
(`b8118021c7c27948565c4f322b5f6e04`). Rewrote `README.md`, which opened on
"compile a fact into one residual direction" and never said the sentence that
decides whether any of it lands — the weights are frozen and the concept is never
in the prompt.

Two corrections landed with it. The input is `concat(mu₆₄, shape₆₄)`, not a "128-d
nomic Matryoshka concept code" — half an L2-normalised 64-d cut, half a PCA
principal axis scaled by `sigma_iso × anisotropy` that can exceed it by two orders
of magnitude. And the training **target is the token-embedding layer, mean-pooled**,
not layer 4. `PROVENANCE.md` corrected in place and dated, old wording quoted
rather than erased.

We think: the artifact worked before anyone knew how it was made, and the recipe is
now on the shelf next to it. Byte-identical reproduction is impossible — the
training corpus is gone (`manifest.bin` is a 71-byte stub reading *"The sky is neon
green."*) — but a functionally equivalent adapter is fully reproducible.

Next: state the layer-4 mismatch openly rather than implying one derivation, and
measure whether the site choice matters.

## 2026-08-05 — the 75% was metric inflation; framing pivot *(reconstructed)*

We did: Grok stress-tested the 360-run benchmark with an independent keyword
structured-flip check. Ran an ultra-fine gain map at Δα=0.01. Ran the
random/shuffled/unrelated controls. Added `anti_fact.py`, `subjects.json`,
`controls.py`, `PAPER_FRAMING.md`, `MEMORY_STEERING.md`.

We think: the soft 75% flip rate was **proxy poison** — of runs with
`inv_gain > 0.1`, 48% are collapsed gibberish the embedding metric still rewards.
Keyword structured-flip drops cell rates to ~8–12%, concentrated almost entirely on
Glub-Tub × Qwen2.5-0.5B-Instruct × α≈0.2. Retracted as stated; reframed as
*directional proxy shift*, with Glub-Tub the clean existence proof.

The real finding from the dense map: the island is **two thin non-contiguous lobes
with a living gap** — stove at α ∈ [−0.21, −0.18], fire-pit at [−0.15, −0.14],
living reading at [−0.17, −0.16]. Non-monotone. Coarse sweeps miss or mis-size it.
Controls: concept 100% honest flip, random 0%, shuffled 0%, **unrelated 80%**.

Next: run the anti-fact headline (C0), which is still pending.

## 2026-06-25 — Betti-1 retracted; anchor probe failed *(reconstructed)*

We did: per-layer fold decay and a persistent-homology robustness battery
(bootstrap / pooling / leave-one-out). Ran the anchor-detection probe.

We think: imposed ± symmetry **dies by ~layer 6** — `cos(Δ⁺,Δ⁻)` goes −1.00 (forced
at 4) → −0.58 (5) → −0.16 (6), then flat — while steering-axis coherence persists
near 0.8 down the whole stack. **The anchor survives; the mirror doesn't**, so a
true-mirror involution loop only holds ~1 layer past injection and the Phase-3
window is layers 4–5. The Betti-1 ≈ 7 loop count **did not survive** robustness
(swung 0–56) and is retracted. The anchor probe failed outright: output-axis
susceptibility was flat and anchored prompts inverted *less* than bare ones.

Next: move the anchor probe to hidden space, where fold-decay showed the durable
structure lives.

## 2026-06-24 — adapter recovered, flip reproduced *(reconstructed)*

We did: tried ~11 from-scratch reconstructions of the antipode — embedding
arithmetic, transformer hidden-state steering on Llama-8B and Qwen-0.5B, PCA
physics, charge/Coulomb. All failed; all drifted or collapsed. Then recovered the
trained adapter (`adapter_final.safetensors`, 128→896) and injected its direction
into the Qwen-0.5B residual at layer 4. Ran a 360-run operator benchmark and two
topology passes.

We think: **the flip rides the trained direction.** magma-hamster → "stove that
heats water", sweet spot α≈0.2. Topology v1 was an artifact — capturing at the
injection layer gives a trivial straight line by construction; v2, on the
propagated final hidden state, gives a curved trajectory (bendiness 2.7) but **no
clean Möbius fold** at the output (fold ≈ −0.13, not −1).

Next: harden the metrics, and map the gain band densely.
