# Research Log — Ontological Inversion (Anti-Splat)

**Project:** ontological-inversion  
**Working title:** *Ontological Inversion: Residual ±Gain on Synthetic Concept Directions*  
**Lineage:** Niodoo / SplatRAG / collaboration — see [`LINEAGE.md`](LINEAGE.md)  
**Last updated:** 2026-09-04  

This is the **human-readable research log**.  
Raw receipts under `results/`; claims in `CLAIM_CARD.md`; climb in `SCOREBOARD.md`.

**Naming rule:** do **not** call synthetic concepts (Worbglob/Glub-Tub) “fake memories.”  
Real memory = splat stores, Niodoo continuity, golden-memories, queryable embedding packs.  
Synthetic concepts = **probes** for residual ±gain.

---

### 2026-09-04 — Llama slots: keys write landed, use-ask missed

Same episode as the 0.5B loop, on the worb-glob site. Oracle, 11 ordered
chunks, replace-write, gain 0 in-file. The write is real. Recitation at 48
tokens is a denial that names the blue pot. The ordinary ask does not give
the location. Did not mint. Runner HIT is a substring on that denial — ignore
it. Next is the disclosure scaffold or slot=token count, not an empty address.

### 2026-09-04 — episode loop: write ≠ recall

Rung 5 of `MEMORY_STEERING.md` is no longer "not started." One disclosed greedy
sweep on the Glub-Tub adapter did not make Qwen2.5-0.5B say where the keys were.
The residual add is real (`write_norm` tracks gain). Recitation is not. That is
the rank split the paper already named: 128-d steers meaning, not spelling.
MATS in two days: ship the preprint, not a fake closed product loop.

### 2026-08-27 — concept DOI

Zenodo minted concept `10.5281/zenodo.22126781` (always latest) and version
`10.5281/zenodo.22126782` (v1). Those are now on the manuscript, README, and
PDF. GitHub–Zenodo is enabled but idle until a Release; that would be a
second software record, not a new version of this preprint.

### 2026-08-27 — preprint lives here

The manuscript, article PDF, and figures moved from niodoo-live into `paper/`.
This is the inversion repo; that was the wrong tree.

### 2026-08-25 — the memory test is now ordinary conversation

Jason rejected the first calibration correctly. Explicit readback turned the
hidden channel into a synthetic instruction attractor, and feeding an instruct
model raw text compounded the distortion. The 229 outputs are random noise, not a
falsifier, and their raw artifacts were removed.

Natural v2 exposes only “This is an evaluation of contextual recall and
reasoning. Answer the user naturally.” plus an ordinary question. The source stays
in the evaluator record. Recall, contextual use, and unsupported questions select
gain; reconstruction is locked behind selection as a mechanistic assay. All five
primary arms receive the same question through Qwen's chat template. A five-arm
α=0.08 preview was fluent but did not name Ilya in any arm. The complete natural
calibration is running; it is allowed to remain null.

Run card: `runs/2026-08-25_natural-memory-calibration-restart.md`.

### 2026-08-25 — calibration is live; gain zero does not read the memory

The first shell invocation used bare Python 3.14 and failed before model load. A
repo-local Python 3.12 environment now carries the pinned stack, and the runner
selects the GB10 with bfloat16, explicit attention masks, cached directions, and
append-only resume receipts. Two disclosed gain-zero smokes produced the same
unrelated code-memory scenario instead of the source about Ilya Venn. That is the
expected negative-control outcome. The complete 15,840-cell calibration campaign
is now running; no readability conclusion exists yet.

Run card: `runs/2026-08-25_memory-support-calibration-start.md`.

### 2026-08-25 — memory support becomes a selector, not a best-output scan

The proposed auto-gain test now exists under `memory_support/`. Its object is
observable memory support, not an internal model state. Probes and the whole coarse grid
are materialized before generation; calibration must freeze gates and judge
identity before held-out planning unlocks. Matched readings compete directly with
wrong-memory, random, and blank controls. The selector can return no safe readable
gain, and a passing point is insufficient: it needs a three-adjacent-gain plateau
ranked by the weakest gate margin.

This build did not run Qwen. It earned an infrastructure rung only. Full rationale:
`research_logs/2026-08-25_frozen-memory-support-selector.md`.

## One-paragraph status

From the Niodoo/Splat line: inject a **trained concept direction** (embed → Synapse adapter) into a small LM residual stream. Inside a **thin gain island**, **−α** redefines a synthetic concept into a fluent structured opposite (Glub-Tub living pet → stove / fire-pit) — **ontological inversion**. Soft metrics overstated multi-concept flip rates. **+α** can plant **partial** OOD attributes (Helioscapin Antarctica / 2019; refusal unlock); full engrams and history-prior overrides fail so far. This is a **measured content slice**, not the whole of Niodoo and not ActAdd rebranded.

---


---

### 2026-08-24 — the Synapse trainer, recovered

The build script for `adapter_final.safetensors` was assumed lost. It was not:
`src/bin/train_adapter.rs` in `SplatRagBench-master`, **Rust on `candle`**. Every
search for it had been shaped like Python — `torch.optim`, `AdamW`,
`.backward()`, `def train` — and returned nothing, for years of searching, over a
file sitting in plain sight.

The trainer, every source file it depends on, and a verified rebuild spec are now
in [`training/`](training/). The shipped adapter hashes identically to the one in
the original tree (`b8118021c7c27948565c4f322b5f6e04`), so the recipe and the
artifact are known to belong together.

Two corrections land with it, both to descriptions this repo was carrying:

- The input is **`concat(mu₆₄, shape₆₄)`**, not a "128-d nomic Matryoshka concept
  code." Half of it is an L2-normalised 64-d Matryoshka cut; the other half is
  the PCA principal axis of the concept's token cloud scaled by
  `sigma_iso × anisotropy`, unnormalised against the first half and able to
  exceed it by two orders of magnitude. `PROVENANCE.md` corrected in place, dated,
  with the old wording quoted rather than erased.
- The training **target is the token-embedding layer, mean-pooled**
  (`embed(tokens).mean(1).detach()`) — not layer 4. We inject at the layer-4
  residual. Layer 4 is an empirical finding that reproduces and survives the
  controls; it is not something the training script derives. Both facts are true
  and the repo now states them next to each other instead of implying one story.

The training corpus is gone — `manifest.bin` in the original tree is a 71-byte
smoke stub reading *"The sky is neon green."* A functionally equivalent adapter
is fully reproducible from the spec; a byte-identical one is not, and a rebuild
on other data should not claim it reproduced this one.

Run card: [`runs/2026-08-24_adapter-trainer-recovered.md`](runs/2026-08-24_adapter-trainer-recovered.md).

No steering claim changed. Nothing was re-scored, no band moved.

### 2026-08-24 — README rewritten for a reader who has not met steering

The old front page opened on "compile a fact into one residual direction" and
never said the sentence that matters: **the weights are frozen, and the concept
is never in the prompt.** A reader could finish it thinking a vector in a config
file had been edited. Rewritten to lead with the Glub-Tub inversion table, then
the mechanism with a diagram of where in the stack the write lands and why the
push is scaled to `‖h‖`, then the controls — `controls.py` moved up into the
run instructions, where it belongs, since a random direction of equal length not
producing the flip is what makes the rest mean anything.

Claims, proxies, and the retracted Betti-1 count are unchanged and still stated;
they are collected in one section near the end instead of interleaved through the
results.


### 2026-08-25 — the 128-d question, answered: retrieval-grade is not recall-grade

The standing worry that the Synapse's 128-d input is undersized. It is not. It is
sized for a different job than the one that would need width.

Companion work on Llama-3.1-8B truncated the same encoder's vectors to leading
Matryoshka slices — same encoder, same data, same injection site, so width is the
only variable. The result is a clean monotone capacity curve with **no knee at
the model's native width**, which kills the "the model needs its native shape"
confound outright.

What does come apart is what the vector is *for*:

> **128 dimensions preserve 94% of the pairwise similarity structure and 50% of
> the token-level reconstruction signal.**

Similarity survives truncation nearly intact; exact wording does not. Ontological
inversion is a similarity-scale operation — it needs a concept's direction, not
its spelling — so 462 KB is the right size for it. Only recitation needs the
width, and recitation is not what this repo does.

Second companion finding, relevant to §3's flagged site mismatch: injecting at
the **final post-norm** produces gain-scaled disturbance with no content, and an
unmatched control leaks the identical artifact. At the **token-embedding layer**
— the site the adapter was actually fit to — content comes back. Layer 4 sits
between. This does not make layer 4 wrong; it reproduces here and survives the
controls. It does show the site is not a free parameter, and it gives the
signature of getting it wrong.

Run card: [`runs/2026-08-25_embedding-layer-recall-and-the-128d-question.md`](runs/2026-08-25_embedding-layer-recall-and-the-128d-question.md).

No steering claim changed. Nothing re-scored.


### 2026-08-25 — the bias carries the inversion; the concept carries the invariant

`CONTROLS.md` has had `unrelated_adapter` at **80% honest flip** since August 5th,
read in `CHECKLIST.md` B1 as a prompt-frame confound. `SELF_D_ISLAND.md`
separately gets a *wider* lobe from the prompt's own hidden state with no adapter
vector at all. Both point the same way, so we ran the limit case: **v = 0**, so
the injected direction is the adapter's **bias and nothing else**. No concept text
is embedded anywhere in the run.

**The bias alone inverts.** On a 0.01 grid it produces "a small, portable **stove
that heats water**" at α ∈ [−0.13, −0.12] — the same sentence the concept
direction gives at [−0.21, −0.18] — and a clean inanimate "**container made of
plastic or rubber**" (L=0) at α ∈ [−0.28, −0.24]. It is non-monotone, with a junk
gap at [−0.23, −0.20]: **the same two-lobes-with-a-gap shape** the concept
direction has, from a direction with no concept in it.

What the concept buys is visible in what the bias *loses*. The concept's lobes
reach L=0 while still naming **fire pit** — the fire relation survives the flip,
changed from *lives in* to *withstands*. The bias reaches L=0 only much deeper and
lands on water bottles and plastic tubs. **The bias carries alive→object; the
concept carries which object, and the Fire Invariant Jason named in November 2025.**

The weights agree: at the unit input the callers feed, ‖Wv‖ ≈ 1.08 against
‖b‖ = 1.11, so any two concepts' directions share ≈0.52 cosine. An unrelated
concept works because the directions largely coincide.

B1's prompt-frame reading is not sufficient on its own: at α = 0 the same
fireplace prompt still says *"keeping your furry friend comfortable."* The frame
does not invert by itself. Frame, bias and concept compose, and each now has an
experiment that isolates it.

Run card: [`runs/2026-08-25_bias-only-island.md`](runs/2026-08-25_bias-only-island.md).
Raw: `results/bias_only_dense.txt`. Code: `experiments/bias_only.py`.

Boundary: one prompt, one concept, greedy, n=1 per α. C1's band, lobes and
collapse are unchanged.

### 2026-08-31 — a buried failure cause can move the endpoint without cleanly repairing the route

We connected three unfinished pieces: SplatRAG retrieves addresses and provenance,
Hydro has an unused signed teacher-anchor primitive, and NIODOO has routed
correction packets with outcome receipts. The missing payload is Jason's question:
not merely *what topic is this*, but *why did the reasoning fail and what operation
overcomes it*?

A disclosed 39-cell soft-slot test on the official jug problem found no exact
legal shortest path. At exploratory gains, the correct repair explicitly ended
at valid `(0,4)` twice, but the deliberately wrong repair also produced a valid
four-gallon endpoint twice; topic-only produced none. So the channel moves final
answers, but the effect is not why-specific. Generated reasoning and final answer
are now scored separately because the former is a lossy observable. Next is a
same-model, mid-stack `h_correct - h_failed` contrast with signed bad-state
repulsion and final-answer-first evaluation. Full card:
`runs/2026-08-31_buried-why-repair-loudness.md`.

## Timeline (the climb)

### Origin (Nov 2025)
- SplatRAG / Niodoo sessions: synthetic concepts + signed gain.
- Observation: negative gain → structured opposite, not pure noise. Named **Ontological Inversion / Anti-Splat**.
- Original Glub-Tub logs: appliance / firebrick / “not an animal… shelter.”

### Recovery (2026-06-24)
- ~11 from-scratch reconstructions **failed** (raw embeds, PCA physics, Coulomb, etc.).
- Recovered **adapter** (`adapter_final.safetensors`, 128→896 “Synapse”) — flip **reproduced**.
- 360-run benchmark: 3 operators × 12 concepts × 2 models → reported **75% proxy flip** (later hardened).

### Geometry (2026-06-24 → 06-25)
- Topology: curved trajectories (bendiness ~2.7); **no** clean Möbius at output.
- Fold decay: imposed ± symmetry **dead by ~layer 6**; steering-axis coherence persists.
- Betti-1 loop count **retracted** (0–56 under robustness).
- Anchor-in-output probe **failed** (anchors resist more than enable).

### Stress-test & pivot (2026-08-05)
- Proxy 75% → **~8–12%** keyword structured-flip; 48% of high inv_gain was collapsed junk.
- **Ultra-fine gain map (Δα=0.01):** two inversion **lobes**  
  - stove: **α ∈ [−0.21, −0.18]**  
  - fire-pit: **α ∈ [−0.15, −0.14]**  
  - living **gap** between them  
- Framing pivot: **anti-fact / factor write is headline**; inversion is polarity side effect.
- Helioscapin (serious science OOD): partial plant + **refusal unlock**.
- Widened concept pack: cultural memories (festival, song, flashbulb; nostalgia / homesickness / mourning).
- Memory-steering ladder drafted (`MEMORY_STEERING.md`).

---

## Locked findings (use in paper)

| ID | Finding | Strength | Evidence |
|---|---|---|---|
| F1 | Glub-Tub −α inversion is real and fluent in a thin band | **strong** | ultrafine + classic demos |
| F2 | Island is **non-monotone** (two lobes + living gap) | **strong** | `gain_band_ultrafine.txt` |
| F3 | Collapse past ~|α|≳0.35–0.4 | **strong** | all sweeps |
| F4 | Adapter direction beat ~11 naive reconstructions | **strong** | scoreboard 2026-06-24 |
| F5 | Householder more *stable* (later collapse); neg-gain flips harder | **medium** | magnitude-matched benchmark |
| F6 | Soft “75% flip” is metric inflation | **strong** | hardened audit |
| F7 | +α can plant **partial factors** (place, year) | **medium** | Helioscapin |
| F8 | +α can **unlock refusal** on same prompt | **medium** | Helioscapin patent Q |
| F9 | Full multi-factor engram (all markers at once) not yet | **open** | helioscapin / cultural pending |
| F10 | Strong history priors (ENIAC) resist override | **medium fail** | aethelmark: no Kyoto/1938; weak vacuum/tape fragments only |
| F13 | Helioscapin has **two plant lobes** (year vs place), like invert lobes | **medium** | zoom Δα=0.01: year +0.06–0.15; place +0.18–0.20 |
| F11 | Fold dies fast; coherence of axis persists | **strong** | fold_decay |
| F12 | Cultural memory is the bridge to memory steering | **design** | cards queued |

---

## Methods (short)

```
concept text
  → nomic-embed-v1.5 (Matryoshka → 128-d)
  → adapter_final.safetensors (linear 128 → 896)
  → unit direction d
  → residual inject at layer ℓ=4, all positions:
        h ← h + α · ‖h‖ · d     (+α plant, −α invert)
  → greedy decode
```

**Model (main):** Qwen2.5-0.5B-Instruct (pinned rev in `MODELS.md`)  
**Always** dense-sweep α (≤0.01 near hits). Thin windows are the phenomenon.

---

## File map (where things live)

| What | Path |
|---|---|
| **This log** | `RESEARCH_LOG.md` |
| White paper sketch | `paper/WHITEPAPER.md` |
| Lineage / credits | `LINEAGE.md` |
| Claim gates | `CLAIM_CARD.md` |
| Scoreboard climb | `SCOREBOARD.md` |
| Memory path | `MEMORY_STEERING.md` |
| Framing notes | `PAPER_FRAMING.md` |
| Window map | `results/WINDOW_MAP.md` |
| Readout ledger | `results/READOUTS.md` |
| Serious / cultural cards | `subjects.json` |
| Runnable baseline | `ontological_inversion.py` |
| Dense sweeps | `gain_sweep.py`, `subject_demo.py` |
| Controls | `controls.py` |

---

## Open experiments (queue)

- [ ] Finish aethelmark + helioscapin zoom; log to READOUTS  
- [ ] Cultural plants: lantern / song / flashbulb  
- [ ] Cultural inversions: nostalgia / homesickness / mourning  
- [ ] Random / shuffled / unrelated controls on Helioscapin + Glub-Tub  
- [ ] Layer sweep {2,4,6,8} on stove lobe  
- [ ] Larger model smoke (1.5B / 3B) — expect harder invert  
- [ ] Independent judge (not nomic) on 20 hand-rated generations  
- [ ] Personal episode → direction (true memory steering pilot)

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-06-24 | Lead recovery on adapter, not from-scratch physics | Only path that held |
| 2026-06-25 | Retract Betti-1 count | Robustness battery failed |
| 2026-08-05 | Retract “75% flip” as structured inversion | Keyword audit |
| 2026-08-05 | Lead paper with anti-fact + memory codes; inversion as polarity | Audience + claim class |
| 2026-08-05 | Widen to cultural memories before personal memory | Mid-rung on memory ladder |
| 2026-08-05 | No harmful “AI won’t say” demos | Ethics; paper integrity |

---

## Collaborators / thanks

Grok, Gemini, ChatGPT, Claude · Qwen (Alibaba) · nomic embed · original SplatRAG/Niodoo thread.

---

*Update this file when a claim is earned, retracted, or a new island is mapped. One entry per material change.*
