# Scoreboard

The rolling ledger of the climb — every eval's rung, newest at the bottom. Progress, not blame
(STANDARDS.md §3). Each run card restates its own recent rungs inline; this is the full ladder.

| Date | Ran by | What we tried | Result | Card |
|---|---|---|---|---|
| 2026-06-24 | Claude (Opus 4.8) | reconstruct the antipode from scratch — 11 ways (embedding NN, transformer hidden-state steering on Llama-8B + Qwen-0.5B, PCA physics, charge/Coulomb, semantic combos) | none reproduced the clean flip — all drift→collapse | — |
| 2026-06-24 | Claude (Opus 4.8) | recover the trained adapter ("Synapse", `adapter_final.safetensors`) and inject its direction into Qwen-0.5B residual | **flip reproduced** — magma-hamster → "stove that heats water", sweet spot α≈0.2 | `README.md` |
| 2026-06-24 | Claude (Opus 4.8) | quantified benchmark — 12 concepts × 3 operators × 5 strengths × 2 models (360 runs) | **generalizes (75%)**; Householder involution is the most *stable* operator (collapse onset 0.61) | `results/REPORT.md` |
| 2026-06-24 | Claude (Opus 4.8) | topology v1 — capture the steered hidden state *at the injection layer* | artifact: trivial straight line (fold=−1.00, bendiness=1.00 by construction) | — |
| 2026-06-24 | Claude (Opus 4.8) | topology v2 — capture the *propagated final* hidden state across the steering band | **MIXED**: curved trajectory (bendiness 2.7) + topological loops (Betti-1≈7, soft); but **no clean Möbius fold** (fold≈−0.13) | `runs/2026-06-24_topology-of-the-flip.md` |
| 2026-06-25 | Claude (Opus 4.8) + council | per-layer fold decay + PH robustness battery (bootstrap / pooling / leave-one-out) | **MIXED**: imposed symmetry dies by layer 6 (Phase-3 window = layers 4–5); coherence persists; **Betti-1 swings 0–56 → loop-count retracted** | `runs/2026-06-25_per-layer-fold-decay.md` |
| 2026-06-25 | Claude (Opus 4.8) | anchor detection via output semantic-axis projection (which axes flip vs resist) | **FAIL**: probe too coarse (susceptibility 0.02–0.06, no separation; animacy near bottom); anchor-presence hint that context *resists* the flip. Next: hidden-space probe | `runs/2026-06-25_anchor-detection.md` |
| 2026-08-05 | Grok (xAI) | paper stress-test: re-score 360 runs with keyword structured-flip; audit proxy poison; draft controls + claim card | **HARDENED**: proxy 75% → ~8–12% keyword-flip cells; 48% of high inv_gain is collapsed garbage; **Glub-Tub α≈0.2 stove/fire-pit remains the real existence proof**. Controls script added (`controls.py`); paper draft reframes generality | `results/HARDENED_AUDIT.md`, `CLAIM_CARD.md`, `PAPER_DRAFT.md` |
| 2026-08-05 | Grok (xAI) + jp insight | **framing pivot**: inversion is side effect; headline = anti-fact factor injection (+gain writes factors model never learned) | **BUILT**: `anti_fact.py` + `anti_facts.json` (zorblite/kelthren/vexorine/cairo2012/glubtub); `PAPER_FRAMING.md`. Run pending for C0 | `anti_fact.py`, `PAPER_FRAMING.md` |
| 2026-08-05 | Grok (xAI) | dense gain sweep step 0.025 classic Glub-Tub + factor probes | **WINDOW MAPPED**: clean invert only at **α=−0.15 (fire pit)** and **α=−0.20 (stove)**; neighbors living again → **thin non-monotone island**. +α does not plant magma/hamster on bare probes. `gain_sweep.py` added | `results/WINDOW_MAP.md`, `results/gain_band_classic_dense.txt` |
| 2026-08-05 | Grok (xAI) | serious subjects: helioscapin anti-fact + culpability/scarcity inversion | **PARTIAL WIN (helioscapin)**: +α plants **Antarctica** + **2019** year; baseline patent Q **refuses**, steered **answers**. Full multi-factor engram not yet. Culpability/scarcity inversion **no clean lobe**. | `subjects.json`, `results/SUBJECT_DEMO.md` |
| 2026-08-05 | Grok (xAI) | **widen to cultural memories** (festival, folk song, flashbulb plant; nostalgia/homesickness/mourning invert) + memory-steering ladder | **CARDS + QUEUE**: `lantern_of_velmire`, `song_of_ashmere`, `first_radio`, `nostalgia`, `homesickness`, `mourning_ritual`. Path doc `MEMORY_STEERING.md`. Sweeps queued after helioscapin/aethelmark | `MEMORY_STEERING.md`, `results/READOUTS.md` |
| 2026-08-05 | Grok (xAI) | research log + white paper sketch v0.1 | **DOCS**: human log `RESEARCH_LOG.md`; sketch `paper/WHITEPAPER.md` (abstract→limitations, fig plan, quote bank) | `RESEARCH_LOG.md`, `paper/WHITEPAPER.md` |
| 2026-08-05 | Grok (xAI) + Jason | **naming correction**: drop “fake memory”; lineage Niodoo/Splat/collab first; OI = synthetic concept residual ±gain | **FIXED** whitepaper v0.2, `LINEAGE.md`, research log, framing | `LINEAGE.md`, `paper/WHITEPAPER.md` |
| 2026-08-05 | Grok | finish checklist + run B1/B2/C1–C3 | **CONTROLS** concept≫random/shuffled; unrelated partial (frame). **Fig** `gain_island.png`. **Involution:** Φ∘Φ=id PASS; **self_d s=0.2 → stove** (self-loop first bite) | `CHECKLIST.md`, `involution_loop.py`, `results/involution_compare.txt` |
| 2026-08-05 | Grok | **self_d dense sweep** 16 strengths | **SELF-LOOP ISLAND** s∈**[0.12, 0.28]** fire-pit→stove plateau; collapse ≥0.7. No external concept vector. | `results/SELF_D_ISLAND.md`, `paper/figures/self_d_island.png` |

> The climb: 11 from-scratch tries didn't hold → the recovered adapter held → it *looked* like it
> generalized under a soft proxy → topology found curve not fold → loops retracted → anchors
> failed → **paper audit showed the 75% was metric inflation and the star result is still real**.
> Next rung: run `controls.py` (random / shuffled / unrelated) so the direction claim is airtight.
> Rungs, not faults, all the way up.
