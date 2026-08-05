# Research Log — Ontological Inversion (Anti-Splat)

**Project:** ontological-inversion  
**Working title:** *Ontological Inversion: Residual ±Gain on Synthetic Concept Directions*  
**Lineage:** Niodoo / SplatRAG / collaboration — see [`LINEAGE.md`](LINEAGE.md)  
**Last updated:** 2026-08-05  

This is the **human-readable research log**.  
Raw receipts under `results/`; claims in `CLAIM_CARD.md`; climb in `SCOREBOARD.md`.

**Naming rule:** do **not** call synthetic concepts (Worbglob/Glub-Tub) “fake memories.”  
Real memory = splat stores, Niodoo continuity, golden-memories, queryable embedding packs.  
Synthetic concepts = **probes** for residual ±gain.

---

## One-paragraph status

From the Niodoo/Splat line: inject a **trained concept direction** (embed → Synapse adapter) into a small LM residual stream. Inside a **thin gain island**, **−α** redefines a synthetic concept into a fluent structured opposite (Glub-Tub living pet → stove / fire-pit) — **ontological inversion**. Soft metrics overstated multi-concept flip rates. **+α** can plant **partial** OOD attributes (Helioscapin Antarctica / 2019; refusal unlock); full engrams and history-prior overrides fail so far. This is a **measured content slice**, not the whole of Niodoo and not ActAdd rebranded.

---

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
| Serious / cultural cards | `serious_subjects.json` |
| Runnable baseline | `ontological_inversion.py` |
| Dense sweeps | `gain_sweep.py`, `serious_demo.py` |
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
