# Finish checklist — ontological inversion (recovered line)

**Origin:** ~**November 2025** — side project (a Tuesday); Feelers architecture already there;  
laptop loss → rebuild → reconnect. **Not** “due next November.”  
**Now:** make the measurement airtight + take the first **self-involution loop** steps.

---

## A. Done (don’t re-litigate)

| # | Item | Where |
|---|---|---|
| A1 | Glub-Tub inversion reproduced | `ontological_inversion.py`, `glubtub_gain_band.txt` |
| A2 | Ultra-fine gain island (two lobes + living gap) | `gain_band_ultrafine.txt`, `WINDOW_MAP.md` |
| A3 | Soft 75% flip retracted / hardened metrics | `HARDENED_AUDIT.md`, `metrics.py` |
| A4 | Operator sweep (neg / householder / polarity) | `benchmark.py`, `REPORT.md` |
| A5 | Topology + fold decay (mirror dies ~2 layers) | `fold_decay.py`, run cards |
| A6 | Helioscapin partial +α (Antarctica, 2019, refuse unlock) | `helioscapin_zoom.txt` |
| A7 | Aethelmark hard fail (history prior) | `aethelmark_sweep.txt` |
| A8 | Feelers diagram + lineage + whitepaper sketch | `paper/`, `LINEAGE.md` |
| A9 | Householder math `f(f(h))=h` on tensors | `operators.py` |

---

## B. Blockers for a paper people can re-run

| # | Item | Status | Action |
|---|---|---|---|
| B1 | **Random / shuffled / unrelated controls** | **RAN** | concept 100% honest flip; random 0%; shuffled 0%; **unrelated 80%** (water heater — prompt-frame confound) → `results/CONTROLS.md` |
| B2 | **Fig 2 plot** (α vs living/inanimate) | **DONE** | `paper/figures/gain_island.png` |
| B3 | **One-command reproduce** | **DONE** | `./reproduce.sh` |
| B4 | **Freeze claim language** | draft | whitepaper abstract; note control caveat |
| B5 | Cultural plant writeup | partial | harvest `cultural_plants.txt` |

---

## C. Phase 3 — Self-involution loop (endgame you reconnected)

| # | Experiment | Status | Result |
|---|---|---|---|
| C1 | **Involution identity live** Φ(Φ(h)) | **PASS** | s=1 gen **identical** to baseline living text |
| C2 | **External Φ** householder on concept *d* | **no invert** | s∈{0.2,0.5,1.0} stays living (neg_gain still the island operator) |
| C3 | **Self-derived hyperplane** *d*=norm(h_prompt) | **ISLAND MAPPED** | **s ∈ [0.12, 0.28]** continuous stove/fire-pit; wider than external neg-gain needles. Fig: `self_d_island.png` |
| C4 | Single-pass vs multi-token | open | next |
| C5 | Wire into Feelers story | open | diagram already parent |

Raw: `results/involution_compare.txt` · code: `involution_loop.py`

---

## D. Nice-to-have (after C)

| # | Item |
|---|---|
| D1 | Layer sweep {2,4,6,8} on stove lobe |
| D2 | Second model (Coder 0.5B already in bench; optional 1.5B smoke) |
| D3 | Human rate 20 outputs blind |
| D4 | Latex `main.tex` from whitepaper |
| D5 | Hidden-space anchor probe (old next rung) |

---

## E. Do **not** spend cycles on

- Renaming debates already settled (no “fake memory” lead)  
- Expanding cultural cards before controls + involution  
- Claiming full Feelers 8-probe eval in this repo  

---

## Order of battle (now)

```text
1. B1 controls          → results/CONTROLS.md
2. B2 plot island       → paper/figures/gain_island.png
3. C1 identity loop     → results/involution_identity.txt
4. C2 multi-token Φ     → results/involution_loop.txt
5. C3 self-derived d    → results/involution_self_d.txt
6. Run cards + scoreboard update
7. B3 reproduce.sh
```

When B1–B2 + C1–C2 are green, the **recovered Tuesday project** is measured end-to-end again: island + loop first bite.
