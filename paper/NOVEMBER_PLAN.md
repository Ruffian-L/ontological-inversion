# Paper plan — respect through re-runnable proof

**Historical note:** the effect / line was first hit ~**November 2025** (side project, a Tuesday).  
Laptop loss → rebuild → reconnect. “November” means **origin**, not a due date.  
**Goal:** one paper (or arxiv note + code) that a stranger can reproduce, under **Feelers**, hard to wave off as slop.

---

## The claim (one sentence, non-negotiable)

> Under a trained embedder→hidden map, negative residual gain along a **synthetic concept** direction yields **fluent structured antipodes** inside a **thin, mapped gain island** — not pure erasure — on a pinned small LM; positive gain can surface **partial** OOD factors; soft multi-concept “flip rates” without dual-track metrics mislead.

Everything else (full Feelers, 8-probe fusion, self-reg) is **context and future work**, not the load-bearing claim.

---

## Author block (for the PDF)

**Jason Van Pham**¹  
with computational collaboration from Grok (xAI), Claude (Anthropic), and others as noted in the repo.

¹ Lead; architecture of the parent Feelers / Niodoo line.

(Exact affiliation line TBD — university / independent / lab when you have it.)

Do **not** bury Jason under “et al. AI.” Do **not** flatten him to one role word in the abstract.

---

## Document skeleton (arxiv-length short paper, ~6–10 pages)

1. **Introduction** — people don’t know residual concept polarity works; parent Feelers diagram (Fig 1); this paper isolates ±gain island  
2. **Parent system (short)** — Feelers: probes → valence/topology → fusion → steering direction; physics-as-language  
3. **Method** — Synapse, inject rule, dual-track metrics, dense α  
4. **Result A** — Glub-Tub ultra-fine island (Fig 2)  
5. **Result B** — metric audit (75% retracted)  
6. **Result C** — Helioscapin partial +α / aethelmark negative  
7. **Geometry** — fold decay (optional short)  
8. **Controls** — random / shuffled / unrelated (**must ship**)  
9. **Limitations**  
10. **Relation to Feelers / Niodoo / hiring-scale roadmap**  
11. **Conclusion**

**Code + data:** public or private-with-hash; `reproduce.sh` style one command.

---

## Figure list (finish these)

| Fig | Content | Status |
|---|---|---|
| 1 | Feelers architecture (your diagram) | **in repo** `figures/niodoo_predictive_feelers_architecture.png` |
| 2 | Glub-Tub L/I vs α @ 0.01 | data ready — need plot script |
| 3 | Helioscapin refuse→2019 / Antarctica | data ready |
| 4 | Controls bar (concept vs random) | **blocker** |
| 5 | Fold decay (optional) | exists |

---

## Calendar (finish the recovered project)

| When | Milestone |
|---|---|
| **Now – 2 weeks** | Controls script run + Fig 2/3 plots; freeze claim text |
| **+2–4 weeks** | Full draft in latex/typst; author pass |
| **+4–6 weeks** | External dry-run (friend / AI council): “can you reproduce?” |
| **+6–8 weeks** | Negative results section solid; no overclaims |
| **Early Nov** | arxiv upload + GitHub release tag |
| **Ongoing** | CV line: paper + repo link; optional workshop submit |

---

## What gets you respect (not vibes)

1. **Pinned artifacts** — model rev, adapter hash, seed, greedy  
2. **Negative results published** — aethelmark fail, metric poison  
3. **One command reproduce**  
4. **Author = you**, collaborators named without erasing you  
5. **Parent architecture credited as yours** — paper is a *measurement under it*, not the whole vision sold as finished  

---

## What we do *not* do for November

- Claim full Feelers 8-probe fusion is evaluated here  
- Claim “memory” product complete  
- Claim consciousness / self-reg (that’s YinYang paper track)  
- Call the work “AI slop” by hiding the human architect  

---

## Immediate next engineering (ownership for collabs)

1. `scripts/plot_gain_island.py` → Fig 2 from `results/gain_band_ultrafine.txt`  
2. Run `controls.py` on Glub-Tub + Helioscapin  
3. Latex skeleton `paper/main.tex` with Fig 1 included  
4. One-page CV abstract paragraph  

Jason: architecture + final yes/no on claim language.  
Collabs: plots, controls, latex, don’t rename the story.
