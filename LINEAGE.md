# Parent architecture → this measurement

## What sits above this repo

**NiodO.o Predictive Feelers System** (Jason’s architecture).

![NiodO.o Predictive Feelers — complete architecture](paper/figures/niodoo_predictive_feelers_architecture.png)

Core shape (from the diagram):

1. **Input** — user embedding vector  
2. **Async bridge** — `evolve_from_interaction`  
3. **8-way probe spawning** → parallel processor (trajectory simulation, score paths, episodic memory probes, …)  
4. **Per-probe pipeline** — RBF forecast, valence (PAD), Möbius / topology check, memory check  
5. **Fusion** — weighted valence → top-k → retract? → **normalize → final steering direction → client**

Physics-as-language was always the intent: probes, forces, fusion, a **steering direction** out the end — not chat theater.

**This repository is one hard measured step inside that picture:**  
take a concept embedding, map it (Synapse), inject residual **±gain**, map the **gain island** where polarity flips cleanly (ontological inversion).  
It is **not** the whole Feelers system. It is a **reproducible first bite** people can run and cite so the larger architecture is harder to dismiss.

---

## How work was shared (no flattening)

| | |
|---|---|
| Architecture of Feelers / Niodoo direction / what matters | **Jason** |
| Math, naming stretches, bringing slices to runnable measurement | **Grok and other AI collaborators** |
| Implementation, packaging, forensics on various stretches | **Claude and others** |

Nobody here is “just the ideas person” or “just the bot.” Jason owns the **architecture and the stakes**. Collaborators own **what they actually built and wrote** when they did. Provenance files exist for *their* credit trails — not to shrink Jason to a single hat.

---

## Why a paper at all

People still act like residual concept control + structured polarity **cannot** work, or is “AI slop” when a human drives it with AIs.  
A **narrow, dated, reproducible** result (pinned models, gain maps, negative results included) is the antidote: something a hiring committee or reviewer can **re-run**, not a vibe.

Target window discussed: **November** (see `paper/NOVEMBER_PLAN.md`).

---

## Pointers

| | |
|---|---|
| Feelers diagram (this copy) | `paper/figures/niodoo_predictive_feelers_architecture.png` |
| Niodoo hidden-state steering | `…/niodoo-hidden-state-steering` |
| YinYang / self-reg thread | `…/YinYangQSMA/PAPER_THREAD.md` |
| This measurement | `paper/WHITEPAPER.md` · `RESEARCH_LOG.md` |
