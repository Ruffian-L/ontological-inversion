# Paper framing — Ontological Inversion (corrected 2026-08-05)

**Source of truth:** [`LINEAGE.md`](LINEAGE.md) + [`paper/WHITEPAPER.md`](paper/WHITEPAPER.md)  
**Do not use “fake memories” as the title or lead claim.**

---

## Hierarchy (earned first)

| Rank | Phenomenon | One sentence | Evidence |
|---|---|---|---|
| **1 — HEADLINE** | **Ontological inversion (−α)** | Synthetic concept residual code; negative gain → fluent structured opposite in a thin island | Glub-Tub ultra-fine lobes |
| **2 — SECONDARY** | **Factor fragments (+α)** | Same codes can surface partial OOD attributes; refusal unlock | Helioscapin partial |
| **3 — GEOMETRY** | Fold decay / curved traj | Mirror dies fast; axis coherence persists | run cards 2.2 / 2.2b |

Lead with **inversion** (named in collab, strongest result). Factor write is open/partial.

---

## Why “fake memory” was wrong

- Real memory in this collab = splat packs, Niodoo continuity, golden-memories, queryable embeddings.  
- Worbglob is a **synthetic concept probe**, not a statement that those memories are fake.  
- Related work starts at **Niodoo / SplatRAG / feelers**, not ActAdd-as-parent.

---

## Title options (use these)

1. *Ontological Inversion: Residual ±Gain on Synthetic Concept Directions* **(preferred)**  
2. *The Anti-Splat: Structured Antipodes under Negative Concept Steering*  
3. *Thin Islands: Dose-Sensitive Ontological Inversion in Small LMs*

---

## Abstract skeleton (anti-fact first)

```
We show that a synthetic fact never present in pretraining can be compiled into a
single residual-stream direction via a frozen text embedder and a trained linear
adapter. Injecting that direction with positive gain causes a small LM to answer
probes using the fact's latent factors — color, origin, causal effects — even
though those attributes never appear in the prompt. We call this anti-fact factor
injection. As a side effect, negative gain on the same direction does not merely
erase the concept: in a narrow band it yields ontological inversion, a fluent
redefinition into a structured opposite (e.g. a living "Glub-Tub" becoming a stove
or fire pit). Controls (random / unrelated directions) and keyword factor scoring
separate real write-in from generic perturbation. Topology measurements show the
imposed polarity is short-lived in depth while the concept axis remains coherent.
The primary result is a write path for unknown structure; inversion is its
polarity twin.
```

---

## Figure plan (paper order)

1. **Fig 1 — Anti-fact cartoon:** text fact → embed → adapter → +α residual → answers with factors; prompt box shows questions *without* the fact.
2. **Fig 2 — Factor hit-rate bars:** baseline vs concept+ vs random+ vs unrelated+ for Zorblite / Kelthren / Vexorine.
3. **Fig 3 — Hard mode Cairo 2012:** London prior vs Cairo under +α (if it works; if not, honest fail is fine).
4. **Fig 4 — Inversion side effect:** Glub-Tub gain band (existing).
5. **Fig 5 — Dose island:** factor-rate and collapse vs α for ±gain.
6. **Fig 6 — Fold decay:** symmetry dies, coherence lives (existing).

---

## Scoring (rank-1 must not use nomic alone)

| Signal | What it measures |
|---|---|
| **Factor hit-rate** | Fraction of planted attributes whose keywords appear in the answer |
| **Leak rate** | Real-world prior / “I don't know” / wrong real entity (e.g. London, mercury) |
| **Control gap** | concept+ − max(random+, unrelated+, baseline) |
| **Cross-prompt consistency** | Same factors across different probes (not one lucky sentence) |

A run **passes** if control gap > 0 on ≥2 synthetic cards with multi-prompt consistency.

Inversion metrics stay secondary (keyword antipode + coherence gate).

---

## What “factors the model doesn't know” means precisely

Not mystical. Operational definition:

1. Choose entity string \(E\) with near-zero pretraining support (novel coinage) **or** a real entity with a **false** property set (historical anti-fact).
2. Write anti-fact \(F(E) = \{f_1,\ldots,f_k\}\) with measurable keywords.
3. Build direction \(\hat d = \mathrm{norm}(W e(F) + b)\) — **\(F\) never enters the generation prompt**.
4. At +α, answers to questions about \(E\) express \(f_i\) at rates above controls.

That is “turning a concept into factors”: the residual direction is a compressed code for \(\{f_i\}\); the model decodes them on demand.

---

## Tonight order of battle

```bash
# 1) A-side showpiece (synthetic first — easier than Cairo)
.venv/bin/python anti_fact.py --names zorblite,kelthren,vexorine,glubtub_factors

# 2) Hard mode (optional; may fail — still publishable as limit)
.venv/bin/python anti_fact.py --names cairo2012

# 3) Direction necessity for inversion (B-side controls)
.venv/bin/python controls.py --concepts glubtub

# 4) Classic band (figure 4)
.venv/bin/python ontological_inversion.py
```

If (1) lands: paper title and abstract switch to anti-fact first tonight.  
If (1) fails and only (4) holds: stay inversion-first but still *frame* synthetic concept as fake-memory code.

---

## Relation to “most AI still don't think this is possible”

They usually deny one of:

- **D1:** You can't add knowledge without weights/prompt tokens.  
  → Anti-fact experiment attacks D1 directly.
- **D2:** Negative steering only suppresses, never structures an opposite.  
  → Inversion attacks D2.
- **D3:** Synthetic OOD strings have no geometry in residual space.  
  → Adapter success + factor consistency attack D3.

Lead with D1. D2 is the encore. D3 is the method lemma.
