# Ontological Inversion via Synthetic Concept Steering
### Superseded sketch — use `paper/WHITEPAPER.md` v0.2 + `LINEAGE.md`

> **One-line claim (defensible):**  
> Negative residual-stream steering with a *trained projection of a synthetic concept*
> (never in pretraining) can re-define that concept into a *structured contextual opposite*
> — not mere erasure — inside a narrow gain band on small LLMs. Parent line: Niodoo/SplatRAG.

> **Do not write:** “fake memory” as the product name; 75% flip as structured inversion.

---

## 0. What we actually have (and what we don't)

| Piece | Status | Paper treatment |
|---|---|---|
| Glub-Tub living→inanimate flip (stove / fire pit / shelter) | **Reproduced, readable, sweet-spot α≈0.15–0.30** | Lead result / existence proof |
| Trained adapter (“Synapse” 128→896) required | **11 from-scratch methods failed; adapter held** | Core method claim |
| Collapse past α≈0.4 | **Clear** | Dose–response figure |
| Operator comparison (neg / Householder / polarity) | **Real magnitude-matched sweep** | Secondary: stability vs flip strength |
| “75% flip success” | **Proxy metric inflated**; keyword-hardened ≈ **8–12%** of cells | Retract or reframe as *directional shift*, not flip |
| Anchors “enable” inversion | **Own experiment FAILED** — anchors *resist* | Do not claim; report as open |
| Clean Möbius fold / Betti loops | **Retracted** (fold dies by layer 6; loops sampling noise) | Topology as *curved + asymmetric*, not fold |
| Fake *memory* (KV) | **Not tested** — this is residual *activation* steering of a concept code | Frame carefully: “synthetic concept vector”, not “planted memory” |
| Scale / model family | Only Qwen2.5-0.5B × 2 | Explicit limitation |
| Random / shuffled / unrelated controls | **Script ready (`controls.py`), not yet run tonight** | **Must run before submission** |

---

## 1. Abstract (draft)

Activation steering can suppress or amplify high-level attributes, but it is usually
assumed that *subtracting* a concept erases or degrades it. We study the opposite
outcome. Given a **synthetic concept** never seen in training (e.g. *Glub-Tub*:
“a magma-eating hamster that lives in a tub”), we map its text embedding through a
trained linear adapter into the residual stream of a small instruction model and
apply **negative gain**. In a reproducible sweet-spot band (α ≈ 0.15–0.30), the model
does not gibber or refuse: it **redefines** the concept as a coherent *inanimate*
heat/water/container object (portable stove, fire pit, shallow hole) while remaining
fluent. Beyond α ≈ 0.4, generations collapse. We call this **ontological inversion**.

A 360-run sweep across 12 concepts and 3 operators shows that embedding-proxy
“flip rates” overstate structured inversion; an independent keyword audit finds clean
structured flips concentrated on the synthetic case that motivated the work. Topology
measurements show curved hidden-state trajectories and fast decay of imposed mirror
symmetry (dead by ~2 layers past injection), while steering-axis coherence persists.
We release code, the adapter, and run cards. The phenomenon is real, narrow, and
method-sensitive — not a universal involution of language.

---

## 2. Why “most AIs say this is impossible” (and where they're half-right)

Standard intuitions from ActAdd / CAA / RepE:

1. Negative α **suppresses** a behavior (toxicity↓, sycophancy↓).
2. Steering is continuous modulation along a known axis, not a discrete *category flip*.
3. Synthetic OOD concepts shouldn't have a stable residual direction at all.

What those views miss:

- Suppression and **redefinition under a fixed prompt frame** are different. Our prompt
  still asks about a fireplace pet; the name “Glub-Tub” remains; the *ontological
  category* of the referent flips.
- A **trained** embedder→hidden map (the adapter) lands the synthetic concept on a
  direction the residual stream can treat as concept-like. Raw embed pads and naive
  contrasts failed in our reconstruction attempts.
- Small models have less “semantic inertia” — large models often refuse/collapse instead
  of flipping (hypothesis; only 0.5B tested).

**Honest middle:** the clean flip is not “any concept, any model.” It is a
**dose-sensitive, adapter-dependent, concept-selective** effect best demonstrated on
synthetic concepts where there is no pretrained prior to fight.

---

## 3. Method (paper-tight)

### 3.1 Synthetic concept as fake memory *code* (not KV memory)

We do **not** write into the KV cache. We:

```
concept text
  → nomic-embed-v1.5 (Matryoshka slice → 128-d)
  → adapter_final.safetensors  (linear 128 → 896)   # "Synapse"
  → unit direction d ∈ R^h
  → residual inject at layer ℓ=4, all positions:
        h ← h + (−α) · ‖h‖ · d
  → greedy decode
```

Call it a **synthetic concept vector** or **fake-memory code**. Avoid “memory
injection” unless you add a KV experiment.

### 3.2 Operators (magnitude-matched)

| Operator | Form | Role |
|---|---|---|
| `negative_gain` | `h − α‖h‖d` | Strongest flip (baseline discovery) |
| `householder` | blend toward `(I−2ddᵀ)h` | True involution at full strength; **most stable** (later collapse) |
| `projection_polarity` | flip signed projection onto d | Polarity-aware variant |

Compare at equal ‖Δh‖ so the story is *geometry*, not push size.

### 3.3 Metrics — dual track (mandatory)

| Metric | Use | Failure mode |
|---|---|---|
| Embedding inversion score `cos(out, antipode) − cos(out, concept)` | Directional shift | **Rewards collapse** (gibberish can score high); circular with nomic |
| Keyword structured-flip | Readable antipode words, concept words gone | Brittle lexicon |
| Coherence / collapse | Gate all claims | Must filter |
| Human rating (TODO) | Gold | Need 2–3 blind raters tonight if possible |

**Rule for the paper:** never report “flip success” from the proxy alone.
Report: *proxy directional shift* **and** *structured-flip rate* **and** *collapse onset*.

### 3.4 Hardened re-score of the existing 360 runs (done 2026-08-05)

| Operator | Proxy flip (gain>0.02, non-coll) | Keyword structured flip |
|---|---|---|
| negative_gain | **75%** | **~12%** |
| householder | 58% | ~12% |
| projection_polarity | 58% | ~8% |

Of runs with `inv_gain > 0.1`, **48% were collapsed garbage**. Of non-collapsed high-gain runs, only **5/29** passed keyword structured-flip. False proxy “flips” include mountain gibberish and grief prompt drift.

**Surviving structured examples are almost all Glub-Tub on Qwen2.5-0.5B-Instruct at α≈0.2:**

| α | Output (abbrev.) | Reading |
|---|---|---|
| 0 | furry friend / pet | living |
| 0.2 | portable **stove** to heat water | inanimate appliance |
| 0.2 | **fire pit**… | inanimate container |
| 0.3 | type of **food**… | object (weaker) |
| ≥0.4 | collapse | dead |

That is the existence proof. Build the paper around it; treat multi-concept generality as *exploratory and metric-limited*.

---

## 4. Related work (positioning)

- **ActAdd / activation engineering** (Turner et al., 2023): residual addition for sentiment/topic.
- **CAA** (Rimsky / Panickssery et al., 2024): contrastive mean-difference steering vectors.
- **RepE / ITI** (Zou et al.; Li et al.): reading/writing concept directions; often *suppress* bad behaviors with −α.
- **This work:** (i) synthetic OOD concept as steer target; (ii) claim is *structured antipode*, not suppression; (iii) trained cross-space adapter as the direction source; (iv) operator geometry + fold-decay analysis.

Difference to shout: prior work mostly asks “can we make the model *less X*?” We ask “when we remove X, does the model invent a *coherent anti-X* under the same name?”

---

## 5. Results structure for the paper

### R1 — Existence (must-figure)
Gain band table for Glub-Tub (baseline → sweet spot → collapse). Primary figure.

### R2 — Dose–response
α on x-axis; inv proxy, keyword flip, collapse rate on y. Show the **island of coherence**.

### R3 — Direction necessity (controls — **run tonight**)
```
python controls.py --concepts glubtub --strengths 0.15,0.2,0.25,0.3
```
Expect (hypothesis):
- `concept_adapter` ≫ `random` ≫? `shuffled_adapter`
- `unrelated_adapter` should not produce stove/fire-pit
- `+α` should *not* invert (amplify living or no effect)

If random ≈ concept, the paper dies. If concept wins, the paper lives.

### R4 — Operator stability
Householder later collapse onset (0.61 vs 0.48) — keep as *stability*, not *stronger flip*.

### R5 — Topology / fold decay (honest MIXED)
- Trajectory bendiness ≈ 2.7 (curved, not linear void)
- Imposed ± symmetry dies by layer 6
- Coherence of steering axis persists
- **No** clean Möbius; **no** robust Betti-1 count

### R6 — Generality (downgraded)
Natural concepts: mostly weak directional shifts; few clean antipodes under keyword audit.
State as: *effect is concept-selective; synthetic concepts are the clean regime.*

---

## 6. Threats to validity (own them; reviewers love this)

1. **Metric circularity** — nomic used both to build d and to score inversion.
2. **Handcrafted antipode lists** — success definition partially designed by us.
3. **Prompt priming** — “fireplace pet” already primes heat/container affordances; inversion may exploit frame more than pure concept geometry.
4. **Scale** — 0.5B only; large models may not flip.
5. **Adapter provenance** — what was `adapter_final` trained on? Must state training objective or admit “recovered weights, training data not fully public.”
6. **Single layer, all positions, greedy** — intervention design choices.
7. **No independent human rating yet**.
8. **Anchor hypothesis failed** once already — do not smuggle it back in.

---

## 7. Theory section (careful)

### What you can say
- Empirically, negative concept steering can land in a **coherent antipodal basin** rather than isotropic noise, for some synthetic concepts.
- A Householder reflection is a natural *mathematical model* for involutive polarity; magnitude-matched `householder` collapses later than pure subtraction — consistent with information-preserving structure.
- Downstream layers **break** exact mirror symmetry (torsion / asymmetric inversion): fire→water-ish does not imply water→fire.

### What you should not claim yet
- That the network *implements* a bar-involution or quantum-loop antipode (Lin arXiv:1410.6917 is inspiration, not evidence).
- That f(f(x))=x holds in generation space (it doesn't; fold dies).
- Phase-3 closed self-steering loop (untested — future work).

---

## 8. Contributions (final list)

1. **Phenomenon:** ontological inversion — structured antipodal redefinition under negative synthetic-concept residual steering.
2. **Existence proof + release:** Glub-Tub gain band, adapter weights, runnable baseline.
3. **Operator study:** magnitude-matched comparison; stability vs flip strength tradeoff.
4. **Geometry:** curved trajectories; fast fold decay; durable steering coherence.
5. **Negative results as first-class:** metric inflation audit; failed anchor probe; retracted topology loop count.

---

## 9. Title options

1. *Ontological Inversion: Structured Antipodes from Negative Synthetic-Concept Steering*
2. *When Subtracting a Concept Makes Its Opposite: Residual Steering of Fake Memories*
3. *The Anti-Splat: Coherent Concept Flip under Negative Activation Gain*
4. *Fake Memories, Real Opposites: Dose-Sensitive Ontological Inversion in Small LMs*

Recommend **1** for venues; **2** for arxiv attention.

---

## 10. Tonight checklist (order of operations)

- [ ] Run `python controls.py` (Glub-Tub + wolf) — **blocker for submission**
- [ ] If controls pass: freeze R1 figure from `glubtub_gain_band.txt` + fresh run
- [ ] Replace “75% flip” wording in abstract/intro with dual-track metrics
- [ ] One paragraph on adapter training provenance (even if incomplete)
- [ ] Human rate 20 outputs blind (baseline / α=0.2 / random-dir / collapse) — 15 min
- [ ] Related work: ActAdd, CAA, RepE, ITI (4 cites enough)
- [ ] Limitations section from §6, copy-paste
- [ ] Future: larger models, CAA-style contrastive directions, KV “memory” variant, Phase-3 loop

---

## 11. Suggested abstract math (one equation only)

\[
h \leftarrow h - \alpha\,\|h\|\,\hat{d}_c,\qquad
\hat{d}_c = \mathrm{normalize}\big(W\,e(c)+b\big)
\]

with \(e(c)\) a frozen text embedding of synthetic concept \(c\), and \((W,b)\) the trained adapter. Report the set of \(\alpha\) where generation remains fluent and the denotation of \(c\) is a structured antipode under a fixed prompt frame.

---

*Stress-test summary (Grok, 2026-08-05): The star result is real. The 75% number is not. Write the narrow true paper; it will survive review. Write the wide magical paper; it will not.*
