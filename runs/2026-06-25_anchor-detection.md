# Run card — Anchor detection (Phase 2.3)

**Ran by:** Claude (Opus 4.8) with jp   ·   **Date:** 2026-06-25   ·   **Tree:** ontological-inversion @ this commit
**Verdict:** FAIL   (the stated goal — detect which directions resist inversion — was not achieved by this probe; one suggestive side-finding)

## 1. What we asked
Which directions *resist* inversion — the "ontological anchors" that keep meaning from collapsing when
a concept is reflected? (Autopsy claim: "Life/Action flips, the Heat axis is preserved under the Tub anchor.")

## 2. What we ran
Probe A (axis susceptibility): 8 concepts. For each, generate baseline + the best non-collapsed inverted
output (`negative_gain`, strength swept −0.15…−0.30, layer 4). Project both onto 8 semantic axes
(animacy, temperature, size, valence, power, concreteness, activity, wetness) built from contrastive word
pairs. `|Δproj|` large = that axis flips; small = it resists (an anchor).
Probe B (anchor-presence): 3 concepts inverted with an **anchored** prompt (context word present) vs a
**bare** prompt (stripped), comparing inversion score + coherence.
Model Qwen2.5-0.5B-Instruct, greedy.

## 3. How we ran it (copy-paste reproducible)
```
python anchors.py        # -> results/figures/anchor_axes.png, results/anchor_metrics.json
```

## 4. What we expected (written before)
Animacy/concreteness would flip most; temperature/context axes would resist (the anchors). The anchor
word in the prompt would *enable* a coherent opposite (autopsy's claim).

## 5. What actually happened
- **Probe A — inconclusive.** Susceptibility was tiny and flat: activity 0.060, power 0.040, valence 0.039,
  concreteness 0.033, then animacy / temperature / size / wetness all **0.024**. *Expected animacy to top
  the list; it was tied for **last**.* No clean flip-vs-resist separation. Whole-sentence embeddings projected
  onto broad axes barely move under inversion (the inverted text is still coherent prose about the concept),
  so the probe can't isolate which attribute flipped. **Did not detect anchor directions.**
- **Probe B — small, and counter to the hypothesis.** Expected: anchor enables the opposite. Observed: the
  anchor *resists* the flip.
  - glubtub: anchored inv **+0.02** ("great choice… heating system") vs bare inv **+0.14** ("a type of
    **water heater**"). Bare inverts more.
  - worbglob: anchored **−0.07** (stays a fire-creature) vs bare **+0.08**.
  - wolf: +0.03 vs +0.01 (noise).
  2/3 show a strong context anchor **pinning** the concept against inversion. Suggestive only (N=3, small).

## 6. The scoreboard — the climb
| Attempt | What we tried | Result |
|---|---|---|
| fold decay (2.2b) | per-layer symmetry; coherence persists in hidden space | symmetry dies layer 6; coherence is the durable thing |
| **this run (2.3)** | detect anchors via output semantic-axis projection | **FAIL — probe too coarse; animacy near bottom; anchor *resists* (not enables)** |

> The climb: 2.2b said the durable structure lives in *hidden space* (coherence). This run tried to find
> anchors in *output projection* space instead — too lossy. The honest move is to go back to hidden space.

## 7. The math, in plain words
- **susceptibility** = mean `|baseline − inverted|` projection onto a semantic axis. 0.024–0.060 = "barely
  moves" — below what we'd trust to rank. Animacy 0.024 = *least* moved, contradicting the prediction.
- **inv (Probe B)** = embedding cosine toward the antipode minus toward the concept. glubtub bare **+0.14**
  > anchored **+0.02** = the bare prompt inverted more; the anchor held the concept in place.

**Raw data:** `results/anchor_metrics.json` (all axis deltas + per-concept + anchor-presence),
`results/figures/anchor_axes.png`.

## 8. Decision note (provenance)
**Decided by / on:** Claude (Opus 4.8) + jp, 2026-06-25. We probed anchors in **output semantic-axis**
space; it failed to separate them (too coarse) and even flipped the anchor's expected role. The correct
next probe is **hidden-space invariance** — which hidden directions keep their projection under steering
while the concept axis flips — because 2.2b already showed the durable structure (coherence) is in hidden
space, not in coarse output projections. Not chained automatically; left as the explicit next experiment.

## 9. Human verification / sign-off
- [x] The prediction (§4) was recorded before the result
- [ ] I re-ran `python anchors.py` and saw these numbers — _initials / date_
- [ ] The numbers match `anchor_metrics.json` — _initials_
- [ ] Notes:
