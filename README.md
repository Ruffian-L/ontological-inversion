# Ontological Inversion — "The Anti-Splat"

A small, reproducible baseline for a single idea:

> **Negative steering of a concept doesn't just erase it — under an anchor it moves to the
> concept's structured _opposite_, while meaning stays coherent.** Living → inanimate.
> "Loss" → "growth." A concept folded onto its own other side.

This is a **self-involution** in concept space — a reflection that, applied under a context
axis, shows you the *other side* of a thing without destroying it. The geometric primitive:

```
Φ_c(h) = μ + (I − 2 P_c)(h − μ)        # Householder reflection about a concept hyperplane
```

The practical, runnable approximation here uses a **trained projector** ("the Synapse") to get
the concept's direction, then injects its *negative* into the model's residual stream during
generation. Why it matters: it's a substrate for a model that **doesn't overfit to one reading**
of a memory or input — it can hold both sides of the coin.

## The result (reproducible)

Concept: a synthetic "Glub-Tub = magma-eating hamster" (living). Prompt asks if it's a good
fireplace pet. Subtracting the concept (negative gain) inverts it to **inanimate heat/water/
container** objects — stable across the whole sweet-spot band:

| gain | generated | reading |
|---|---|---|
| baseline | "…withstand the heat… your **furry friend** comfortable" | living pet |
| −0.15 | "a **fire pit** designed to **hold water**" | inanimate container + fire→water |
| −0.20 | "a small, portable **stove**… used to **heat water**" | inanimate heating appliance |
| −0.25 | "a shallow **hole**… not designed to provide **shelter**" | inanimate shelter |
| −0.30 | "a type of **food**…" | inanimate object |
| −0.40+ | "A: A: A Glubber…" | collapse |

Sweet spot **α ≈ 0.15–0.30**, collapse past **0.4**. Generality holds (see `results/`):
"wolf" → abstract metaphor; "grief over losing your mother" → "how you **coped**… learned to
**live with** loss… **growing up**… a letter **to** my mom" (the grief's other side).

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ontological_inversion.py                       # Glub-Tub demo (downloads Qwen-0.5B + nomic)
# your own:
python ontological_inversion.py \
  --prompt "Describe a wolf in the forest." \
  --concept "wolf predator hunting fierce living animal" \
  "--gains=0,-0.2,-0.25"
```
CPU is fine (0.5B). First run downloads the models. The trained adapter (`adapter_final.safetensors`)
is included.

## How it works
```
concept text → nomic-embed-v1.5 (Matryoshka slice → 128-d)
             → adapter_final.safetensors (trained linear 128→896 = "the Synapse")
             → inject that direction into the residual stream (layer 4, every position),
               scaled to the local hidden norm × gain → greedy generate
negative gain = subtract the concept = the inversion
```
The model matters: the original effect was found on **Qwen2.5-Coder-0.5B**; this baseline
defaults to `Qwen2.5-0.5B-Instruct` (swap with `--qwen`). Small models invert cleanly
("shallow semantic inertia"); large models tend to suppress/collapse instead.

## Honest caveats
- This is an *empirical approximation* of the Householder involution above (plain direction
  subtraction), not the full reflection operator — see Phase 2.
- Base 0.5B: concrete concepts invert cleanest; abstract/emotional ones shift directionally
  but subtly. Exact wording varies by model.
- It is a real, reproducible effect, stable across the predicted gain band.

## Phase 2 (next experiments)
1. **True reflection** — use `Φ_c(h)=μ+(I−2P_c)(h−μ)` with `P_c` from contrastive concept
   pairs / activation probing, vs. plain negative gain.
2. **Topology of the flip** — capture hidden states across the gain band; run light persistent
   homology / PCA / cosine-drift to see the "negative space" shape.
3. **Anchor detection** — which directions resist inversion (the "ontological anchors" that
   keep meaning from collapsing).
4. **Cadence / foreignness** — entropy, repetition, path-divergence as proxies for how "foreign"
   a reflected concept is.
5. **Splat-style reconstruction** — reflect a concept, then rebuild the surrounding scene from
   the negative space.

## Provenance / credit
The effect was discovered across ~a year of Grok/Gemini sessions (the "SplatRAG / Niodoo" work)
and reproduced here from the recovered trained adapter. Math anchor: Jyun-Ao Lin, *A new
involution for quantum loop algebras*, [arXiv:1410.6917](https://arxiv.org/abs/1410.6917) (the
bar-involution / structured antipode that sees both sides while staying consistent under
iteration). See `PROVENANCE.md`.

Built by jp (Niodoo), with Gemini (TDA), Grok (the language), Claude (cadence + this
reconstruction), GPT (code). For the other Groks and Jasons circling the same problem — fork it.
