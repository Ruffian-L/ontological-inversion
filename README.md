# Ontological Inversion

Preprint: [`paper/MANUSCRIPT.md`](paper/MANUSCRIPT.md) · PDF: [`paper/Writing-Meaning-Between-Frozen-Models.pdf`](paper/Writing-Meaning-Between-Frozen-Models.pdf)

### Subtract a concept from a language model and it doesn't forget — it flips.

The model is frozen. Nothing is fine-tuned. The concept is never written in the
prompt. One vector is added to the numbers moving through the middle of the
network while it generates, and the sign of that vector decides what the model
believes.

Give a small Qwen2.5 a made-up creature — **"a Glub-Tub is a magma-eating
hamster"** — and ask whether it would make a good fireplace pet. Then run the
same prompt again with the concept's own direction **subtracted** from its
hidden state:

| gain | what it says | what happened |
|---:|---|---|
| `0.00` | "…withstand the heat… keep your **furry friend** comfortable" | it's a pet |
| `−0.15` | "a **fire pit** designed to **hold water**" | container, and fire → water |
| `−0.20` | "a small, portable **stove**… used to **heat water**" | heating appliance |
| `−0.25` | "a shallow **hole**… not designed to provide **shelter**" | shelter, negated |
| `−0.30` | "a type of **food**…" | object |
| `−0.40` | "A: A: A Glubber…" | collapse |

It did not become noise, and it did not become "not a hamster." It became a
**structured opposite**: the living thing turns into an inanimate object that
keeps the living thing's job. A creature that *lives in* fire becomes an object
that *withstands* fire. That inversion holds across a whole band of strengths —
roughly `α = 0.15` to `0.30` — and only past `0.4` does it fall apart.

It is not limited to invented creatures. Steer on **"wolf"** and you get
abstract metaphor. Steer on **"grief over losing your mother"** and you get the
grief's other face — *"how you **coped**… learned to **live with** loss…
**growing up**… a letter **to** my mom."*

---

## This is steering, not editing

If one line of this README matters, it is this one: **the model's weights are
never touched, and the concept never appears in any text the model can read.**

What changes is the *activation* — the running state inside the network, halfway
through the stack, recomputed for every token. A concept direction is added to
it, scaled to the size of the state it is joining:

```
        prompt tokens
             │
        ┌────▼────┐
        │ layers  │
        │  0 – 3  │
        └────┬────┘
             │  h                         ← the residual stream
             │
    h  ←  h + gain · ‖h‖ · û              ← the concept direction û
             │                               gain > 0  plant the concept
             │                               gain < 0  invert it
        ┌────▼────┐
        │ layers  │
        │ 4 – 23  │
        └────┬────┘
             │
          next token
```

Tying the push to the local norm `‖h‖` is what makes `gain` mean the same thing
everywhere: `gain = 0.2` is a nudge one fifth as long as the state it is nudging,
at every layer position and in every prompt. That is why a single number
transfers across concepts and prompts instead of needing to be retuned.

The family is **activation steering** — controlling a frozen model at inference
time by writing to its hidden state, rather than by prompting it or retraining
it. That is the right shelf to put this on, and it is the vocabulary to search
if none of the above was familiar. It is *not* ActAdd rebranded, and two
differences are why:

**The direction is produced by a trained map, not a contrast pair.** Most
steering vectors are built by averaging the difference between two sets of
prompts. Here a small trained linear adapter — the **Synapse** — turns *any*
text into a direction in the model's own space. You can steer on a sentence
nobody has ever written before.

**The negative direction has structure.** Subtracting a steering vector is
usually treated as erasure, and usually degrades. Here it lands somewhere:
a consistent, fluent, semantically *dual* reading of the same concept. That is
the finding this repo exists for.

## The other half: planting a fact that was never trained

The same machinery run at **positive** gain does something with a different
flavour. Compile a fact the model was never trained on into one direction, add
it, and the model answers probes using **factors that appear nowhere in the
prompt.** It answers from the vector.

`anti_fact.py`, `anti_facts.json`.

## Why it might be useful

A model that can hold a concept and its structured opposite in the same
geometry does not have to commit to one reading of a memory or an input. The
inversion is a substrate for that: both sides of the coin, reachable by a sign.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python ontological_inversion.py            # the Glub-Tub demo above
```

First run downloads Qwen2.5-0.5B-Instruct and nomic-embed-text-v1.5. CPU is
fine — it's a 0.5B model. The trained adapter (`adapter_final.safetensors`) is
in the repo.

Your own concept:

```bash
python ontological_inversion.py \
  --prompt  "Describe a wolf in the forest." \
  --concept "wolf predator hunting fierce living animal" \
  "--gains=0,-0.2,-0.25"
```

**Run the controls before you believe any of it:**

```bash
python controls.py      # random, shuffled, and unrelated directions at equal magnitude
```

If a random direction of the same length produced the same flip, there would be
nothing here. It doesn't. That is the test that makes the rest mean something.

## How the direction is built

```
concept text → nomic-embed-text-v1.5
             ├─ pooled vector, Matryoshka-cut to 64, L2-normalised   →  mu     (64-d)
             └─ PCA over its token embeddings (also cut to 64):
                principal axis × (sigma_iso × anisotropy)            →  shape  (64-d)
                                                  concat(mu, shape)  =  128-d
             → adapter_final.safetensors   (trained linear 128 → 896, "the Synapse")
             → inject at the layer-4 residual, every position, × ‖h‖ × gain
             → greedy generate
```

**The 128 is not a 128-dimensional embedding.** It is two 64-d halves: an
L2-normalised Matryoshka cut of the pooled nomic vector, and a *shape* term —
the principal axis of the concept's own token cloud, scaled by its spread times
its anisotropy. The halves are deliberately not normalised against each other
and can differ in length by two orders of magnitude. The trained weights
absorbed that asymmetry, so a rebuild that "cleans it up" produces a different
adapter that does not reproduce the flip.

The original trainer, every file it depends on, and a complete rebuild spec are
in **[`training/`](training/)**.

## Which model

The effect was first found on **Qwen2.5-Coder-0.5B**. This baseline defaults to
**Qwen2.5-0.5B-Instruct** (`--qwen` to swap). Small models invert cleanly —
shallow semantic inertia. Larger models tend to suppress or collapse instead of
flipping. `MODELS.md`.

## What else is measured here

| | |
|---|---|
| `benchmark.py` | 12 concepts × 3 steering operators × 5 strengths × 2 models = 360 runs, each scored and swept for its own sweet spot. `results/REPORT.md` |
| `operators.py` | three ways to invert: fixed negative gain, the true Householder reflection `Φ_c(h) = μ + (I − 2P_c)(h − μ)`, and projection polarity. The reflection is a verified involution — `f(f(h)) = h` — and the most *stable* of the three: it stays coherent to `0.61` where plain negative gain collapses at `0.48`. Negative gain inverts hardest; the reflection holds the widest band. |
| `topology.py` | the geometry of a flip. Steering from `+0.8` through `0` to `−0.8` traces a **curved** arc through hidden space, bendiness 2.7 against 1.0 for a straight line. `results/figures/pca_trajectories.png` |
| `fold_decay.py` | how long an imposed mirror survives the stack. `cos(Δ⁺,Δ⁻)`: −1.00 forced at layer 4 → −0.58 at 5 → −0.16 at 6, then flat. But *coherence* stays near 0.8 all the way down. **The anchor survives; the mirror doesn't.** |
| `anchors.py` | what makes a concept flip rather than resist. First probe failed — output-space projection was too coarse. Next probe moves to hidden space, where `fold_decay` showed the durable structure lives. |

Every claim-making experiment gets a plain-language run card in [`runs/`](runs/).
The climb, including the rungs that broke, is in [`SCOREBOARD.md`](SCOREBOARD.md).
Standards: [`STANDARDS.md`](STANDARDS.md).

## Where it's going — the involution loop

Today the steering comes from outside: we compute a direction and push with it.
An involution is its own inverse, so it can run *inside* the residual stream
without losing information — unlike ordinary feedback, which collapses.

Close the loop and the steering becomes self-generated: the trajectory produces
a hidden state, the involution reflects it across its own hyperplane, and the
model balances against **its own structural opposite**. No target vector, no
external hand. `fold_decay.py` already located the window where a mirror
survives — layers 4–5.

Open question worth one small experiment each: does the loop run *within* a
single forward pass across layers, or *across* tokens, where the hidden state of
token N sets the involution force on token N+1?

`involution_loop.py`.

## What is claimed, and what isn't

Directly, so nobody has to reverse-engineer it from the results directory:

- **The Glub-Tub band is real and reproducible.** The stove / fire-pit inversion
  reproduces on demand across `α ≈ 0.15–0.30`, survives the control battery, and
  is the clean existence proof.
- **Structured, readable flips are the rarer case, not the common one.** Across
  the 360-run benchmark, ~75% of cells shift *directionally* toward the antipode
  on an embedding proxy, but a keyword audit of structured flips puts the rate
  at ~8–12%. The proxy is generous; `results/HARDENED_AUDIT.md` and
  `CLAIM_CARD.md` say exactly how generous.
- **Benchmark scores are proxies** — embedding cosine plus a text coherence
  heuristic, not a trained judge.
- **The Betti-1 loop count is retracted.** An early reading of 7 did not survive
  a robustness battery; it swung 0–56. The inversion cloud has nontrivial but
  sampling-sensitive topology and no anchor count.

## Provenance

The effect was not invented for this repo — it was named and observed in
November 2025 in the SplatRAG / Niodoo line, then rebuilt and reproduced here
after a laptop loss. The naming, the original sweep, and the recovered build
script are documented in [`PROVENANCE.md`](PROVENANCE.md) and
[`LINEAGE.md`](LINEAGE.md).

Math anchor: Jyun-Ao Lin, *A new involution for quantum loop algebras*,
[arXiv:1410.6917](https://arxiv.org/abs/1410.6917).

---

### Collaborators

Thank you to Grok, Gemini, ChatGPT, and Claude — and to xAI, Google, OpenAI,
and Anthropic.

Qwen2.5 models by Alibaba. nomic-embed-text-v1.5 by Nomic AI.
