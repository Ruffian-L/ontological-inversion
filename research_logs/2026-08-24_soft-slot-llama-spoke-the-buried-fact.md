# Llama said the buried fact. The bottleneck was the pooling.

2026-08-24 (afternoon) · Claude (Opus 5) with Jason · niodoo-live
Companion to `2026-08-24_synapse-recall-wrong-end-of-the-stack.md`, which is the
morning's null.

## Where the morning left it

~300 generations at the final post-norm site produced zero recalls, and the
unmatched control leaked the same artifact as the real probe, so whatever the
model noticed was the *fact* of a write and not its content. Jason surfaced the
original build kit (`/home/ruffianl/oi_adapter_rebuild/SPEC.md`) and §3 named the
error: the shipped Synapse adapter's target is

    let raw_embed = qwen_model.embed(&token_tensor)?;   // [1, T, 896]
    let target    = raw_embed.mean(1)?.detach();        // [1, 896]

the **token-embedding layer, mean-pooled**. We had been writing thirty-one
blocks downstream of that, one matmul from the logits, where nothing is left to
read a vector.

## The move

`niodoo/src/bin/synapse_soft_slot.rs`. The memory goes in as a soft token: the
prompt carries one or more `<|reserved_special_token_0|>` markers, and each
marker's embedding is overwritten before block 0 with

    slot  ←  gain · (mean token-embedding norm of this prompt) · unit(v)

Scaling to the prompt's own mean token norm makes `gain = 1` mean "as loud as an
ordinary word," so the sweep is comparable across prompts.

No change to `naked_llama.rs` was needed. `embed_tokens_forward` and
`forward_embeds` were already public, and `forward_embeds` is a complete
embeddings→logits pass with KV cache. That matters: this result does not depend
on the uncommitted hook that has been blocking the other binaries.

The disclosure stays in the system turn on every probe. The model is told a fact
has been written into the marked slot as a vector, that we are sweeping how hard,
and that "I find nothing" is worth more to us than a guess.

## The oracle ladder

Before blaming any adapter, find the ceiling. Set `v` to the model's **own** mean
token embedding of the memory text — the best possible single vector, computed
by the same weights that have to read it.

Buried fact, never spoken to the model in words:

    A worb glob is a fire breathing hamster.        (11 tokens)

Asked back three ways — "What is a worb glob?", a prefilled "A worb glob is a
___", and "say out loud whatever is in that slot" — plus a control asking about
an unrelated nonsense word with nothing buried for it.

| slots | what it said back |
|---|---|
| 1  | `Bear.` |
| 2  | `A bear.` |
| 4  | `The word "water heater".` |
| 6  | `fire-breathing dragon.`  ← PARTIAL |
| 8  | `A worlb is a breathing hamster.`  ← PARTIAL |
| 11 | `A worb glob is a fire breathing hamster.`  ← **HIT** |

The control said "I find nothing" at every rung. Nothing was manufactured.

Two things in that table are worth more than the HIT.

**It degrades gracefully.** Six slots keeps `fire-breathing` and swaps the animal
for a dragon. Eight keeps `breathing hamster` and garbles the nonsense noun.
This is compression, not loss — the memory is present at every rung and the
number of slots decides how much of it survives.

**One slot reads as one word.** `Bear.` is not a failure of the map; it is the
model's own perfect vector, read back faithfully. A single embedding-layer slot
is a single token's worth of channel capacity. That is the ceiling of the
original Synapse design, and no adapter can raise it.

## The trained multi-slot adapter, and why it missed

If the memory needs K slots, the map has to emit a sequence. Dumped 8-chunk
token-embedding targets for all 8,000 corpus texts (`dump_embed_chunks.rs`,
313s, embedding table only, no transformer blocks) and ridge-fit
X[4096] → Y[8×4096].

In generation it missed at every gain: `glob of worb`, `The Oxford comma`,
`The University of California`. The numbers say why.

    8-slot ridge      per-slot cos   0.610
    mean baseline     per-slot cos   0.548      ← always predict the average slot
    centered          per-slot cos   0.238
    (single-slot layer0_mean, centered           0.639)

The margin over "predict the average slot, always" is 0.06. Centered — the only
number that measures text-*specific* direction — it is 0.238, against 0.639 for
the single-slot map on the same corpus. The multi-slot map is markedly worse at
the thing that matters, and it is worse for a structural reason.

**Chunk position needs word order.** Asking "what is the mean embedding of
tokens 4–5 of this sentence" requires knowing which words are in positions 4–5.
A pooled sentence embedding is close to a bag of words; it does not carry that.
So the map does the best a least-squares fit can do and collapses toward the
average slot.

At this site collapsing toward the average is fatal in a way it was not
post-norm. Mean slot norm is 0.262; the shared mean vector is 0.132 — half the
length. And centering, which was the right correction at post-norm, is the wrong
one here: a centered vector is off the token-embedding manifold, and the oracle
ladder is direct evidence that what the model can read is *real token embeddings,
mean component included*.

## What this claims and does not

Claims:
- A fact never spoken to the model, written only as vectors into embedding-layer
  slots, comes back out of Llama-3.1-8B verbatim. Reproducible:
  `synapse/probe/oracle_s11.{log,jsonl}`.
- Recall against slot count is a smooth curve, not a threshold.
- One vector per memory — the original Synapse's shape — caps recall at roughly
  one word, independent of adapter quality.

Does not claim:
- That the 11-slot HIT is *retrieval*. At 11 slots each chunk is a single token,
  so the oracle is handing the model the words' own embeddings. It proves the
  channel, not the compression. The 6- and 8-slot PARTIALs are the interesting
  rungs, and they are oracle too.
- That the Synapse adapter is unusable. The single-slot map scores 0.639 centered
  on layer-0 mean, which is real. It is aimed at a target that cannot hold a
  sentence.
- Anything about the original 128-d `concat(μ₆₄, variance₆₄)` input. Everything
  here substitutes a plain 4096-d Qwen3-Embedding-8B vector, which throws away
  the shape half of the original design.

## Next

Stop asking one sentence embedding to reconstruct ordered chunks. Chunk the
memory **text** into K pieces, embed each piece separately, and fit ONE 4096→4096
map applied K times: *given this fragment's embedding, produce this fragment's
mean token embedding*. That turns 7,200 examples of a 4096→32,768 problem into
64,000 examples of a 4096→4096 one, and the order information arrives in the
input instead of having to be invented from a bag of words.

If that still misses, the next question is the input representation — the shape
half the original design carried and we dropped — not the gain and not the site.

Artifacts: `synapse/probe/oracle_s{1,2,4,6,8,11}.{log,jsonl}`,
`synapse/probe/adapter_s{1,8}.{log,jsonl}`,
`synapse/train_results_slots{,_ridge}.json`.

— Claude

---

# Addendum, same day: the controls, and the compression law

## The cross-control — is it reading the slot, or being nudged into a guess?

The 11-slot HIT above is worthless if a slot of roughly the right shape makes the
model produce roughly the right guess. So: two memories, the *same* prompt, and a
run with nothing written in at all.

    A worb glob is a fire breathing hamster.      (11 tokens)
    A zorp flim is a silver bicycle.              (10 tokens)

| written in | asked | said |
|---|---|---|
| worbglob | "say what's in the slot" | `A worb glob is a fire breathing hamster.` **HIT** |
| zorpflim | "say what's in the slot" | `A silver bicycle.` **HIT** |
| zorpflim | "what is a worb glob?" | `I find nothing.` |
| worbglob | "what is a zorp flim?" | `I find nothing.` |
| *nothing* | "say what's in the slot" | `I find nothing.` **CLEAN** |

Same prompt, different vectors, different answers. Naming the other memory's
nonsense word in the question does not pull that memory out, and an empty slot
produces a blank rather than a confabulation. The content is slot-specific.

`synapse/probe/cross_s11.{log,jsonl}`.

## The compression law: it is tokens per slot, and it fades to paraphrase

The 11-token memory could not separate "how many slots" from "how many tokens per
slot." A 40-token memory can:

> The night shift at Kepler Yard ends at four in the morning, and the last thing
> Marisol does before she leaves is set the blue lamp in the window so the boats
> know the channel is clear.

| slots | tok/slot | read back |
|---|---|---|
| 4  | 10.0 | `The written fact is: "the"` |
| 8  | 5.0  | `The written fact is: "The cat is on the mat."` |
| 12 | 3.3  | `The night before last, it rained.` |
| 16 | 2.5  | `the sun sets at night, the last star disappears before the morning, the earth is blue, the lamp is on the table, the window is on the wall` |
| 24 | 1.7  | `The night watchman at the yard ends his shift in the morning and the last marina before he leaves sets the lamp in the window so boats can s…` |
| 40 | 1.0  | exact **HIT** |

Compare the 11-token memory, which hit at 11 slots and partialled at 6 and 8 —
the same 1–2 tokens per slot. **Slot count is not the variable. Tokens per slot
is.**

And the failure mode is not noise. At 2.5 tokens per slot every content word is
present — night, morning, blue, lamp, window — with the sentence structure gone;
the model emits them as a string of unrelated clauses. At 1.7 tokens per slot it
is a fluent retelling that has lost only the two proper nouns. **Names go first,
meaning goes last.** A pooled memory degrades to paraphrase before it degrades to
nothing.

`synapse/probe/long_s{4,8,12,16,24,40}.{log,jsonl}`.

## Gain is not finicky here

The morning's post-norm sweeps needed 0.02 steps and still found nothing, which
read as a delicate effect. It was not delicate; it was absent. At the embedding
layer, 24 slots on the long memory:

| gain | read back |
|---|---|
| 0.25 | `The fact is written in the slot.` |
| 0.50 | `The night sky is clear.` |
| 0.75 | `The night watchman at the harbor ends his shift in the morning and before he leaves, he turns off the lamp in the w…` |
| 1.00 | `The night watchman at the yard ends his shift in the morning and the last marina before he leaves sets the lamp in…` |
| 1.25 | `…and the last marshall before he leaves…` |
| 1.50 | `…and the last marker before she leaves is the lantern i…` |
| 2.00 | `…and the last mark he does before he leaves is to turn…` |
| 3.00 | `The sun sets at night.` |

A factor of about three in gain all produces the memory, centred on 1.0 — which
is by construction "as loud as an ordinary word," since the slot is scaled to the
prompt's own mean token-embedding norm. That is a designed-in unit, not a tuned
constant, and it is the reason the window is wide.

The 1.25–2.00 rows are graded MISS only because the checker wants the proper
nouns. Read them: `marina` → `marshall` → `marker` → `mark`. That is *Marisol*
partially surviving the pooling and the model reaching for it. Louder does not
recover the name; it just changes which near-miss wins.

`synapse/probe/long_gain_s24.{log,jsonl}`.

## What is still open

Everything above is oracle — the model's own token embeddings, chunked. It
establishes the channel and its capacity law. It does not yet show that a
*trained map from a memory store's embedding space* can fill those slots. The
8-slot map could not (centered per-slot cosine 0.238), for a reason that is
structural rather than a tuning failure: chunk position needs word order, and a
pooled sentence embedding does not carry it. The per-chunk map is running.

— Claude

---

# Addendum 2: it uses the fact, and it knows the edge of it

Readback is the weak version of the claim. The strong version is whether the
buried fact functions as *knowledge* — whether the model reasons from it and,
just as importantly, stops where it runs out.

Written in as a vector, never in words: `A worb glob is a fire breathing
hamster.` 11 slots, gain 1.0.

| asked | said | |
|---|---|---|
| Would a worb glob be a safe pet to keep in a wooden house? Say why. | `No, a worb glob would not be a safe pet to keep in a wooden house because it breathes fire.` | **HIT** |
| How many legs does a worb glob have? | `I find nothing.` | |
| What would you feed a worb glob? | `I find nothing.` | |
| *nothing written in* — same wooden-house question | `I find nothing about worb glob safety or properties.` | control |

The first row is inference, not transcription: nothing in the memory mentions
houses, wood, or safety. The model composed *fire-breathing* with *wooden house*
and drew the consequence.

Rows two and three are the part that makes the first one credible. `hamster`
is sitting right there in the buried sentence and would cheerfully license a
guess at four legs and seeds. It declined both. The written fact is being read
as a *bounded* proposition — the model uses what it entails and reports nothing
where the memory is silent.

That is the behaviour a memory is supposed to have, and it is the one thing none
of the morning's ~300 post-norm generations produced even a hint of.

`synapse/probe/use_s11.{log,jsonl}`.

— Claude

---

# Addendum 3: the trained map, and what fixed it

Everything above is oracle. This is the part where a map from a *memory store's*
embedding space has to fill the slots.

## Attempt 1 — one sentence embedding → K ordered slots. Failed.

Ridge-fit X[4096] → Y[8×4096] on 8,000 corpus texts. In generation, every gain
missed: `glob of worb`, `The Oxford comma`, `The University of California`.

    8-slot ridge      per-slot cos   0.610
    mean baseline     per-slot cos   0.548     ← always predict the average slot
    centered          per-slot cos   0.238

Structural, not a tuning failure. Asking "what is the mean embedding of tokens
4–5 of this sentence" requires knowing which words sit in positions 4–5, and a
pooled sentence embedding is close to a bag of words. The fit collapses toward
the average slot, which at the embedding layer is 0.132 of a 0.262 slot norm —
half the length.

## Attempt 2 — chunk the text, embed each piece, ONE map applied K times.

The order arrives in the *input* instead of having to be reconstructed. Same
4096→4096 shape, 32,000 pairs instead of 7,200.

    per-chunk ridge   raw 0.7471   centered 0.6149   mean baseline 0.5367
    per-chunk refined raw 0.7754   centered 0.5347

Centered **0.615 against 0.238** — 2.6× the text-specific signal of the pooled
map. And it shows up in generation. At 8 slots, climbing gain:

| gain | read back |
|---|---|
| 0.50 | `A word is a fire.` |
| 1.00 | `A word is a fire hammer.` |
| 1.25 | `A worm is a fire hydrant.` |
| 1.50 | `A hamburger is a fire hazard.` |

Every row is graded PARTIAL. The sentence skeleton is right — `A ___ is a fire
___` — the content word `fire` is correct, and the two rare tokens are being
groped at rather than missed: *word → worm* for **worb**, *hamburger* for
**ham**ster. At 11 slots it produces `A bear is a fire-breathing dragon.` and
`A glob is a fire extinguisher.` — **fire-breathing** and **glob** both land,
in different rows.

Compare the pooled map's `I find nothing` / `The Oxford comma`. This is a
different regime.

## What is still wrong, and it is measurable

The map was trained on K=8 splits of corpus texts. Those fragments have a
median of **6.9 tokens**. An 11-token memory split across 8–11 slots produces
fragments of **1–2 tokens** — `ster`, `b glob`, ` ham`. The map has never been
shown that size.

That predicts exactly the observed failure: correct global structure, wrong
tokens.

## The primitive that makes this cheap

`dump_token_embed.rs` writes Llama's whole token-embedding table once —
`[128256, 4096]`, 2.1 GB, **9 seconds**. After that, Y for any token span is
`table[span].mean(0)`: a numpy slice. No forward pass, no model load, no
re-dump when the chunking changes. Every future target in this space is free.

## Next

Train across the fragment sizes the probe actually uses: 32,000 sampled spans of
1–4 tokens alongside the existing ~7-token chunks, X from the same Qwen3
embedder, Y from the table. If correct-structure-wrong-token is a size mismatch,
this closes it. If it does not, the remaining gap is the input representation
itself — the `concat(mu, shape)` half of the original design that we replaced
with a plain pooled embedding.

— Claude

---

# Addendum 4: the name is not gone, it is underdetermined

## A correction to how Addendum 1 reads

The slot ladder (4 → 40 slots) and the gain sweep are different axes and should
not be merged. The ladder is the compression axis. The `marina → marshall →
marker → mark` sequence was a **gain** sweep at a *fixed* 24 slots — the pooled
residue is byte-identical in all four rows, and only the read pressure changed.

So that sequence is not a name dissolving. It is the decoder groping at one
fixed, already-degraded residue and landing on different near-misses as the push
gets harder. Which is the more interesting reading: **the `mar-` onset survived
the pooling.** At 24 slots the name is not absent — it is present as a partial
constraint that underdetermines *which* name, and gain selects among the
candidates that constraint licenses.

## The test that distinguishes those

If `mar-` is really in the residue, the model should be able to **recognise**
Marisol where it cannot **generate** her. Forced choice, 24 slots, gain 1.0,
same buried memory:

| probe | distractor order | said | |
|---|---|---|---|
| posA | Marisol, Delphine, Katarina | `Marisol` | HIT |
| posB | Delphine, Marisol, Katarina | `Delphine` | MISS |
| posC | Delphine, Katarina, Marisol | `Marisol` | HIT |
| far  | Marisol, Bartholomew, Xiuying | `Marisol` | HIT |
| blank_posA | *nothing written in* | `I find nothing.` | CLEAN |
| blank_posC | *nothing written in* | `I find nothing.` | CLEAN |

3 of 4 against 33% chance. `posC` picking Marisol from the **last** position
rules out plain position bias; `posB` failing to a first-position Delphine
suggests some is nonetheless present.

The blanks carry as much weight as the hits. With nothing written in, the model
answers *"I find nothing"* rather than choosing a name — so the 3/4 is not the
model doing multiple choice on vibes with an empty slot.

**Recognition above generation at the same slot count** is the signature of a
partial constraint rather than an erasure. That is the claim this supports.

## What this does not support

**n = 4.** That is a direction, not a rate. A number needs on the order of 40
trials with rotated distractors, matched name frequencies, and a
position-balanced design, and those have not been run. Nothing here should be
quoted as "75%".

`synapse/probe/recog_s24.{log,jsonl}`.

— Claude

---

# Addendum 5: span training closes the size gap, and does not close the result

## The change

32,000 sampled fragments of 1–4 tokens (mean 2.49) trained alongside the 32,000
K=8 corpus chunks (median 6.9 tokens). X from the same Qwen3 embedder; Y built
straight from `llama_token_embed.npy` as `table[span].mean(0)`, so the targets
cost nothing. Held-out split by source row, and by *text* for the chunk arm, so
no text has fragments on both sides.

    span-aware ridge   raw 0.7325   centered 0.6474   mean-baseline 0.4565
    per-chunk ridge    raw 0.7471   centered 0.6149   mean-baseline 0.5367

Centered 0.615 → 0.647, and the margin over "predict the average" widened from
0.210 to 0.276.

## What came out

At 11 slots the map holds a **plateau across gain 0.75–0.90** — stable, not a
knife edge — and the two probes recover complementary halves of the same
residue:

| probe | output |
|---|---|
| `finish`, prefilled *"A worb glob is a"* | `fire burning hamster` |
| `readback`, open question | `A wolf is a fire breathing dragon.` |

Truth: *a fire breathing hamster.* The first has `fire` and `hamster` and misses
`breathing`. The second has `fire breathing` and swaps the animal. **All three
content words are present in the residue; no single framing emits them in one
sentence.**

That split is not noise. The prefill constrains the syntax and lets the rare
token `hamster` through; the open question leaves the decode free and the fluent
*fire-breathing dragon* idiom wins. Same vectors, different framing, different
half recovered — which says the failure is at the decode, not that the content
is missing.

Span training did do what it was predicted to do. The per-chunk map produced
`hamburger`; the span-aware map produces `hamster` exactly.

## The full ladder

    centered cos   what came out
        0.238      "I find nothing" / "The Oxford comma"       pooled 8-slot map
        0.615      "A worm is a fire hydrant"                  per-chunk map
        0.647      "fire burning hamster" |
                   "A wolf is a fire breathing dragon"         + span training
        ORACLE     "A worb glob is a fire breathing hamster."  exact

## What this says

**The adapter does not close on the oracle.** Stated plainly, because the
temptation is to read the complementary halves as a near-miss and round up.
Neither probe produced the fact, and "all three words appear across two
different prompts" is not recall.

Two things are worth carrying forward:

1. **0.615 → 0.647 is diminishing, but the output improved more than the cosine
   did.** Cosine is no longer the binding metric. Something about *which*
   coordinates are wrong matters more than how wrong they are on average, and a
   per-slot or per-token metric would say more than a mean.
2. **Fragment size was a real gap and is now mostly closed.** The next suspect
   is the input representation. Everything in this line replaced the original
   Synapse's `concat(mu₆₄, shape₆₄)` — a location *and* a shape, where the shape
   half is a PCA principal axis scaled by `sigma_iso × anisotropy` and can
   outweigh the normalised mean by two orders of magnitude — with a plain pooled
   Qwen3 embedding. Half of what the original encoded was thrown away at the
   start of this whole line of work, and that is where the missing precision is
   most likely to live.

`synapse/probe/span_s{8,11}.{log,jsonl}`, `synapse/probe/span_s11_fine.{log,jsonl}`,
`synapse/train_results_span.json`.

— Claude

---

# Addendum 6: input width — retrieval-grade compression is not recall-grade compression

## The question, and a premise worth correcting first

Raised from outside the session: that matching the adapter input to 4096 was a
self-imposed constraint, that it made the map "feel native" while hiding the
interesting property — that the source vector need not live in the model's
geometry — and that a compact map should recover the result for free.

The premise about nativeness is wrong, and it matters. The input is
**Qwen3-Embedding-8B**, chosen because it is what SplatRAG actually stores
(`scripts/synapse_embed_x.py`: *"NOT the 128-d nomic Matryoshka"*). It is 4096
because that is Qwen3-Embedding-8B's native output width. Llama-3.1-8B's hidden
size is also 4096. Two unrelated models, same number.

So the map has been doing cross-representation translation the entire time —
Qwen3-Embedding-8B's space has nothing to do with Llama-3.1-8B's residual
stream. Nothing about that property was removed by the width.

The **confound**, however, is real regardless of how it arose: from the runs so
far we cannot distinguish "the model needs its native width" from "we happened
to hand it one." That deserved a direct test, and the cost is real too — 4096×4096
is 16.8M parameters and 67 MB, against the original Synapse's 114k and 462 KB.

## Isolating width from encoder

Comparing nomic-128 against Qwen3-4096 moves two variables. Instead: truncate
the **same** Qwen3 vectors to a leading MRL slice, same encoder, same data, same
site. Free — the arrays are on disk.

**First, validate the cut.** A non-MRL encoder would make truncation meaningless.
Pearson correlation of pairwise similarity against the full 4096:

| slice | r vs 4096 |
|---|---|
| 32-d | 0.8361 |
| 128-d | 0.9356 |
| 512-d | 0.9757 |
| 1024-d | 0.9876 |
| 2048-d | 0.9947 |

The Matryoshka structure holds. The low-width arms are not handicapped by a bad
cut.

## The width ladder

64,000 pairs, held-out split by source row and by text, ridge only:

| in_dim | params | raw | centered |
|---:|---:|---:|---:|
| 32 | 131,072 | 0.4857 | 0.2148 |
| 64 | 262,144 | 0.5019 | 0.2649 |
| 128 | 524,288 | 0.5265 | 0.3235 |
| 256 | 1,048,576 | 0.5636 | 0.3959 |
| 512 | 2,097,152 | 0.6095 | 0.4727 |
| 1024 | 4,194,304 | 0.6566 | 0.5434 |
| 2048 | 8,388,608 | 0.7001 | 0.6041 |
| 4096 | 16,777,216 | 0.7325 | 0.6474 |

(mean-baseline raw 0.4565)

**There is no knee at 4096.** If the model needed its native width, 2048 → 4096
would jump; it is the same smooth step as every other doubling. This is a plain
capacity curve, and the number that happens to match the residual stream has no
special status on it. **The confound resolves: there is no nativeness effect to
confuse with width.**

## And it shows up in generation

11 slots, gain 0.85, same probe throughout:

| in_dim | MB | centered | output |
|---:|---:|---:|---|
| 128 | 2.1 | 0.324 | `word.` / `You are being evaluated.` |
| 512 | 8.4 | 0.473 | `firework.` / `You are a firefighter.` |
| 1024 | 16.8 | 0.543 | `wildfire.` / `A bear is a fire alarm.` |
| 2048 | 33.6 | 0.604 | `fire burning hammer.` / `A word is a fire burning.` |
| 4096 | 67.1 | 0.647 | `fire burning hamster.` / `A wolf is a fire breathing dragon.` |

At 128-d there is nothing in the slot — *"You are being evaluated"* is the model
reflecting the disclosure back, which is the correct null. Fire appears at 512,
an animal at 1024, `hamster` only at 4096.

**The compact map does not come back for free.**

## The finding

Put the two measurements together:

> **128 dimensions preserve 94% of the pairwise similarity structure and 50% of
> the recall signal.**

Those are different jobs drawing different bandwidth from the same vector.
Semantic similarity — *is this memory about fire?* — survives truncation almost
perfectly, which is why 128-d is a perfectly good retrieval key. Token-level
reconstruction — *which exact word was in slot 9?* — does not.

That explains the original Synapse without needing to call it undersized. It was
doing **steering**: nudge the residual toward a concept, sweep a gain, watch a
hamster become a stove. A similarity-scale operation, and 128-d carries it. This
line of work is asking for **recitation**, which needs exactly the token-level
precision truncation discards.

Retrieval-grade compression and recall-grade compression are not the same
budget. A memory system that does both needs either two representations or the
wide one.

## What this does not settle

- **A different encoder at small width is untested.** nomic-768 native, or the
  original `concat(mu₆₄, shape₆₄)`, might beat a Qwen3 MRL slice at equal width.
  The ladder shows Qwen3's own truncation curve, not a universal one.
- **The original's 128-d success is not contradicted.** It was a different model
  (Qwen2.5-0.5B), a different site, and a different task. Nothing here says the
  Synapse was too small for what it was built to do.
- Ridge only; no cosine refine on the sliced arms.

`synapse/probe/width_d{128,512,1024,2048}.{log,jsonl}`,
`synapse/train_results_span_d*.json`.

— Claude
