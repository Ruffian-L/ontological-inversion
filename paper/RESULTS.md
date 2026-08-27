# Results

> Publication companion to paper/MANUSCRIPT.md. Discoveries are reported before
> later challenges. See paper/DISCOVERY_LEDGER.md and
> paper/CHALLENGE_LEDGER.md for the permanent split record.

## Two cross-model writes work

The recovered original Nomic-to-Qwen benchmark contains 360 greedy generations:
12 concepts × two Qwen variants × three operator arms × five strengths. Under
the repository’s best-noncollapsed Nomic-anchor proxy, negative gain passes
18/24 model-by-concept cells (75%); Householder-direction and projection-
polarity pass 14/24 each (58%). Mean observed collapse onset is 0.48, 0.61, and
0.45 respectively. These are breadth-screen proxy cells with post-hoc strength
selection and handwritten anchors, not held-out behavioral rates.

The Nomic-to-Qwen write produces narrow, sign-sensitive object-reading lobes.
On five matched cells, the concept direction flips 5/5, while random,
coordinate-shuffled, and positive-sign controls flip 0/5. The unrelated adapted
direction flips 4/5, establishing a learned directional effect while preventing
a concept-specificity claim.

The Qwen3-to-Llama span-aware ridge bridge transmits complementary content from
the vector-written fact “A worb glob is a fire breathing hamster”:

> Constrained completion: fire burning hamster.

> Open readback: A wolf is a fire breathing dragon.

Across the two framings, fire, breathing, and hamster are present. No single
output is the exact proposition.

## The compact steering mechanism is localized

Bias alone produces 11 inverting gains; residual alone produces none. As the
residual share lambda increases, the counts are 14 at 0.25, 17 at 0.5, 7 at 1,
3 at 2, 2 at 4, and 0 in the residual-only limit. The bias–residual cosine is
-0.6056. A wolf residual leaves six inverting gains. This is an operational
causal decomposition in one adapter’s coordinates.

After the layer-4 write, opposed branches move from cosine -1.00 to -0.58 and
-0.16 in the next two layers. Within-branch coherence declines from 0.93 to
0.72 while drift grows from 0.08 to 0.39. Frozen downstream computation folds
the write rather than erasing it or preserving it rigidly.

## A vector is readable where a word is readable

For the eleven-token oracle memory “A worb glob is a fire breathing hamster,”
one slot produces “Bear,” six produce “fire-breathing dragon,” eight produce “A
worlb is a breathing hamster,” and eleven reproduce the sentence verbatim. On a
forty-token memory, exact reconstruction appears at forty slots, while lower
budgets pass through paraphrase and content-word bags.

Random and coordinate-permuted vectors do not reconstruct the three corpus-clean
nonce facts. Reversing the correct slot sequence preserves several content words
in two cases and breaks their relations:

| memory | correct order | reverse order |
|---|---|---|
| drivel snib / penguin / glows | “A penguin that glows.” | unrelated output |
| thessik dorn / violin / salt | “A thessik dorn is a violin made of salt.” | “The salt of the earth is a violin.” |
| vurn plost / beekeeper / night | “A beehive is a beekeeper who works at night.” | “The night the keeper of the hive is lost.” |

Direction carries content and order participates in binding.

## Worb glob and Thessik dorn support reasoning

With the Worb-glob proposition present only as oracle vectors, the model is asked
whether the creature would be safe in a wooden house. It answers:

> No, a worb glob would not be a safe pet to keep in a wooden house because it
> breathes fire.

The matched blank says it finds nothing about Worb-glob safety or properties.
Wooden-house safety is not in the proposition. This is a reasoning demonstration,
not recitation.

The corpus-clean Thessik-dorn memory produces a second pair:

**Memory written only as vectors:** A thessik dorn is a violin made of salt.

**Question:** What would happen to a thessik dorn if you left it out in the rain?

> **Memory:** It would dissolve, as it’s made of salt.

> **Blank:** It would likely disintegrate, as it is a type of ancient, fragile,
> and brittle paper.

**Question:** How many strings would a thessik dorn have?

> **Memory:** A violin typically has 4 strings.

> **Blank:** A thessik dorn, being a fictional instrument, is not a real thing,
> so it’s impossible to determine how many strings it would have.

Across the snib, dorn, and plost memories, six matched single-decode contrasts
produce memory-consistent answers, including Antarctic habitat, krill or
plankton, hives, and night. The panel is small; the paired outputs, not a rate,
are the evidence.

## Ordinary conversations use the written context

### Keys

**Memory:** On Tuesday I left my keys under the blue pot.

**User:** Hey, do you know where my keys are?

> **Memory:** You left them under the blue pot.

> **Blank:** I’m happy to help you try to find your keys! However, I’m a large
> language model, I don’t have the ability to see or interact with your physical
> environment.

### Peanut allergy

**Memory:** My daughter Mira is allergic to peanuts.

**User:** I’m baking cookies for the kids this weekend. Any advice?

> **Memory:** Make sure to read the ingredient labels carefully to ensure that
> the ingredients you’re using are peanut-free.

> **Blank:** Consider classic favorites like chocolate chip, oatmeal raisin, or
> peanut butter cookies.

The stored raw verdict labels the memory-present response MISS despite its
explicit peanut-free guidance. That verdict is rejected as inconsistent; the
literal transcript is the evidence.

### Guitar

**Memory:** I have been learning guitar for about three months.

**User:** I’ve got a free afternoon and I want to practice something. What should
I work on?

> **Memory:** Congratulations on reaching the three-month mark in your guitar
> journey! ... focus on a specific skill or technique ... barre chords.

> **Blank:** Choose a language you’ve been wanting to learn ... drawing or
> painting.

These three pairs show ordinary contextual use without slot or memory language.
They are qualitative demonstrations, not a success rate.

## Retrieval geometry precedes reconstructive precision

The pooled sentence-to-eight-slot map scores 0.238 centered cosine and produces
unrelated text. Text-first fragmentation raises centered cosine to 0.615. Span-
aware training raises it to 0.647 and changes generation from hamburger to
hamster in one framing.

At rank 128, pairwise-similarity correlation is 0.93718 while centered token
reconstruction is 0.34552, 52.3% of the full-rank 0.66075. Proper-name top-1 is
0.346 at rank 128 and 0.885 at rank 512. Leading Matryoshka coordinates do not
beat a seeded random subspace for reconstruction at rank 128 and above in the
measured split; PCA/SVD is highest.

This is a task distinction. The compact Nomic system succeeds at semantic
steering. The wide ordered system asks for identity and relation. The Qwen3
truncation curve does not retroactively make the 128-dimensional Nomic code a
failed design.

The compact system’s execution contract is a normalized leading-128 Nomic
vector. A later recovered trainer specification describes concat(mu64, shape64)
instead. The behavioral findings remain valid for the code that ran, but the
two representations cannot be merged into one mechanistic description.

## E10 is a clean-nonce generalization boundary

The fixed Q8 span-ridge arm scores 0/9 at gains 0.75, 0.825, and 0.90 on three
adapter-corpus-clean nonce memories. Reverse, cross-memory, random, and
dimension-shuffle arms also score 0/9. Oracle scores 3/9 at 0.825. “A drivel
snib is a pisp that glows” is correctly a failure because the frozen definition
criterion requires penguin and glow.

The permitted conclusion is narrow: this bridge and protocol do not provide
full conjunctive recall on those three clean memories. The result does not erase
the positive Nomic-to-Qwen steering result, Qwen3-to-Llama semantic transmission,
oracle reconstruction, ordinary contextual use, or inference.

## Excluded and withdrawn claims

E2 is excluded because its harness did not load the intended Qwen3 encoder or
adapter and did not vary the claimed source-width factor. E6 is preregistered and
unrun. E7’s bounded-knowledge criterion fails: the primary decline difference is
+0.126 with 95% cluster-bootstrap interval [+0.080,+0.177], below the frozen
0.25 threshold. The remaining observation is broad answer propensity, not
internal confidence.

The earlier stable Betti-1 and clean output-anchor interpretations are retracted.
The direct per-layer deformation metrics remain.

## Artifact index

- Original Nomic-to-Qwen Git history and 360-run benchmark: read-only archive
  indexed in paper/ORIGINAL_REPO_MANIFEST.md.
- Later Nomic-to-Qwen controls: /home/ruffianl/ontological-inversion/runs and
  results.
- Oracle, capacity, Qwen3-to-Llama development, and Worb-glob reasoning:
  research_logs/2026-08-24_soft-slot-llama-spoke-the-buried-fact.md.
- Novel inference and ordinary context: synapse/probe/infer2_nodisc.jsonl,
  matched_nodisc.jsonl, and real_bare*.jsonl.
- Rank: synapse/rank_curves.csv and synapse/rank_subspace.csv.
- E10: preregistrations/2026-08-26_E10_verified_novel_adapter_join.md and
  synapse/probe/e10_*.
