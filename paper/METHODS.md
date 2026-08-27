# Methods

> Publication companion to paper/MANUSCRIPT.md. This file describes the two
> positive cross-model pipelines and the scope of their controls. It is not an
> independent paper.

## Frozen endpoints and two write regimes

The experiments study two distinct jobs.

1. **Semantic steering:** frozen Nomic v1.5 → normalized leading-128 vector →
   one affine 128→896 adapter → frozen Qwen2.5-0.5B layer-4 residual stream.
2. **Ordered reconstruction and use:** frozen Qwen3-Embedding-8B fragments → one
   affine ridge 4096→4096 map → ordered input-embedding slots of frozen
   Llama-3.1-8B-Instruct.

Neither pipeline updates either endpoint. Their input representations, target
models, write sites, and behavioral readouts differ, so results are not pooled.

## Nomic-to-Qwen steering

The executed original reproduction and benchmark take a Nomic sentence
embedding, retain the leading 128 coordinates, L2-normalize them, and apply one
affine 128→896 map. A later recovered trainer specification describes a
different input: concat(mu64, shape64), where shape64 is a PCA principal axis
scaled by isotropic spread and anisotropy. The June behavior reported in the
paper belongs to the executed leading-coordinate route. The recovered trainer
targets Qwen input-token-embedding means, but its input contract must not be
substituted for the public execution code.

The shipped adapter’s SHA-256 is
f32024e313f9cfef477ab4ce0ae9539177805f7873901fa0605bb8b3ea3b02bd and its MD5
is b8118021c7c27948565c4f322b5f6e04. Archived and current copies are byte-
identical. Its original training corpus is unavailable; the rebuild
specification is functionally complete but does not support a byte-identical
retraining claim.

The original breadth screen runs 12 concepts through Qwen2.5-0.5B-Instruct and
Qwen2.5-Coder-0.5B-Instruct under negative-gain, Householder-direction, and
projection-polarity arms at strengths 0.1, 0.2, 0.3, 0.5, and 0.8: 360 greedy
generations total. Within each model-by-concept-by-operator cell, the report
selects the best noncollapsed strength when one exists; cells with none are
failures. A proxy flip requires inversion gain
greater than 0.02, where inversion gain is Nomic cosine to handwritten antipode
anchors minus cosine to handwritten concept anchors. Collapse uses lexical-
diversity, repeated-four-gram, top-token, and alphabetic-character thresholds.
These design choices make the benchmark a breadth screen rather than a held-out
behavioral rate.

The principal Glub-Tub prompt is decoded greedily. The response surface is
measured at 0.01 gain increments. Matched controls use random, coordinate-
shuffled, unrelated-adapter, and positive-sign directions. The affine
decomposition evaluates bias b, concept residual Wv, b+lambda Wv for several
lambda values, and a wolf residual substitution. These are coordinate-dependent
causal interventions in one stored adapter.

The original Git receipts are in the read-only archive indexed by
paper/ORIGINAL_REPO_MANIFEST.md. Later receipts are under
/home/ruffianl/ontological-inversion, particularly
runs/2026-06-24_topology-of-the-flip.md,
runs/2026-06-25_per-layer-fold-decay.md, and
runs/2026-08-25_bias-vs-residual.md.

## Qwen3-to-Llama ordered slots

Memory text is split into ordered fragments before source encoding. Qwen3 emits
x_i in 4,096 dimensions. The same fragment is tokenized by Llama; the mean of
its input-token embeddings is target y_i. After target centering, ridge
regression with penalty 1 fits one shared affine map x_i→y_i.

The span-aware training set combines 32,000 ordinary chunks with 32,000 sampled
one-to-four-token spans, mean length 2.49. The held-out split is by source row and
text. The prompt contains reserved markers whose embeddings are replaced before
block 0. Each slot is normalized relative to the prompt’s mean ordinary-token
embedding norm and scaled by gain. The source memory is evaluator-only and never
appears as readable inference text.

The retained adapter is SHA f689bbf0…. Centered held-out cosine is 0.6474. This
metric is treated as static reconstruction evidence, not as a behavioral recall
certificate.

## Oracle and controls

The target-space oracle replaces each slot with the corresponding Llama token-
embedding mean. It proves whether the frozen target can read the site; it does
not test retrieval or cross-model translation.

- Random controls preserve approximate write scale without content direction.
- Coordinate permutations preserve values and norm while destroying arrangement.
- Reverse order preserves every slot vector and changes only sequence.
- Cross-memory controls write one memory and ask about another.
- Blank controls retain the prompt and omit the write.

Readouts are separated into exact reconstruction, semantic transmission,
ordinary contextual use, and inference. Exact recall requires all frozen
conjunctive content groups in one output. Complementary words across prompts are
semantic-transmission evidence but are not rounded up to exact recall.

## Rank and subspace measurements

The ridge bridge is refit at ranks 4, 8, 16, 32, 64, 128, 256, 512, 1024,
2048, and 4096. The fixed split has 28,800 training and 3,200 held-out span pairs
and 49,990 held-out pairwise-cosine pairs. Measurements are:

- pairwise-similarity correlation against full-width Qwen3;
- centered cosine for Llama token-embedding reconstruction;
- descriptive proper-name nearest-neighbor top-1;
- centered relation reconstruction.

Matched-rank subspaces are leading Matryoshka coordinates, top PCA/SVD, and one
seeded random subspace. Results are descriptive for one split.

## E10 corpus-clean challenge

E10 uses three nonce memories absent from the adapter corpus, fourteen ordered
slots, Qwen3-Embedding-8B-Q8_0, the existing span-aware ridge map, frozen Llama,
and gains 0.75, 0.825, and 0.90. The model is told it is participating in a
vector-memory evaluation. Expected facts and semantic criteria are evaluator-
only. One greedy output is recorded per cell.

Aligned, reversed, cross-memory, random, dimension-shuffle, oracle, and blank
arms share the same prompt family. E10 tests clean-nonce conjunctive
generalization for this protocol. It is not the first cross-model join test and
cannot negate the earlier Qwen3-to-Llama semantic-transmission receipt.

## Exclusions and instrument boundaries

The planned E2 source-width experiment is excluded. Its harness used target
width 896, did not load Qwen3 or the ridge adapter, and made widths at or above
896 identical except seed. E6 remains preregistered and unrun. E7’s decline
patterns are surface behavior, not calibrated confidence. Anti-guessing and
neutral instructions are treated as distinct interventions, and inference is
interpreted only with matched blanks.

All model-facing tests disclose the evaluation frame. No model run was performed
for the 26 August scope correction.
