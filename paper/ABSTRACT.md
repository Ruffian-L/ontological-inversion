# Abstract

Can a representation produced by one frozen model be written directly into a
second frozen language model and change what it says and reasons about? We report
an accumulated body of evidence across two cross-model write regimes. In the
first, a frozen Nomic embedding supplies a normalized 128-dimensional vector,
mapped by one affine layer to the 896-dimensional space of Qwen2.5-0.5B and
added at transformer layer 4. An original 360-run breadth screen spans 12
concepts, two Qwen variants, three operators, and five strengths. Its best-
noncollapsed Nomic-anchor proxy passes 18/24 cells for negative gain and 14/24
for each alternative operator; these are proxy screen cells, not held-out rates.
For a fictional magma-eating hamster, later dense controls show stable,
sign-sensitive steering with two narrow object-reading lobes. A matched
direction flips 100% of tested cells, while random and coordinate-shuffled
directions flip none. An unrelated adapted direction flips 80%, so the effect is
directional but not yet concept-specific. An affine decomposition localizes the
inversion to the learned bias in the shipped coordinates and shows antagonism
between bias and concept residual.

In the second regime, Qwen3-Embedding-8B fragments are mapped by ridge regression
into ordered input-embedding slots of frozen Llama-3.1-8B-Instruct. This
Qwen-to-Llama bridge transmits memory-specific content: one development probe
yields “fire burning hamster” under a constrained completion and “a wolf is a
fire breathing dragon” under open readback, recovering complementary parts of
the same vector-written fact across a gain plateau. Target-space oracle slots go
further, reconstructing corpus-clean nonce facts and supporting novel inference,
including the fire-eating hamster safety demonstration and a salt violin
dissolving in rain. Ordinary conversational probes also show vector-written
context affecting practical answers.

Rank experiments explain why the regimes have different bandwidth needs. At
rank 128, Qwen3 preserves pairwise similarity at r=0.937 while centered target-
token reconstruction is 0.346, 52.3% of its full-rank value. Compact codes can
therefore support semantic steering before they support ordered reconstruction.
A later preregistered test does not negate either cross-model write: it shows
only that this fixed ridge bridge fails full conjunctive recall on three adapter-
corpus-clean nonce memories, scoring 0/9 at each of three gains. The result is a
positive channel map with a measured generalization boundary: cross-model vector
writing, reconstruction, contextual use, and inference have been demonstrated,
while retrieval geometry alone does not guarantee exact novel recall.

**Priority note.** Jason is the originator and lead researcher of the
experimental program. The recovered original Git baseline is dated 24 June
2026; its committed provenance asserts origin in November 2025 SplatRAG/Niodoo
sessions. Dated run cards, transcripts, hashes, and commits are the priority
record; later related-work positioning does not replace that record.

**Intellectual-independence note.** Jason conceived and built the work through
original experimentation with Gemini, Grok, ChatGPT, and Claude as research
collaborators. None of the academic papers cited in the manuscript was used to
conceive or build the system. The references were added afterward solely to
position the finished work for paper readers.
