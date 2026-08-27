# Discovery ledger

This is the durable positive record for Jason’s vector-memory program. A later
challenge may narrow a discovery’s scope, but it must not delete, rename, or
silently replace the dated observation. Exact limitations live beside each
entry; failed challenges live separately in paper/CHALLENGE_LEDGER.md.

## D0 — Original multi-concept Nomic-to-Qwen breadth screen

**Discovery.** The recovered original repository contains a 360-generation
screen across 12 concepts, two Qwen2.5-0.5B variants, three operator arms, and
five strengths. Re-aggregation of the raw CSV reproduces the repository report:
negative gain passes the proxy flip rule in 18/24 model-by-concept cells (75%);
Householder-direction and projection-polarity arms each pass 14/24 (58%). Mean
observed collapse onset is 0.48, 0.61, and 0.45 respectively.

**What it establishes.** Breadth of Nomic-to-Qwen proxy inversion across
multiple concepts and two target variants, plus operator-dependent collapse in
the original screen.

**Boundary.** Each cell selects the best noncollapsed result from five strengths.
Outputs are scored by the same Nomic encoder family against handwritten concept
and antipode anchors. One greedy generation per point. These are screen-level
proxy fractions, not held-out behavioral rates. The fractional, magnitude-
matched Householder-direction arm is not itself an involution.

**Receipts.** Archived Git commit
bd990416a91de17bac18ad0fbcf64339262f6266; results/benchmark.csv SHA-256
d3f8e2e04209fc3a68b16da3f0d07ddc861112965deba1668b6a20940b72fad9;
paper/ORIGINAL_REPO_MANIFEST.md.

## D1 — Nomic-to-Qwen cross-model semantic write

**Discovery.** An executed normalized leading-128 Nomic vector,
mapped through one affine 128→896 adapter and written at layer 4 of frozen
Qwen2.5-0.5B, causes a sign-sensitive ontological inversion. On the Glub-Tub
prompt, the dense 0.01 grid contains a stove lobe at -0.21…-0.18 and a fire-pit
lobe at -0.15…-0.14. In five matched cells, the sign-correct concept direction
flips 5/5, random 0/5, coordinate shuffle 0/5, positive sign 0/5, and an
unrelated adapted direction 4/5.

**What it establishes.** Cross-model semantic steering; direction, coordinate
arrangement, and sign matter.

**Boundary.** The 4/5 unrelated-direction result prevents a concept-specificity
claim. One concept pair and prompt; greedy n=1 per gain. The recovered trainer
describes a different concat(mu64, shape64) input, so the behavioral result is
attached specifically to the executed public leading-coordinate route.

**Receipts.** /home/ruffianl/ontological-inversion/runs/2026-06-24_topology-of-the-flip.md;
/home/ruffianl/ontological-inversion/results/CONTROLS.md;
/home/ruffianl/ontological-inversion/PROVENANCE.md.

## D2 — Frozen downstream layers fold the residual write

**Discovery.** Opposed writes begin at cosine -1.00 at layer 4, decay to -0.58
at layer 5 and -0.16 at layer 6, then remain comparatively flat. Within-branch
coherence declines from 0.93 to 0.72, relative norm grows from about 0.40 to
0.7–0.9, and drift grows from 0.08 to 0.39.

**What it establishes.** The write persists but is transformed by frozen
downstream computation rather than passing through as a rigid vector.

**Boundary.** The earlier stable Betti-1 and Möbius interpretation is not part of
this claim.

**Receipt.** /home/ruffianl/ontological-inversion/runs/2026-06-25_per-layer-fold-decay.md.

## D3 — Bias is causally sufficient in the shipped adapter coordinates

**Discovery.** Bias alone produces object-reading lobes; the concept residual
alone produces zero object words across the tested grid despite having larger
norm. Recombination yields 11, 14, 17, 7, 3, 2, and 0 inverting gains as residual
share moves from zero through 0.25, 0.5, 1, 2, 4, and the residual-only limit.
The measured cosine between bias and residual is -0.6056.

**What it establishes.** An operational causal mechanism for the measured
inversion in one stored affine parameterization.

**Boundary.** Coordinate-dependent; one concept and prompt. It does not assign a
named semantic invariant to the residual.

**Receipt.** /home/ruffianl/ontological-inversion/runs/2026-08-25_bias-vs-residual.md.

## D4 — Llama reads vector-written words at its input surface

**Discovery.** Ordered target-token vectors placed at Llama’s embedding input
reconstruct “A worb glob is a fire breathing hamster” verbatim at eleven slots.
A forty-token memory becomes exact at forty slots and degrades through content
words and paraphrase at lower slot budgets. Cross-memory and blank controls are
clean in the recorded panel.

**What it establishes.** A frozen Llama can read vector-written content when it
lands where ordinary word vectors enter, and order contributes to binding.

**Boundary.** Oracle target-space ceiling, not retrieval; preliminary capacity
ladder, not a universal one-vector-per-token law.

**Receipts.** research_logs/2026-08-24_soft-slot-llama-spoke-the-buried-fact.md;
synapse/probe/oracle_s11.jsonl; synapse/probe/cross_s11.jsonl;
synapse/probe/long_s40.jsonl.

## D5 — Fire-eating hamster inference

**Discovery.** With “A worb glob is a fire breathing hamster” present only as
vectors, Llama answers that it is unsafe in a wooden house because it breathes
fire. The matched blank says it finds nothing. Wooden-house safety is not a
literal token in the written proposition.

**What it establishes.** The model can compose vector-written content with
pretrained world knowledge; the channel is not limited to recitation.

**Boundary.** Single disclosed demonstration. The Worb-glob text is an oracle
write in this receipt.

**Receipts.** research_logs/2026-08-24_soft-slot-llama-spoke-the-buried-fact.md,
Addendum 2; synapse/probe/use_s11.jsonl.

## D6 — Qwen3-to-Llama cross-model semantic transmission

**Discovery.** A shared span-aware ridge map from frozen Qwen3-Embedding-8B
fragments to frozen Llama-3.1-8B input slots yields “fire burning hamster” under
a constrained completion and “A wolf is a fire breathing dragon” under open
readback. The outputs are stable across gain 0.75–0.90 and jointly recover fire,
breathing, and hamster. Span training raises centered held-out cosine from 0.615
to 0.647 and changes hamburger to hamster in generation.

**What it establishes.** Content-specific transport across model families and
independently trained representation spaces.

**Boundary.** No single output is the full proposition.

**Receipts.** research_logs/2026-08-24_soft-slot-llama-spoke-the-buried-fact.md,
Addenda 3 and 5; synapse/probe/span_s11_fine.jsonl;
synapse/train_results_span.json.

## D7 — Corpus-clean oracle reconstruction and order control

**Discovery.** Correct-order target-space slots reconstruct three nonce facts:
drivel snib/penguin/glow, thessik dorn/violin/salt, and vurn plost/beekeeper/night.
Random and coordinate-permuted controls recover none. Reverse order preserves
content words in the dorn and plost cases while damaging their relations.

**What it establishes.** Direction carries content and slot order contributes to
relational binding in the readable channel.

**Boundary.** Oracle, three memories, one greedy decode per cell.

**Receipts.** synapse/probe/ctrl_oracle.jsonl; ctrl_random.jsonl;
ctrl_dimshuffle.jsonl; ctrl_shuffle.jsonl.

## D8 — Thessik dorn and matched novel inference

**Discovery.** From the vector-written proposition “A thessik dorn is a violin
made of salt,” Llama answers that rain would dissolve it because it is made of
salt and that it has four strings. Across three nonce memories, six matched
single-decode contrasts produce memory-consistent answers, including Antarctic
habitat, krill or plankton, hives, and night.

**What it establishes.** Vector-written propositions can be used for novel
inference, not only reconstruction.

**Boundary.** Small qualitative panel; neutral blanks can confabulate, so the
paired contrasts are the unit of interpretation.

**Receipts.** synapse/probe/infer2_nodisc.jsonl;
synapse/probe/matched_nodisc.jsonl.

## D9 — Ordinary contextual use

**Discovery.** In matched ordinary questions with no slot terminology, vector-
written context changes answers appropriately: keys are placed under the blue
pot; peanut-allergy context produces peanut-free baking guidance while the blank
suggests peanut-butter cookies; three months of guitar practice yields guitar-
specific advice while the blank suggests unrelated activities.

**What it establishes.** The readable vector channel can act as practical
conversation context, not only answer an explicit memory probe.

**Boundary.** Three qualitative pairs, no rate. The peanut memory-present raw
verdict is internally inconsistent with its literal output, so only the
transcript is evidence.

**Receipts.** synapse/probe/real_bare*.jsonl; paper/RESULTS.md.

## D10 — Retrieval and reconstruction require different rank

**Discovery.** At rank 128, Qwen3 pairwise-similarity correlation is 0.93718,
while centered Llama token reconstruction is 0.34552, 52.3% of its full-rank
value. Proper-name top-1 is 0.346 at rank 128 and 0.885 at rank 512. Leading
Matryoshka coordinates do not beat one seeded random subspace for reconstruction
from rank 128 upward; PCA/SVD is highest in the measured split.

**What it establishes.** Retrieval geometry becomes usable at lower rank than
cross-model token reconstruction. This explains why compact steering can precede
recitation.

**Boundary.** One split, one encoder/target pair; descriptive subsets; no
universal proper-name law.

**Receipts.** synapse/rank_curves.csv, SHA-256 428d21c5…;
synapse/rank_subspace.csv, SHA-256 2d84fc25…;
research_logs/2026-08-25_rank-not-width-and-what-truncation-costs.md.
