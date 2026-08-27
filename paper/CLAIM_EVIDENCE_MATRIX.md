# Claim–evidence matrix

Internal wording ledger for paper/MANUSCRIPT.md. Positive discoveries appear
first. A later challenge qualifies only the exact row it tests; it never erases
the dated discovery record in paper/DISCOVERY_LEDGER.md.

## Established discoveries

| claim | status and scale | receipt | permitted wording |
|---|---|---|---|
| Original Nomic-to-Qwen benchmark shows breadth | 360 greedy runs; negative gain 18/24 proxy cells; Householder-direction and projection-polarity 14/24 each | archived benchmark.csv SHA d3f8e2e0…; commit bd990416… | “The original screen shows broad proxy inversion across 12 concepts and two Qwen variants.” |
| Original benchmark fractions are held-out behavioral rates | prohibited; best noncollapsed of five strengths, same-encoder proxy, handwritten anchors | benchmark.py, metrics.py, concepts.json | Call them proxy-flip screen cells, not independent success rates. |
| Nomic-to-Qwen is a working cross-model semantic write | supported demonstration; one concept/prompt, dense 0.01 gain grid | ontological-inversion topology run and controls | “A frozen Nomic-derived code crosses one affine map and sign-sensitively steers frozen Qwen.” |
| Executed June input is concat(mu64, shape64) | false; original reproduction/benchmark use normalized leading-128 Nomic coordinates | archived ontological_inversion.py and operators.py; current training/SPEC.md | Separate executed leading-128 route from the recovered trainer’s structured input contract. |
| Archived and current shipped adapters are the same artifact | supported; byte-identical SHA-256 f32024e3… | both adapter_final.safetensors files | “The original adapter is byte-identical to the later copy.” |
| Direction and sign, not norm alone, drive the Qwen inversion | concept 5/5; random 0/5; shuffle 0/5; positive sign 0/5 | ontological-inversion/results/CONTROLS.md | “Direction, coordinate arrangement, and sign matter.” |
| The Qwen steering direction is concept-specific | not established; unrelated adapter 4/5 | same controls | Do not call concept-specific; say a shared fitted component may dominate. |
| Bias drives the measured inversion in shipped coordinates | bias-only 11 inverting gains; residual-only 0; cos(b,Wv)=-0.6056 | ontological-inversion/runs/2026-08-25_bias-vs-residual.md | “Operational causal decomposition in one affine parameterization.” |
| Frozen downstream layers fold the layer-4 write | opposed cosine -1.00→-0.58→-0.16; coherence 0.93→0.72; drift 0.08→0.39 | ontological-inversion/runs/2026-06-25_per-layer-fold-decay.md | “The intervention persists as a transformed state deformation.” |
| Ordered Llama target vectors are readable | exact 11-token and 40-token oracle demonstrations | soft-slot log; oracle_s11 and long_s40 raw files | “Llama reads vector-written words at the embedding-input site.” |
| Direction carries content and order binds relations | three corpus-clean oracle memories; random/dimshuffle 0; two reverse-order outputs retain words but break relations | ctrl_oracle/random/dimshuffle/shuffle JSONL | “Direction and slot order play distinct roles in this panel.” |
| Worb-glob supports novel inference | wooden-house safety answer cites fire breathing; matched blank clean | soft-slot log Addendum 2; use_s11.jsonl | “The oracle proposition is composed with pretrained knowledge in one matched demonstration.” |
| Thessik dorn and companion facts support inference | 6/6 memory-consistent matched single-decode contrasts across three memories | infer2_nodisc.jsonl; matched_nodisc.jsonl | “Six matched demonstrations support vector-memory inference.” |
| Vector memory affects ordinary conversation | keys, peanut allergy, guitar; three qualitative pairs | real_bare*.jsonl; paper/RESULTS.md | Quote transcripts; no success rate. Peanut transcript overrides inconsistent stored MISS verdict. |
| Qwen3-to-Llama is a working cross-model semantic join | span-aware development outputs carry complementary fire/breathing/hamster content across gain plateau | span_s11_fine.jsonl; soft-slot log Addendum 5 | “A shared ridge bridge transmits memory-specific content across model families.” |
| Qwen3-to-Llama span result is a complete proposition | not claimed; complementary fire/breathing/hamster content across two framings | span_s11_fine.jsonl; soft-slot log Addendum 5 | Call it semantic transmission. |
| Retrieval similarity saturates before reconstruction | rank 128: r=0.93718, centered cosine 0.34552 vs full 0.66075 | rank_curves.csv SHA 428d21c5… | Use exact metrics and one-split scope. |
| Leading Matryoshka coordinates are privileged for reconstruction | falsified in one split; below seeded random from rank 128 upward | rank_subspace.csv SHA 2d84fc25… | “Retrieval-oriented leading coordinates do not automatically optimize inversion.” |

## Scoped challenges and withdrawals

| claim | status and scale | receipt | permitted wording |
|---|---|---|---|
| The fixed ridge bridge gives full clean-nonce conjunctive recall | not supported by E10; aligned 0/9 at gains .75/.825/.90; oracle 3/9 at .825 | E10 preregistration; e10_summary.json SHA 6d34b951… | “This protocol did not generalize to full conjunctive recall on three corpus-clean memories.” |
| E10 shows cross-model joining fails | prohibited overgeneralization | D1 and D6 predate E10; E10 tests only Q8/span-ridge/14-slot clean-nonce protocol | Explicitly say E10 does not erase either cross-model write. |
| The E10 glows near miss is full recall | unsupported | e10_adapter.jsonl | “A partial attribute below the penguin-plus-glow criterion.” |
| Final post-norm addition is a readable memory channel | rejected in tested Llama setup; roughly 300 generations, content-insensitive disturbance | wrong-end-of-stack log | Scope to that late site and operation. |
| E2 supports a native encoder-width result | invalid and excluded; intended encoder and adapter were not loaded | e2_raw.jsonl and harness inspection | No empirical E2 result. |
| The model exhibits bounded knowledge | withdrawn; +0.126, 95% CI [+0.080,+0.177], below 0.25 | E7 raw SHA 77caf18d… | “Memory broadly changed answer propensity; bounded knowledge is not established.” |
| Stable Betti-1 near seven or a clean Möbius structure | retracted; robustness range 0–56, output fold cosine -0.13 | ontological hardened audit | Keep per-layer geometry; omit stable topological invariant. |
| The residual causally carries a named fire invariant | unsupported | bias-vs-residual run card | Do not assign a named semantic property without a direct complement test. |
| E6 name-frequency × redundancy has a result | unrun | E6 preregistration | Preregistered future experiment only. |
| Q4_K_M reproduces Q8 | pending; no completed matched arm in this paper | model/request ledger | No claim until separately labeled receipts exist. |

## Prohibited shortcuts

- Never turn a challenge-ledger null into a universal verdict on cross-model writing.
- Never pool oracle and adapter generations.
- Never restore an adapter-corpus contamination claim for Worb-glob; that was a token grep, not a training span.
- Never convert a centered-cosine ratio into a fraction of tokens recovered.
- Never turn three ordinary conversation pairs into a rate.
- Never present the original best-of-five proxy fractions as held-out rates.
- Never collapse the executed leading-128 input into the recovered trainer’s
  concat(mu64, shape64) contract.
- Never call decline an internal confidence measure.
- Never reinstate E2 by fitting curves to the invalid intervention.
- Never infer an author, affiliation, or intellectual origin from tool logs.
- Never describe the cited academic papers as influences on the conception or
  construction of this work. They are retrospective paper positioning only.
- Credit Gemini, Grok, ChatGPT, and Claude as the named AI research
  collaborators; do not silently replace them with literature authors.
- Never claim model-size scaling caused a downstream force or answer without the
  required matched factorial and scaler receipts.
