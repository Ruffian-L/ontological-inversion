# Keys episode on Llama oracle slots — write landed, use missed

Same-day sequel to `2026-09-04_episode-loop-write-landed-mouth-missed.md`.
That was the 0.5B inversion adapter. This is the site that already recited
worb-glob: Llama-3.1-8B input soft-slots, oracle, 11 ordered chunks.

## Lane

Oracle. Host's own token embeddings of the memory text, split into 11 ordered
chunks, mean-pooled per chunk. Not a single paragraph mean. Not the Qwen3→Llama
adapter. Write is replace: `slot ← gain · tok_norm · unit(v)`.

CPU, because this machine's candle CUDA 13.3 PTX will not load on driver 580.

## What moved

The write is real. Gain 0 is blank on every probe. Gain>0 writes 11 slots.
`write_norm` tracks `gain · tok_norm`.

The mouth is not the worb-glob HIT. Recitation at 48 tokens is a denial that
names `blue pot`. A 4-token preflight at the same gain opened `On Tuesday I left`.
The use-ask (`Hey, do you know where I left my keys?`) never answers with the
location. Wrong memory (peanuts) does not leak the pot. Zorp stays empty with
the keys vector sitting in the slots.

The system disclosure tells the model that "I find nothing" is the result we
want. That scaffold is now a candidate, not a guess we get to ban.

## What we did not do

Did not mint. Did not run adapter mode. Did not treat the runner's HIT column
as a finding (it fires on the substring `blue pot` inside the denial). Did not
claim the product loop. Did not touch `universe_domain.safetensors`.

## Next

Disclosure-off recitation, or `--slots` equal to the episode's token count.
Then, only if the use-ask holds, mint.
