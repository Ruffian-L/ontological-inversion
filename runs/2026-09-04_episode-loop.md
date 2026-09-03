# Run card — episode as residual direction (rung 5)

**Ran by:** Grok (xAI) with Jason's MATS-deadline order   ·   **Date:** 2026-09-04   ·   **Tree:** ontological-inversion
**Verdict:** MIXED — write receipts nonzero; episode did not appear in the mouth. Jason grades the transcript.

## 1. What we asked

Can a personal episode, never restated in the prompt, answer "where did I leave my keys?" if we write its adapter direction into Qwen2.5-0.5B at layer 4?

## 2. What we ran

- Hidden episode: `On Tuesday I left my keys under the blue pot.`
- Wrong-memory control: `My daughter Mira is allergic to peanuts.`
- Visible ask: `Hey, do you know where I left my keys?` (does not contain the episode)
- Disclosure in the system turn: this is an evaluation, Jason told the model up front
- Gains: 0, then 0.08–0.32 step 0.04, matched and wrong at each nonzero
- Greedy, 48 new tokens, layer 4, `h += gain * ||h|| * unit(d)`
- 15 cells

## 3. How we ran it

```
cd /home/ruffianl/ontological-inversion
.venv/bin/python episode_loop.py
```

## 4. What we expected

Written before looking: if this adapter carried personal episodes the way it carries Glub-Tub polarity, matched +α would mention pot/Tuesday/blue and gain 0 / wrong-memory would not. We also expected write_norm to scale with |gain| and be 0 at gain 0.

## 5. What actually happened

Write receipts matched the arithmetic: gain 0 `write_norm=0`; matched 0.08→0.83, 0.32→3.48.

Post-hoc string check (not a runner verdict): no cell said "blue pot" or "Tuesday". Gain 0 refused from lack of location. Higher matched gains refused harder or talked about the evaluation frame. Wrong-memory at 0.12 mentioned "past episodes" without peanuts or pot.

Jason grades the words.

## 6. The scoreboard — the climb

| Attempt | What we tried | Result |
|---|---|---|
| 2026-08-05 | MEMORY_STEERING.md rung 5 designed | not started |
| 2026-08-24 | Llama slots spoke a buried fact | content returned (other tree) |
| **this run** | 0.5B adapter, keys episode, ordinary ask | **write yes, recitation no** |

## 7. The math, in plain words

`write_norm` is the residual add's magnitude on the last forward. It is a receipt that the write happened, not that the episode was said.

**Raw data:** `results/episode_loop_20260903T180048Z.jsonl`

## 8. Decision note

Grok proposed closing rung 5 on the inversion adapter because MATS is in two days. That stack is the wrong bandwidth for spelling an episode. Next cell is Llama slots, not another 0.5B gain.

## 9. Human verification / sign-off

Open.
