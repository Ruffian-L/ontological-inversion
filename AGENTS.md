# AGENTS.md — read this before you touch anything

Owner: **Jason**. Research, not production, unless he says so.
The law for how work is recorded is [`STANDARDS.md`](STANDARDS.md). This file is
the short version an agent needs on entry.

## The rule you will otherwise forget

**Every mutation and every measured run gets logged in the same turn it happens.
Nobody should have to ask you for it.**

A *mutation* is: changing code, adding or deleting files, retraining, changing a
model or config, moving data. A *measured run* is any sweep, benchmark, control,
or eval that produced output.

In the same turn, write **all four**:

1. **A changelog entry** in [`CHANGELOG.md`](CHANGELOG.md) — house format, newest
   at the **top**, directly under the H1: `## YYYY-MM-DD — short title`, then
   **We did** (what actually happened, real counts and file paths, including what
   you did *not* do), **We think** (the hypothesis — this is a hypothesis log, not
   a status report), **Next** (the mutation that tests it). Keep it short.
2. **A run card** in `runs/YYYY-MM-DD_short-title.md` — skeleton in
   [`run_card_template.md`](run_card_template.md). Who ran it, when, verdict
   (PASS/FAIL/MIXED), what you asked, the copy-paste command, **what you expected
   before you saw the result**, what actually happened with the correct answer
   beside every result, and the boundary of what it does *not* claim.
3. **A row in** [`SCOREBOARD.md`](SCOREBOARD.md) — newest at the bottom.
   *What we tried → result.* A failure is a rung, not a fault.
4. **An entry in** [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — the narrative index,
   newest entries above the Timeline section.

When a subject spans several runs or a change of mind, it also earns a long-form
file in [`research_logs/`](research_logs/) — `YYYY-MM-DD_title.md`. A run card
answers *what did this experiment show*; a research log answers *what did we come
to understand, and what did we get wrong on the way*. Record the wrong turns: two
of them cost real time on 2026-08-25 and are now rules below.

Then **commit**, scoped to what you actually touched. A long-idle tree can carry
someone else's unstaged work; that is a separate commit or a question for Jason,
never swept in with yours.

**Failures stay.** Never rewrite or delete a past entry to make it look right —
append the correction, dated. Retractions in this repo (the 75% flip, Betti-1)
are load-bearing and are why the surviving claims are believable.

## Method rules specific to this repo

- **Always dense-sweep the gain at ≤0.01 near any candidate. Never report a single α.**
  The phenomenon is *thin, non-monotone lobes with living readings between them*
  (`results/WINDOW_MAP.md`). A coarse grid can land in a gap and report a null on
  a band that works. This is the single most common way to get a wrong answer here.
- **Hand Jason the `tail -f` command whenever you start a run**, unprompted, with
  the swept parameter visible per generation. Write to a real file, run
  unbuffered (`python -u`), and never pipe through `cut`/`head` at capture time —
  a write-time truncation destroyed a result on 2026-08-25.
- **Run the controls before claiming a direction is concept-specific.**
  `controls.py`; see `results/CONTROLS.md`. Note `unrelated_adapter` scores 80%.
- **Trust the bytes.** Pinned model revisions are in [`MODELS.md`](MODELS.md) and
  wired through `modelpin.py`. Use offline mode once cached.
- Decode is greedy, so pinned weights make runs deterministic. If two runs of the
  same cell differ, something is wrong — investigate, don't average.

## Naming

Do **not** call the synthetic concepts (Worbglob / Glub-Tub) "fake memories."
They are **probes** for residual ±gain. Real memory means the splat stores,
Niodoo continuity, golden-memories. See `RESEARCH_LOG.md` and `PAPER_FRAMING.md`.

## What not to spend cycles on

Renaming debates already settled · expanding cultural cards before controls and
involution · claiming the full Feelers 8-probe eval lives in this repo.
See `CHECKLIST.md` §E.
