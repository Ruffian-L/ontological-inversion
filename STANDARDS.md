# Standards (this repo)

How we run experiments and record evidence here, so any human — or any AI — can follow what
happened, why, and what it earned. Adapted from the Niodoo standard
(`~/projects/team_build/STANDARDS.md`, jp). That document is the source of law; this is its
local application. When in doubt, follow it; when it's wrong, fix it *here*, not in a side note.

## 1. The core rule
An evidence claim isn't real unless it's **plain-text, back-and-forth, human-readable**. When the
real work is math/dense numbers, that's allowed — and it becomes our job to **translate it in plain
words right next to the real numbers**. Surfaced, never hidden.

## 2. Every claim gets a run card
One experiment that makes a claim = one **run card** in `runs/`, written in plain language. Header:
*who ran it & when* (which AI, date, tree/commit) + a one-word **verdict** (PASS / FAIL / MIXED).
Then: what we asked · what we ran · the exact copy-paste command · what we expected (*before* the
result) · what actually happened (with the correct answer beside every result) · the scoreboard ·
the math in plain words (+ pointer to raw data) · decision note · human sign-off.
Skeleton: `run_card_template.md`.

## 3. The scoreboard is the point (the climb)
`SCOREBOARD.md` is the rolling ledger — every eval's rung, *what we tried → result*, newest at the
bottom. A failed attempt is a **rung, not a fault**. A win never appears out of nowhere; you can
look back and see "that didn't hold… that flickered… **then this one held.**" Each run card restates
its own recent rungs inline.

## 4. Provenance is decisions, not just dates
Record *which knob was turned, in which direction, and why* — not only the timestamp. And **trust the
bytes**: model id, exact config, exact commands. If something mismatches, say so plainly.

## 5. The math rule
Surface the **few** numbers that matter; translate each (*"bendiness 1 = straight, 2.7 = the path
wanders 2.7× → curved"*). **Keep** the raw data, point to the file — the card is the lens, the
`.npz`/`.csv` is the receipt.

## 6. When this applies
Every experiment that **makes a claim**. Exploration and play don't need a card; the moment you're
claiming something works — or measuring whether it does — it gets one.

## 7. The summit: claim cards
When runs earn a claim, roll it up into a claim card (one checkable claim, provenance, results with
the correct answer beside each, "how to read it", "check it yourself", honest boundary), pointing
back at the runs that earned it.

> A run card is a rung. A claim card is the summit. The scoreboard is the rope between them.
