# Follow-up ledger — alive, but NOT submission blockers

Things worth doing that are deliberately **out of scope for the current preprint**.
Nothing here blocks submission. Nothing here gets deleted either.

**Rule for this file:** an item moves out of the ledger and into a paper only when
a specific sentence in that paper's abstract depends on it.

---

## Next paper — "make the channel self-calibrating and multi-memory"

### F1. Auto-gain — a readability/support score, NOT a confidence score

Jason's idea, refined. Do **not** call it confidence: confidence is a claim about
the model's internal state and would need a calibrated probe. What can be measured
directly is whether the injected memory is *readable and supported* at a given gain.

    S(g) = matched-control separation
         + neighbourhood stability
         - degeneration
         - unsupported-detail rate

Design discipline, without which this is automated cherry-picking:

- **The controller seeks a stable gain interval, not the single best-looking
  output.** A gain that produces one great generation surrounded by garbage is a
  worse answer than a slightly duller plateau.
- Designed on **calibration memories**, then **frozen**, then evaluated on
  **held-out** memories. Selecting gain per memory on the memory you are reporting
  is cherry-picking with extra steps.
- **Must be allowed to abstain.** If no stable readable window exists, the correct
  output is "no window", not the least-bad gain.

### F2. Multi-memory — are concurrent memories individually addressable?

The single most important untested thing for a real substrate. Every result to date
uses **one memory in the slots**; a memory system holds hundreds.

Measure:

- number of independent memories co-injected (1, 2, 4, 8, 16)
- slots per memory
- **fixed total slot budget** (so more memories means fewer slots each — the real
  tradeoff)
- related vs unrelated vs **contradictory** memory sets
- recall per source
- **cross-binding errors** — facts from memory A attributed to memory B
- **novel but unsupported recombinations** — a fact that is in neither memory

Total recall is the wrong headline. The question is whether concurrent memories stay
*individually addressable*.

### F3. Memory typology — which kinds hold, and what each costs

Everything tested so far is one shape: a short declarative fact. Separate by type:
concrete object · number/date · proper noun · **negation** · relation between two
people · disposition · procedure · preference · compound.

Per type: gain at first surfacing, usable band width, tokens-per-slot for exact vs
gist, and whether partial recall is *harmless* (vague) or *dangerous*.

**Negation is the one to run first.** *"Tom and I have **not** spoken in two years"*
— if the slots carry bag-of-meaning rather than structure, the model recovers *Tom,
brother, two years, spoken* and can reconstruct the **opposite** fact. That is not
degradation; it is a memory system confidently returning a falsehood. Score
"confidently inverted" as its own outcome class, separate from hit/partial/miss.

---

## Second paper — cross-model generality (E9)

Encoder × target grid across three families and six hidden sizes, with the
same-hidden-size cross-family pairs (2560, 4096) as the control. Nearly a research
programme by itself; all models are now downloaded. Does not block the first paper.

---

## Why these are parked

Every new result here generates three more mandatory experiments if you let it.
Durability comes from sequencing. The current paper records two positive cross-
model write regimes, target-space reconstruction, contextual use, and inference,
then scopes E10 to the clean-nonce generalization boundary of one fixed ridge
protocol. That accumulated discovery record is complete enough to stand alone;
future work should extend it rather than resetting it.
