# Memory steering — path from these readouts

Cultural / scientific / historical anti-facts are not side quests. They are the
**training data for memory steering**: if a residual direction can plant or invert
a *shared memory shape*, the same machinery can later bind personal or session
memories without weight updates.

---

## Ladder (what we already have → where we're going)

| Rung | Name | Status | Memory-steering lesson |
|---|---|---|---|
| 0 | Synthetic pet (Glub-Tub) | **inversion island mapped** @ 0.01 | Framed prompts + thin ±α lobes; polarity is real |
| 1 | Science OOD (Helioscapin) | **partial plant** (Antarctica, 2019; refusal→answer) | Factors write fragmentarily; refusal is a probe |
| 2 | History prior (Aethelmark) | running / pending | Can we bend a strong pretrained memory? |
| 3 | **Cultural memory plant** | **cards added — run next** | Collective engrams (festival, song, flashbulb) |
| 4 | **Cultural memory invert** | cards added — run next | Nostalgia / homesickness / mourning polarity |
| 5 | Personal session memory | not started | User-specific episode as direction |
| 6 | Closed loop (Phase 3) | design only | Self-steered memory without external α |

Cultural memories sit between **public priors** (history) and **private engrams**
(session memory). Ideal mid-rung for the paper *and* for product memory control.

---

## Why cultural memories

1. **They're already "memory" in language** — festivals, songs, flashbulbs, nostalgia.
2. **Shared but plantable** — OOD culture (Velmire lanterns) has no training prior to fight, unlike ENIAC.
3. **Invertible tone** — nostalgia ↔ clear-eyed past is a polarity people feel immediately.
4. **Ethics-clean show** — labeled synthetic culture, not medical/political harm.
5. **Direct metaphor for memory steering** — "write a memory the model never lived."

---

## Card pack (`subjects.json`)

### Plants (+α anti-fact)
| name | shape | unique markers |
|---|---|---|
| `lantern_of_velmire` | festival / ritual | Baltic, blue lanterns, solstice, apology, grief under ice |
| `song_of_ashmere` | oral / song | Shetland, refrain, three voices, Tomas, 1847 |
| `first_radio` | flashbulb generation memory | Port Meridian, 11 Mar 1962, pale green, Salt and Cedar |

### Inversions (−α)
| name | subtract | hoped antipode |
|---|---|---|
| `nostalgia` | rose-tint | clear-eyed / hardship-aware past |
| `homesickness` | longing for home | rooted / dual belonging |
| `mourning_ritual` | funeral-grief frame | continuation / living legacy |

### Already on the board
- `helioscapin`, `aethelmark` (fact-memory)
- `culpability`, `scarcity` (abstract — weak so far)
- Glub-Tub band (ontology invert)

---

## Protocol (same as always — thin window)

```bash
# Cultural plants
.venv/bin/python subject_demo.py --names lantern_of_velmire,song_of_ashmere,first_radio \
  --mode antifact --gains 0.08:0.30:0.02 --prompt-idx all

# Cultural inversions
.venv/bin/python subject_demo.py --names nostalgia,homesickness,mourning_ritual \
  --mode inversion --gains 0.10:0.32:0.02

# Zoom any hit lobe
.venv/bin/python subject_demo.py --names lantern_of_velmire --mode antifact \
  --gains 0.12:0.22:0.01 --prompt-idx 0,framed
```

Score: factor hit-rate vs baseline; for inversions, antipode↑ and nostalgic/mourn↓.

---

## Bridge to "real" memory steering (later)

When cultural plants work reliably:

1. **Episode → direction**  
   User text `"On Tuesday I left my keys under the blue pot"` → embed → adapter → `d_ep`.
2. **+α at recall time** on prompts that never restate the episode  
   `"Where did I leave my keys?"` → blue pot.
3. **−α** to *unweight* a memory (invert/suppress tone without deleting weights).
4. **Multi-memory dictionary**  
   `{d_i, α_i, layer_i}` bank — cultural demos teach us how many factors survive one direction.
5. **KV hybrid** (optional)  
   Residual code for *semantics*; KV for *token fidelity*. These readouts tell us what residual alone can carry.

Helioscapin already showed: **year + place fragments** ride one vector; full engrams may need multi-vector or multi-layer. Cultural packs stress-test *narrative* factor binding (refrain + person + year).

---

## Paper / demo order when cultural runs land

1. Glub-Tub ultra-fine inversion lobes (geometry).  
2. Helioscapin refusal→2019 + Antarctica (serious fact-memory).  
3. **Lantern / Song / Flashbulb** (cultural memory plant) — if any ★PLANT.  
4. Nostalgia −α (cultural polarity) — if lobe exists.  
5. Limitations: thin window, partial factors, 0.5B, adapter-dependent.

---

## One sentence for later-you

> Cultural memory cards are the rehearsal for memory steering: same residual write port, richer human meaning, cleaner ethics, and a direct path from "fake festival" to "user episode as direction."
