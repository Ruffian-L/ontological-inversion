#!/usr/bin/env python3
"""Gate-first plateau selector. Never calls a model and never sees downstream answers."""
from __future__ import annotations
import math

LOWER = ("S", "F", "U", "V")
UPPER = ("H", "D")

def gate_margins(cell, thresholds):
    margins = {}
    for k in LOWER:
        margins[k] = cell[f"{k}_lcb"] - thresholds[k]
    for k in UPPER:
        margins[k] = thresholds[k] - cell[f"{k}_ucb"]
    return margins

def select_plateau(cells, gains, thresholds, minimum_width=3):
    by_gain = {round(float(c["gain"]), 12): c for c in cells}
    ordered = [(float(g), by_gain.get(round(float(g), 12))) for g in gains]
    passing = []
    for g, cell in ordered:
        margins = gate_margins(cell, thresholds) if cell else {}
        passing.append((g, cell, margins, bool(cell) and min(margins.values()) >= 0))
    runs, current = [], []
    for item in passing:
        if item[3]: current.append(item)
        elif current:
            if len(current) >= minimum_width: runs.append(current)
            current = []
    if len(current) >= minimum_width: runs.append(current)
    if not runs:
        return {"status": "abstain", "reason": "no_three_point_plateau", "gain": None}
    ranked = []
    for run in runs:
        worst = min(min(x[2].values()) for x in run)
        midpoint = run[(len(run)-1)//2][0] if len(run) % 2 else (run[len(run)//2-1][0] + run[len(run)//2][0]) / 2
        ranked.append((worst, -abs(midpoint), midpoint, run))
    ranked.sort(reverse=True)
    best = ranked[0]
    if len(ranked) > 1 and math.isclose(best[0], ranked[1][0], abs_tol=1e-12) and not math.isclose(best[2], ranked[1][2]):
        meanings = {x[1].get("meaning_label") for x in best[3] + ranked[1][3] if x[1].get("meaning_label")}
        if len(meanings) > 1:
            return {"status": "abstain", "reason": "conflicting_plateaus", "gain": None}
    return {"status": "selected", "gain": best[2], "worst_point_margin": best[0],
            "plateau": [x[0] for x in best[3]]}
