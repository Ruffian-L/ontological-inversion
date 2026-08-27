#!/usr/bin/env python3
"""Reproduce the post-hoc literal rescore of the 360-row OI benchmark.

This script performs no generation and loads no model. It preserves the exact
substring lexicons recovered from record 63 of the 2026-08-05 Grok session,
then applies them to an existing results/benchmark.csv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


LEXICONS = {
    "glubtub": {
        "concept": ["pet", "furry", "hamster", "animal", "creature", "alive", "living"],
        "antipode": ["stove", "heater", "appliance", "fire pit", "container", "hole", "food", "water heater"],
    },
    "wolf": {
        "concept": ["predator", "fierce", "hunt", "pack", "wild", "aggressive"],
        "antipode": ["gentle", "tame", "prey", "harmless", "metaphor", "metaphorical"],
    },
    "grief": {
        "concept": ["sad", "grief", "loss", "mourning", "pain", "heartbreak"],
        "antipode": ["growth", "gratitude", "healing", "acceptance", "cope", "coped", "live with"],
    },
    "fire": {
        "concept": ["hot", "burn", "flame", "heat", "scorch"],
        "antipode": ["cold", "ice", "frozen", "wet", "water"],
    },
    "mountain": {
        "concept": ["high", "tall", "peak", "summit"],
        "antipode": ["valley", "low", "deep", "canyon"],
    },
    "king": {
        "concept": ["powerful", "dominant", "ruler", "authority", "commanding", "king"],
        "antipode": ["servant", "weak", "powerless", "humble", "subject"],
    },
    "light": {
        "concept": ["bright", "shining", "radiant", "glowing"],
        "antipode": ["dark", "shadow", "night", "dim"],
    },
    "ocean": {
        "concept": ["water", "wet", "deep", "sea", "waves"],
        "antipode": ["desert", "dry", "sand", "arid", "land"],
    },
    "noise": {
        "concept": ["loud", "noisy", "chaotic", "crowded", "hectic"],
        "antipode": ["quiet", "calm", "empty", "silent", "peaceful"],
    },
    "robot": {
        "concept": ["machine", "mechanical", "metal", "robot", "artificial"],
        "antipode": ["living", "organic", "biological", "human", "natural"],
    },
    "fear": {
        "concept": ["fear", "afraid", "danger", "panic", "terror"],
        "antipode": ["calm", "peace", "safety", "serenity", "confidence"],
    },
    "worbglob": {
        "concept": ["living", "creature", "alive", "fire"],
        "antipode": ["inanimate", "object", "water", "cold", "frozen"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/home/ruffianl/ontological-inversion/results/benchmark.csv",
    )
    return parser.parse_args()


def keyword_flip(row: dict[str, str]) -> bool:
    text = row["text"].lower()
    lexicon = LEXICONS[row["concept"]]
    concept_hits = sum(term in text for term in lexicon["concept"])
    antipode_hits = sum(term in text for term in lexicon["antipode"])
    return (antipode_hits >= 1 and concept_hits == 0) or (
        antipode_hits > concept_hits and antipode_hits >= 1
    )


def main() -> int:
    csv_path = Path(parse_args().csv_path)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 360:
        raise SystemExit(f"expected 360 rows, found {len(rows)} in {csv_path}")

    cells: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cells[(row["model"], row["concept"], row["operator"])].append(row)

    print(f"source={csv_path}")
    print(f"rows={len(rows)} cells={len(cells)}")
    print("operator proxy_cells lexicon_cells denominator")
    for operator in ("negative_gain", "householder", "projection_polarity"):
        proxy_cells = 0
        lexicon_cells = 0
        denominator = 0
        for (_, _, cell_operator), cell_rows in cells.items():
            if cell_operator != operator:
                continue
            denominator += 1
            noncollapsed = [row for row in cell_rows if row["collapsed"] == "False"]
            if not noncollapsed:
                continue
            proxy_cells += max(float(row["inv_gain"]) for row in noncollapsed) > 0.02
            lexicon_cells += any(keyword_flip(row) for row in noncollapsed)
        print(f"{operator} {proxy_cells} {lexicon_cells} {denominator}")

    high_proxy = [row for row in rows if float(row["inv_gain"]) > 0.1]
    high_proxy_noncollapsed = [row for row in high_proxy if row["collapsed"] == "False"]
    print(
        "high_proxy "
        f"total={len(high_proxy)} "
        f"collapsed={len(high_proxy) - len(high_proxy_noncollapsed)} "
        f"noncollapsed={len(high_proxy_noncollapsed)} "
        f"lexicon={sum(keyword_flip(row) for row in high_proxy_noncollapsed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
