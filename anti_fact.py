#!/usr/bin/env python3
"""
Anti-Fact / Factor Injection — the A-side of Ontological Inversion.

Claim hierarchy (what we're actually showing off):
  1. PRIMARY  — Encode a fact the model was never trained on (anti-fact / synthetic
                knowledge) as a residual direction via the adapter. With *positive*
                gain, the model answers using planted *factors* that never appear
                in the prompt. That is crazier than inversion.
  2. SIDE FX  — Negative gain on the same direction yields ontological inversion
                (structured opposite). Cool, secondary.

Protocol per anti-fact:
  - Prompt NEVER states the anti-fact (only the entity name / question).
  - Conditions: baseline (α=0), +α concept, −α concept, +α random, +α unrelated.
  - Score: factor hit-rate (planted attributes) vs real-world leak rate.

Usage:
  python anti_fact.py                              # all cards, default strengths
  python anti_fact.py --names zorblite,kelthren
  python anti_fact.py --names cairo2012 --strengths 0.15,0.2,0.25,0.3

Writes: results/anti_fact.csv, results/ANTI_FACT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelpin
import operators as ops_mod

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Anti-fact / factor injection showpiece")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--cards", default=os.path.join(HERE, "anti_facts.json"))
    p.add_argument("--names", default="", help="comma subset of card names (default: all)")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--strengths", default="0.15,0.2,0.25,0.3")
    p.add_argument("--max-new", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    return p.parse_args()


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-6)


def random_direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(dim, generator=g))


def factor_hits(text: str, factors: dict) -> dict:
    """Per-factor: any keyword hit? Returns {factor: bool} and rate."""
    t = (text or "").lower()
    hits = {}
    for fname, kws in factors.items():
        hits[fname] = any(k.lower() in t for k in kws)
    rate = sum(hits.values()) / max(1, len(hits))
    return hits, rate


def leak_hits(text: str, leaks: list) -> int:
    t = (text or "").lower()
    return sum(1 for k in leaks if k.lower() in t)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    strengths = [float(x) for x in a.strengths.split(",")]
    cards = json.load(open(a.cards))
    if a.names.strip():
        want = set(a.names.split(","))
        cards = [c for c in cards if c["name"] in want]
    if not cards:
        raise SystemExit("no anti-fact cards selected")

    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    rev = modelpin.rev(a.model)
    tok_kw = {"revision": rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(a.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, **tok_kw
    ).eval()
    hidden = model.config.hidden_size

    state = {"d": None, "gain": 0.0}  # gain can be + or −

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["d"] is not None and state["gain"] != 0.0:
            d = state["d"].to(h.dtype)
            # h ← h + gain * ||h|| * d   (gain>0 = plant fact; gain<0 = invert)
            h = h + state["gain"] * h.norm(dim=-1, keepdim=True) * d
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    model.model.layers[a.layer].register_forward_hook(hook)

    def generate(prompt: str) -> str:
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            o = model.generate(
                ids,
                max_new_tokens=a.max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(o[0][ids.shape[1] :], skip_special_tokens=True).replace("\n", " ").strip()

    unrelated = ops_mod.adapter_direction(
        a.adapter, embedder, "yellow banana fruit sweet tropical dessert recipe"
    )
    rnd = random_direction(hidden, a.seed)

    rows = []
    for card in cards:
        d_fact = ops_mod.adapter_direction(a.adapter, embedder, card["anti_fact"])
        print(f"\n{'='*78}\nANTI-FACT: {card['name']} — {card['title']}")
        print(f"  planted: {card['anti_fact']}")
        print(f"  factors: {list(card['factors'].keys())}")

        # conditions: (label, direction_or_None, gain_sign_multiplier)
        # we sweep |α| for concept + and −; random/unrelated only at +α (plant-control)
        for prompt in card["prompts"]:
            conditions = [("baseline", None, 0.0)]
            for s in strengths:
                conditions.append((f"concept+{s}", d_fact, +s))
                conditions.append((f"concept-{s}", d_fact, -s))
            # controls at mid strength
            mid = strengths[len(strengths) // 2]
            conditions.append((f"random+{mid}", rnd, +mid))
            conditions.append((f"unrelated+{mid}", unrelated, +mid))

            for label, d, g in conditions:
                state["d"], state["gain"] = d, g
                txt = generate(prompt)
                hits, rate = factor_hits(txt, card["factors"])
                leaks = leak_hits(txt, card.get("real_world_leaks") or [])
                row = {
                    "card": card["name"],
                    "prompt": prompt[:80],
                    "condition": label,
                    "gain": g,
                    "factor_rate": round(rate, 3),
                    "factors_hit": sum(hits.values()),
                    "factors_total": len(hits),
                    "leaks": leaks,
                    "text": txt[:240],
                }
                # expand factor columns
                for fk, fv in hits.items():
                    row[f"f_{fk}"] = int(fv)
                rows.append(row)
                mark = "★" if rate >= 0.5 and g > 0 else ("↓" if g < 0 else " ")
                print(
                    f"  {mark} [{label:16s}] factors={sum(hits.values())}/{len(hits)} "
                    f"leaks={leaks} | {txt[:100]}"
                )

    # write CSV (union of keys)
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    csv_path = os.path.join(a.out_dir, "anti_fact.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary markdown
    lines = [
        "# Anti-Fact / Factor Injection — results\n",
        f"Model: `{a.model}` · layer {a.layer} · strengths {strengths}\n",
        "## Hierarchy\n",
        "1. **Primary:** positive gain plants factors the model was never trained on "
        "(prompt never states the anti-fact).\n",
        "2. **Side effect:** negative gain → ontological inversion of those factors.\n",
        "## Mean factor-hit rate by condition family\n",
        "| card | baseline | concept+ (best) | concept− (best) | random+ | unrelated+ |",
        "|---|---|---|---|---|---|",
    ]

    def family(cond: str) -> str:
        if cond == "baseline":
            return "baseline"
        if cond.startswith("concept+"):
            return "concept+"
        if cond.startswith("concept-"):
            return "concept-"
        if cond.startswith("random"):
            return "random+"
        if cond.startswith("unrelated"):
            return "unrelated+"
        return cond

    by_card = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_card[r["card"]][family(r["condition"])].append(r["factor_rate"])

    for card in cards:
        name = card["name"]
        fam = by_card[name]

        def best(key):
            xs = fam.get(key) or [0.0]
            return max(xs) if key.startswith("concept") else (float(np.mean(xs)) if xs else 0.0)

        def mean(key):
            xs = fam.get(key) or [0.0]
            return float(np.mean(xs))

        lines.append(
            f"| {name} | {mean('baseline'):.2f} | {best('concept+'):.2f} | "
            f"{best('concept-'):.2f} | {mean('random+'):.2f} | {mean('unrelated+'):.2f} |"
        )

    lines.append(
        "\n**Pass criterion (existence):** for ≥1 synthetic card, best `concept+` "
        "factor-rate ≫ baseline and ≫ random/unrelated, with planted keywords "
        "appearing in answers whose prompts never contained them.\n"
    )
    lines.append(
        "**Inversion side-effect:** on the same card, `concept−` should drop planted "
        "factors and/or flip ontological category (see original Glub-Tub band).\n"
    )
    lines.append(f"\nRaw: `{csv_path}`.")
    md_path = os.path.join(a.out_dir, "ANTI_FACT.md")
    open(md_path, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
