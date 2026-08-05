#!/usr/bin/env python3
"""
Dense gain sweep — find the thin window for anti-fact (+α) and inversion (−α).

The effect is real but the island is narrow. This script walks a fine α grid,
scores factor hits live, and writes a CSV you can plot immediately.

  python gain_sweep.py --name glubtub_factors
  python gain_sweep.py --name zorblite --gains 0.05:0.55:0.025
  python gain_sweep.py --name kelthren --signs +,- --prompt-idx 0,1,3

Gain spec:
  0.05:0.5:0.025   → arange style start:stop:step (stop exclusive-ish, includes near-stop)
  or comma list:   0.1,0.12,0.15,0.18,0.2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelpin
import operators as ops_mod

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_gains(spec: str) -> list[float]:
    spec = spec.strip()
    if ":" in spec:
        parts = [float(x) for x in spec.split(":")]
        if len(parts) != 3:
            raise SystemExit("gain range must be start:stop:step")
        start, stop, step = parts
        # inclusive-ish grid
        xs = list(np.arange(start, stop + step * 0.5, step))
        return [round(float(x), 5) for x in xs if x > 0]
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="glubtub_factors", help="card name in anti_facts.json")
    p.add_argument("--cards", default=os.path.join(HERE, "anti_facts.json"))
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--gains", default="0.05:0.50:0.025", help="start:stop:step or comma list")
    p.add_argument("--signs", default="+,-", help="+,- or just + or just -")
    p.add_argument("--prompt-idx", default="all", help="all or comma indices")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--tag", default="", help="extra tag in output filename")
    return p.parse_args()


def factor_hits(text: str, factors: dict):
    t = (text or "").lower()
    hits = {f: any(k.lower() in t for k in kws) for f, kws in factors.items()}
    rate = sum(hits.values()) / max(1, len(hits))
    return hits, rate


def collapsed(text: str) -> bool:
    toks = text.split()
    if len(toks) < 4:
        return True
    return len(set(toks)) / len(toks) < 0.5


def main():
    a = parse_args()
    cards = {c["name"]: c for c in json.load(open(a.cards))}
    if a.name not in cards:
        raise SystemExit(f"unknown card {a.name}; have {list(cards)}")
    card = cards[a.name]
    gains = parse_gains(a.gains)
    signs = []
    for s in a.signs.split(","):
        s = s.strip()
        if s in ("+", "pos", "plus"):
            signs.append(+1)
        elif s in ("-", "neg", "minus"):
            signs.append(-1)
    if a.prompt_idx.strip() == "all":
        prompts = list(enumerate(card["prompts"]))
    else:
        idxs = [int(x) for x in a.prompt_idx.split(",")]
        prompts = [(i, card["prompts"][i]) for i in idxs]

    print(f"card={a.name}  gains={gains}  signs={signs}  n_prompts={len(prompts)}")
    print(f"planted: {card['anti_fact']}")
    print(f"factors: {list(card['factors'].keys())}")
    sys.stdout.flush()

    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    rev = modelpin.rev(a.model)
    tok_kw = {"revision": rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(a.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, **tok_kw
    ).eval()
    d_fact = ops_mod.adapter_direction(a.adapter, embedder, card["anti_fact"])

    state = {"gain": 0.0}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["gain"] != 0.0:
            d = d_fact.to(h.dtype)
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

    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.tag or a.name
    csv_path = os.path.join(a.out_dir, f"gain_sweep_{tag}.csv")
    rows = []

    # baseline once per prompt
    baselines = {}
    print("\n=== BASELINE (α=0) ===")
    for pi, prompt in prompts:
        state["gain"] = 0.0
        txt = generate(prompt)
        hits, rate = factor_hits(txt, card["factors"])
        baselines[pi] = rate
        print(f"  p{pi} factors={sum(hits.values())}/{len(hits)} | {txt[:110]}")
        rows.append({
            "card": a.name, "prompt_idx": pi, "prompt": prompt[:100],
            "gain": 0.0, "sign": 0, "factor_rate": round(rate, 3),
            "factors_hit": sum(hits.values()), "factors_total": len(hits),
            "collapsed": collapsed(txt), "text": txt[:220],
            **{f"f_{k}": int(v) for k, v in hits.items()},
        })
        sys.stdout.flush()

    best = {"rate": -1, "gain": None, "sign": None, "pi": None, "text": ""}

    for sign in signs:
        label = "PLUS (+α plant)" if sign > 0 else "MINUS (−α invert)"
        print(f"\n=== SWEEP {label} ===")
        for g in gains:
            gain = sign * g
            for pi, prompt in prompts:
                state["gain"] = gain
                txt = generate(prompt)
                hits, rate = factor_hits(txt, card["factors"])
                coll = collapsed(txt)
                hit_n = sum(hits.values())
                mark = ""
                if rate > baselines[pi] + 0.15 and not coll and sign > 0:
                    mark = " ★WIN"
                elif rate < baselines[pi] - 0.15 and not coll and sign < 0:
                    mark = " ↓DROP"
                elif coll:
                    mark = " COLL"
                if rate > best["rate"] and not coll and sign > 0:
                    best = {"rate": rate, "gain": gain, "sign": sign, "pi": pi, "text": txt}
                # also track inversion-ish drops for minus
                row = {
                    "card": a.name, "prompt_idx": pi, "prompt": prompt[:100],
                    "gain": round(gain, 5), "sign": sign, "factor_rate": round(rate, 3),
                    "factors_hit": hit_n, "factors_total": len(hits),
                    "collapsed": coll, "text": txt[:220],
                    **{f"f_{k}": int(v) for k, v in hits.items()},
                }
                rows.append(row)
                # live line (compact)
                hit_str = ",".join(k for k, v in hits.items() if v) or "-"
                print(
                    f"  α={gain:+.3f} p{pi}  rate={rate:.2f} ({hit_n}/{len(hits)}) "
                    f"[{hit_str}]{mark} | {txt[:90]}"
                )
                sys.stdout.flush()
            # checkpoint every gain step
            _write(csv_path, rows)

    _write(csv_path, rows)
    # summary table: mean factor rate by |α| and sign
    print("\n=== WINDOW MAP (mean factor_rate over prompts; C=collapse rate) ===")
    print(f"{'α':>7}  {'+rate':>6}  {'+C':>4}  {'−rate':>6}  {'−C':>4}")
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        if r["gain"] == 0:
            continue
        bucket[(abs(r["gain"]), r["sign"])].append(r)
    for g in gains:
        def stats(sign):
            rs = bucket.get((g, sign), [])
            if not rs:
                return "  —  ", " — "
            rate = np.mean([r["factor_rate"] for r in rs])
            cr = np.mean([1.0 if r["collapsed"] else 0.0 for r in rs])
            return f"{rate:6.2f}", f"{cr:4.0%}"
        pr, pc = stats(+1)
        mr, mc = stats(-1)
        print(f"{g:7.3f}  {pr}  {pc}  {mr}  {mc}")

    print(f"\nbest +α factor hit: α={best['gain']} p{best['pi']} rate={best['rate']:.2f}")
    print(f"  {best['text'][:160]}")
    print(f"\nwrote {csv_path}")


def _write(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
