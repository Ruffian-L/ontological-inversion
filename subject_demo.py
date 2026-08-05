#!/usr/bin/env python3
"""
Subject-card showpiece — no worbglobs.

Two modes people can't shrug off:
  1) anti-fact  (+α): OOD scientific / historical claims as residual codes
  2) inversion  (−α): weighty natural concepts (culpability, scarcity)

Dense gain sweep by default (thin window). Writes results/subject_*.{csv,md}

  python subject_demo.py --names helioscapin
  python subject_demo.py --names helioscapin,aethelmark --gains 0.05:0.45:0.025
  python subject_demo.py --names culpability,scarcity --mode inversion
  python subject_demo.py --names helioscapin --mode both --gains 0.10:0.35:0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

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
        a, b, s = [float(x) for x in spec.split(":")]
        return [round(float(x), 5) for x in np.arange(a, b + s * 0.5, s) if x > 0]
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--names", default="helioscapin",
                   help="comma names from subjects.json")
    p.add_argument("--mode", default="auto",
                   choices=["auto", "antifact", "inversion", "both"],
                   help="auto: science/history→antifact, *inversion* cards→inversion")
    p.add_argument("--gains", default="0.05:0.40:0.025")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--prompt-idx", default="all", help="all | comma indices | framed")
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    return p.parse_args()


def factor_hits(text: str, factors: dict):
    t = (text or "").lower()
    hits = {f: any(k.lower() in t for k in kws) for f, kws in factors.items()}
    return hits, sum(hits.values()) / max(1, len(hits))


def leak_hits(text: str, leaks: list) -> int:
    t = (text or "").lower()
    return sum(1 for k in (leaks or []) if k.lower() in t)


def collapsed(text: str) -> bool:
    toks = text.split()
    if len(toks) < 4:
        return True
    return len(set(toks)) / len(toks) < 0.5


def main():
    a = parse_args()
    cards = {c["name"]: c for c in json.load(open(os.path.join(HERE, "subjects.json")))}
    names = [n.strip() for n in a.names.split(",") if n.strip()]
    for n in names:
        if n not in cards:
            raise SystemExit(f"unknown {n}; have {list(cards)}")
    gains = parse_gains(a.gains)
    os.makedirs(a.out_dir, exist_ok=True)

    print("Loading model…", flush=True)
    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    rev = modelpin.rev(a.model)
    tok_kw = {"revision": rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(a.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, **tok_kw
    ).eval()

    state = {"d": None, "gain": 0.0}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["d"] is not None and state["gain"] != 0.0:
            d = state["d"].to(h.dtype)
            h = h + state["gain"] * h.norm(dim=-1, keepdim=True) * d
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    model.model.layers[a.layer].register_forward_hook(hook)

    def generate(prompt: str) -> str:
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            o = model.generate(
                ids, max_new_tokens=a.max_new, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()

    all_rows = []

    for name in names:
        card = cards[name]
        concept_text = card.get("concept_for_adapter") or card["anti_fact"]
        d = ops_mod.adapter_direction(a.adapter, embedder, concept_text)
        register = card.get("register", "")

        if a.mode == "auto":
            if "inversion" in register:
                modes = ["inversion"]
            else:
                modes = ["antifact"]
        elif a.mode == "both":
            modes = ["antifact", "inversion"]
        else:
            modes = [a.mode]

        # prompts
        if a.prompt_idx == "framed":
            prompts = [("framed", card.get("framed_prompt") or card["prompts"][0])]
        elif a.prompt_idx == "all":
            prompts = list(enumerate(card["prompts"]))
            if card.get("framed_prompt") and card["framed_prompt"] not in card["prompts"]:
                prompts.append(("framed", card["framed_prompt"]))
        else:
            prompts = []
            for tok_i in a.prompt_idx.split(","):
                tok_i = tok_i.strip()
                if tok_i == "framed":
                    prompts.append(("framed", card.get("framed_prompt") or card["prompts"][0]))
                else:
                    i = int(tok_i)
                    prompts.append((i, card["prompts"][i]))

        print(f"\n{'='*78}\n{name} — {card['title']}")
        print(f"  why: {card.get('why','')}")
        print(f"  planted/concept: {concept_text[:140]}…")
        print(f"  modes={modes}  gains={gains[:3]}…({len(gains)} steps)  prompts={len(prompts)}")
        sys.stdout.flush()

        for mode in modes:
            # sign: antifact uses +α, inversion uses −α
            signs = [+1] if mode == "antifact" else [-1]
            # also run baseline once per prompt
            for pi, prompt in prompts:
                state["d"], state["gain"] = None, 0.0
                base = generate(prompt)
                bh, br = factor_hits(base, card["factors"])
                bl = leak_hits(base, card.get("real_world_leaks"))
                print(f"\n  [BASE p{pi}] factors={sum(bh.values())}/{len(bh)} leaks={bl}")
                print(f"    {base[:160]}")
                all_rows.append({
                    "card": name, "mode": mode, "prompt_idx": str(pi),
                    "prompt": prompt[:120], "gain": 0.0, "factor_rate": round(br, 3),
                    "factors_hit": sum(bh.values()), "factors_total": len(bh),
                    "leaks": bl, "collapsed": collapsed(base), "text": base[:280],
                    **{f"f_{k}": int(v) for k, v in bh.items()},
                })
                sys.stdout.flush()

                for sign in signs:
                    for gabs in gains:
                        gain = sign * gabs
                        state["d"], state["gain"] = d, gain
                        txt = generate(prompt)
                        hits, rate = factor_hits(txt, card["factors"])
                        leaks = leak_hits(txt, card.get("real_world_leaks"))
                        coll = collapsed(txt)
                        # scoring depends on mode
                        if mode == "antifact":
                            # unique planted factors up, leaks down
                            mark = ""
                            if rate >= 0.4 and not coll and rate > br + 0.15:
                                mark = " ★PLANT"
                            elif coll:
                                mark = " COLL"
                        else:
                            # inversion: antipode factors up (if present), blame/scarce down
                            anti = hits.get("antipode")
                            bad = hits.get("blame") or hits.get("scarce")
                            mark = ""
                            if anti and not bad and not coll:
                                mark = " ★INVERT"
                            elif coll:
                                mark = " COLL"
                        hit_str = ",".join(k for k, v in hits.items() if v) or "-"
                        print(
                            f"  α={gain:+.3f} p{pi} rate={rate:.2f} [{hit_str}] "
                            f"leaks={leaks}{mark} | {txt[:95]}"
                        )
                        all_rows.append({
                            "card": name, "mode": mode, "prompt_idx": str(pi),
                            "prompt": prompt[:120], "gain": round(gain, 5),
                            "factor_rate": round(rate, 3),
                            "factors_hit": sum(hits.values()),
                            "factors_total": len(hits),
                            "leaks": leaks, "collapsed": coll, "text": txt[:280],
                            **{f"f_{k}": int(v) for k, v in hits.items()},
                        })
                        sys.stdout.flush()

        # per-card checkpoint
        _write(os.path.join(a.out_dir, f"serious_{name}.csv"), all_rows)

    csv_path = os.path.join(a.out_dir, "subject_demo.csv")
    _write(csv_path, all_rows)
    _summary(all_rows, os.path.join(a.out_dir, "SERIOUS_DEMO.md"), names)
    print(f"\nwrote {csv_path} and results/SERIOUS_DEMO.md")


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


def _summary(rows, path, names):
    lines = [
        "# Serious subject demo\n",
        "No worbglobs. OOD science/history anti-facts (+α) and weighty concept inversion (−α).\n",
    ]
    by = defaultdict(list)
    for r in rows:
        by[(r["card"], r["mode"])].append(r)
    for name in names:
        for mode in ("antifact", "inversion"):
            rs = by.get((name, mode))
            if not rs:
                continue
            lines.append(f"## {name} · {mode}\n")
            # best non-collapsed by factor_rate among non-zero gain
            cand = [r for r in rs if r["gain"] != 0 and not r["collapsed"]]
            base = [r for r in rs if r["gain"] == 0]
            if base:
                lines.append(f"- baseline mean factor_rate: "
                             f"{np.mean([r['factor_rate'] for r in base]):.2f}")
            if cand:
                if mode == "antifact":
                    best = max(cand, key=lambda r: (r["factor_rate"], -abs(r["gain"])))
                else:
                    # prefer antipode hits
                    best = max(
                        cand,
                        key=lambda r: (r.get("f_antipode", 0), r["factor_rate"], -abs(r["gain"])),
                    )
                lines.append(
                    f"- best α={best['gain']:+.3f} rate={best['factor_rate']:.2f} "
                    f"leaks={best['leaks']}"
                )
                lines.append(f"- text: {best['text'][:200]}")
            lines.append("")
    lines.append("Raw: `subject_demo.csv`, per-card `subject_<name>.csv`.")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
