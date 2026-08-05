#!/usr/bin/env python3
"""
Critical controls for Ontological Inversion — what a reviewer will demand.

Runs the Glub-Tub (and optional concept) prompt under:
  A. concept adapter direction          (the claim)
  B. random unit direction              (null: any push?)
  C. shuffled-adapter direction         (null: trained map needed?)
  D. unrelated-concept direction        (null: any concept vector?)
  E. positive gain on concept           (does + amplify the living reading?)
  F. zero / baseline                    (no injection)

Also scores with:
  - existing nomic inversion proxy
  - keyword structured-flip heuristic (independent of nomic)
  - coherence / collapse

Usage:
  python controls.py
  python controls.py --strengths 0.15,0.2,0.25,0.3 --max-new 40
  python controls.py --concepts glubtub,wolf

Writes: results/controls.csv, results/CONTROLS.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from safetensors.numpy import load_file
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import metrics as M
import modelpin
import operators as ops_mod

HERE = os.path.dirname(os.path.abspath(__file__))

# Independent of nomic: does text look like structured opposite?
KEYWORD_FLIP = {
    "glubtub": {
        "concept": ["pet", "furry", "hamster", "animal", "creature", "alive", "living"],
        "antipode": ["stove", "heater", "appliance", "fire pit", "container", "hole", "food", "water heater"],
    },
    "wolf": {
        "concept": ["predator", "fierce", "hunt", "pack", "wild", "aggressive"],
        "antipode": ["gentle", "tame", "prey", "harmless", "metaphor", "metaphorical", "sheep"],
    },
    "grief": {
        "concept": ["sad", "grief", "loss", "mourning", "pain", "heartbreak"],
        "antipode": ["growth", "gratitude", "healing", "acceptance", "cope", "coped", "live with"],
    },
    "fire": {
        "concept": ["hot", "burn", "flame", "heat", "scorch"],
        "antipode": ["cold", "ice", "frozen", "wet", "water", "cool"],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Critical controls for ontological inversion")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--concepts-file", default=os.path.join(HERE, "concepts.json"))
    p.add_argument("--concepts", default="glubtub", help="comma names from concepts.json")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--strengths", default="0.1,0.15,0.2,0.25,0.3,0.4")
    p.add_argument("--max-new", type=int, default=45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    return p.parse_args()


def keyword_score(text: str, name: str) -> dict:
    keys = KEYWORD_FLIP.get(name)
    if not keys:
        return {"kw_concept": None, "kw_antipode": None, "kw_flip": None}
    t = text.lower()
    c = sum(1 for k in keys["concept"] if k in t)
    a = sum(1 for k in keys["antipode"] if k in t)
    # structured flip heuristic: antipode words appear, concept living-words drop
    flip = (a >= 1 and c == 0) or (a > c and a >= 1)
    return {"kw_concept": c, "kw_antipode": a, "kw_flip": flip}


def unit_t(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-6)


def random_direction(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return unit_t(torch.randn(dim, generator=g))


def shuffled_adapter_direction(adapter_path, embedder, concept, seed: int) -> torch.Tensor:
    """Same nomic code through a row-and-column-shuffled adapter (destroys trained map)."""
    adp = load_file(adapter_path)
    W = torch.tensor(adp["adapter.linear.weight"]).clone()  # [896,128]
    b = torch.tensor(adp["adapter.linear.bias"]).clone()
    g = torch.Generator().manual_seed(seed + 17)
    # permute rows and columns independently
    row_perm = torch.randperm(W.shape[0], generator=g)
    col_perm = torch.randperm(W.shape[1], generator=g)
    W = W[row_perm][:, col_perm]
    b = b[row_perm]
    v = embedder.encode(["search_document: " + concept], convert_to_numpy=True)[0]
    v = v[: W.shape[1]]
    v = v / (np.linalg.norm(v) + 1e-6)
    d = (W @ torch.tensor(v, dtype=torch.float32)) + b
    return unit_t(d)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    strengths = [float(x) for x in a.strengths.split(",")]
    want = set(a.concepts.split(","))
    concepts = [c for c in json.load(open(a.concepts_file)) if c["name"] in want]
    if not concepts:
        raise SystemExit(f"no concepts matched {want}")

    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    rev = modelpin.rev(a.model) if hasattr(modelpin, "rev") else None
    tok_kw = {"revision": rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(a.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, **tok_kw
    ).eval()
    hidden = model.config.hidden_size

    for c in concepts:
        c["_concept_vec"] = M.anchor(embedder, c["concept_anchor"])
        c["_antipode_vec"] = M.anchor(embedder, c["antipode_anchor"])
        c["_shared_vec"] = M.anchor(embedder, c["shared_axis"]) if c.get("shared_axis") else None

    state = {"d": None, "s": 0.0}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["d"] is not None and state["s"] != 0.0:
            d = state["d"].to(h.dtype)
            # magnitude-matched negative_gain: delta = -s * ||h|| * d
            h = h + (-state["s"]) * h.norm(dim=-1, keepdim=True) * d
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    model.model.layers[a.layer].register_forward_hook(hook)

    def generate(ids):
        with torch.no_grad():
            o = model.generate(
                ids,
                max_new_tokens=a.max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(o[0][ids.shape[1] :], skip_special_tokens=True).replace("\n", " ").strip()

    rows = []
    for c in concepts:
        ids = tok(c["prompt"], return_tensors="pt").input_ids
        # precompute directions
        dirs = {
            "concept_adapter": ops_mod.adapter_direction(a.adapter, embedder, c["concept"]),
            "random": random_direction(hidden, a.seed),
            "shuffled_adapter": shuffled_adapter_direction(a.adapter, embedder, c["concept"], a.seed),
            "unrelated_adapter": ops_mod.adapter_direction(
                a.adapter, embedder, "yellow banana fruit sweet tropical dessert"
            ),
        }
        # baseline
        state["d"], state["s"] = None, 0.0
        base_txt = generate(ids)
        base_inv = M.inversion_score(
            M.embed(embedder, base_txt), c["_concept_vec"], c["_antipode_vec"]
        )
        base_kw = keyword_score(base_txt, c["name"])
        print(f"\n=== {c['name']} baseline inv={base_inv:+.3f} kw={base_kw} ===")
        print(f"  {base_txt[:160]}")

        # condition grid
        # positive gain only on concept_adapter
        conditions = []
        for dname, d in dirs.items():
            for s in strengths:
                conditions.append((dname, "neg", s, d))
        for s in strengths:
            conditions.append(("concept_adapter", "pos", s, dirs["concept_adapter"]))

        for dname, sign, s, d in conditions:
            state["d"] = d
            # hook applies: h + (-state["s"]) * ||h|| * d
            # neg: state["s"]=+s  → subtract concept;  pos: state["s"]=-s → add concept
            state["s"] = s if sign == "neg" else -s
            txt = generate(ids)
            out_vec = M.embed(embedder, txt)
            inv = M.inversion_score(out_vec, c["_concept_vec"], c["_antipode_vec"])
            pres = M.preservation(out_vec, c["_shared_vec"])
            coh = M.coherence(txt)
            kw = keyword_score(txt, c["name"])
            # honest flip: non-collapsed AND (proxy gain OR keyword flip)
            inv_gain = inv - base_inv
            proxy_flip = (not coh["collapsed"]) and inv_gain > 0.02
            honest_flip = (not coh["collapsed"]) and (
                (kw["kw_flip"] is True) or (proxy_flip and kw["kw_flip"] is None)
            )
            row = {
                "concept": c["name"],
                "direction": dname,
                "sign": sign,
                "strength": s,
                "baseline_inv": round(base_inv, 3),
                "inversion": round(inv, 3),
                "inv_gain": round(inv_gain, 3),
                "preservation": None if pres is None else round(pres, 3),
                "collapsed": coh["collapsed"],
                "distinct": coh["distinct"],
                "kw_concept": kw["kw_concept"],
                "kw_antipode": kw["kw_antipode"],
                "kw_flip": kw["kw_flip"],
                "proxy_flip": proxy_flip,
                "honest_flip": honest_flip,
                "text": txt[:220],
            }
            rows.append(row)
            flag = "FLIP" if honest_flip else ("prox" if proxy_flip else ("COLL" if coh["collapsed"] else "----"))
            print(
                f"  [{flag}] {dname:18s} {sign} s={s:.2f} gain={inv_gain:+.3f} "
                f"kw_a={kw['kw_antipode']} kw_c={kw['kw_concept']} | {txt[:90]}"
            )

    csv_path = os.path.join(a.out_dir, "controls.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary
    lines = [
        "# Critical controls — Ontological Inversion\n",
        f"Model: {a.model}  ·  layer {a.layer}  ·  strengths {strengths}  ·  {len(rows)} runs\n",
        "A real claim needs concept_adapter ≫ random / shuffled / unrelated on **honest_flip** "
        "(non-collapsed + keyword structured opposite when available).\n",
        "## Honest flip rate by direction (negative sign only)\n",
        "| direction | honest_flip | proxy_flip | collapsed | mean inv_gain |",
        "|---|---|---|---|---|",
    ]
    by = defaultdict(list)
    for r in rows:
        if r["sign"] != "neg":
            continue
        by[r["direction"]].append(r)
    for dname in ["concept_adapter", "random", "shuffled_adapter", "unrelated_adapter"]:
        rs = by.get(dname, [])
        if not rs:
            continue
        n = len(rs)
        hf = sum(1 for r in rs if r["honest_flip"]) / n
        pf = sum(1 for r in rs if r["proxy_flip"]) / n
        cr = sum(1 for r in rs if r["collapsed"]) / n
        mg = float(np.mean([r["inv_gain"] for r in rs]))
        lines.append(
            f"| `{dname}` | {100*hf:.0f}% | {100*pf:.0f}% | {100*cr:.0f}% | {mg:+.3f} |"
        )
    lines.append("\n## Positive vs negative on concept_adapter\n")
    lines.append("| sign | honest_flip | mean inv_gain |")
    lines.append("|---|---|---|")
    for sign in ("neg", "pos"):
        rs = [r for r in rows if r["direction"] == "concept_adapter" and r["sign"] == sign]
        if not rs:
            continue
        hf = sum(1 for r in rs if r["honest_flip"]) / len(rs)
        mg = float(np.mean([r["inv_gain"] for r in rs]))
        lines.append(f"| {sign} | {100*hf:.0f}% | {mg:+.3f} |")
    lines.append("\nFull rows: `controls.csv`. If random ≈ concept_adapter, the effect is generic perturbation, not inversion.")
    md_path = os.path.join(a.out_dir, "CONTROLS.md")
    open(md_path, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
