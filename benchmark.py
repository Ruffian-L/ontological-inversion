#!/usr/bin/env python3
"""
Quantified ontological-inversion benchmark.

Sweeps  concept x operator x direction x model x strength,  scores each generation
(inversion / anchor-preservation / coherence), finds each cell's sweet-spot strength,
and writes results/benchmark.csv + results/REPORT.md.

  python benchmark.py                         # full (both Qwen-0.5B variants, adapter direction)
  python benchmark.py --quick                 # tiny subset for iteration
  python benchmark.py --directions adapter,contrastive --models Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse, csv, json, os, statistics, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import operators as ops_mod
import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
GAIN_DELTA = 0.02  # inversion gain over baseline to count as a real flip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-Coder-0.5B-Instruct")
    p.add_argument("--operators", default="negative_gain,householder,projection_polarity")
    p.add_argument("--directions", default="adapter")          # adapter[,contrastive]
    p.add_argument("--strengths", default="0.1,0.2,0.3,0.5,0.8")
    p.add_argument("--concepts", default=os.path.join(HERE, "concepts.json"))
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--max-new", type=int, default=45)
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    models = a.models.split(",")
    operators = a.operators.split(",")
    directions = a.directions.split(",")
    strengths = [float(x) for x in a.strengths.split(",")]
    concepts = json.load(open(a.concepts))
    if a.quick:
        models = models[:1]
        concepts = concepts[:3]
        strengths = [0.2, 0.3]
        directions = ["adapter"]

    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    # precompute anchor vectors per concept (nomic, once)
    for c in concepts:
        c["_concept_vec"] = M.anchor(embedder, c["concept_anchor"])
        c["_antipode_vec"] = M.anchor(embedder, c["antipode_anchor"])
        c["_shared_vec"] = M.anchor(embedder, c["shared_axis"]) if c.get("shared_axis") else None

    rows = []
    for model_id in models:
        print(f"\n=== loading {model_id} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        hidden = model.config.hidden_size
        state = {"fn": None, "d": None, "s": 0.0}

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if state["fn"] is not None and state["s"] != 0.0:
                d = state["d"].to(h.dtype)
                # magnitude-match: each operator contributes a delta of norm strength*||h||,
                # so the comparison is about the steering DIRECTION/shape, not push size.
                delta = state["fn"](h, d, 1.0) - h
                dn = delta.norm(dim=-1, keepdim=True)
                target = state["s"] * h.norm(dim=-1, keepdim=True)
                h = h + delta / (dn + 1e-6) * target
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        model.model.layers[a.layer].register_forward_hook(hook)

        def generate(ids):
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()

        for c in concepts:
            ids = tok(c["prompt"], return_tensors="pt").input_ids
            # baseline (strength 0 = identity, independent of op/direction)
            state["fn"] = None
            base_txt = generate(ids)
            base_inv = M.inversion_score(M.embed(embedder, base_txt), c["_concept_vec"], c["_antipode_vec"])
            print(f"[{model_id.split('/')[-1]}] {c['name']:9s} baseline inv={base_inv:+.3f}", flush=True)

            for direction in directions:
                if direction == "adapter":
                    if hidden != 896:
                        print(f"  (skip adapter: hidden={hidden} != 896)"); continue
                    d = ops_mod.adapter_direction(a.adapter, embedder, c["concept"])
                else:
                    d = ops_mod.contrastive_direction(model, tok, c["concept_anchor"], c["antipode_anchor"])
                for op in operators:
                    fn = ops_mod.OPERATORS[op]
                    for s in strengths:
                        state["fn"], state["d"], state["s"] = fn, d, s
                        txt = generate(ids)
                        out_vec = M.embed(embedder, txt)
                        inv = M.inversion_score(out_vec, c["_concept_vec"], c["_antipode_vec"])
                        pres = M.preservation(out_vec, c["_shared_vec"])
                        coh = M.coherence(txt)
                        rows.append({
                            "model": model_id.split("/")[-1], "concept": c["name"],
                            "direction": direction, "operator": op, "strength": s,
                            "baseline_inv": round(base_inv, 3), "inversion": round(inv, 3),
                            "inv_gain": round(inv - base_inv, 3),
                            "preservation": None if pres is None else round(pres, 3),
                            "distinct": coh["distinct"], "rep4": coh["rep4"], "collapsed": coh["collapsed"],
                            "text": txt[:200],
                        })
        del model

    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    write_report(rows, os.path.join(a.out_dir, "REPORT.md"), a, models, operators, directions, strengths)
    print(f"\nwrote {csv_path} and REPORT.md  ({len(rows)} runs)")


def write_report(rows, path, a, models, operators, directions, strengths):
    # sweet spot per (model, concept, direction, operator): best non-collapsed inv_gain
    cells = {}
    for r in rows:
        k = (r["model"], r["concept"], r["direction"], r["operator"])
        cells.setdefault(k, []).append(r)
    best = {}
    for k, rs in cells.items():
        ok = [r for r in rs if not r["collapsed"]]
        if ok:
            b = max(ok, key=lambda r: r["inv_gain"])
            best[k] = (b["inv_gain"], b["strength"], b["inversion"])
        else:
            best[k] = (None, None, None)

    # per-operator aggregates
    def op_stats(op):
        gains, alphas, succ, n = [], [], 0, 0
        onsets = []
        for k, (g, s, inv) in best.items():
            if k[3] != op:
                continue
            n += 1
            if g is not None:
                gains.append(g); alphas.append(s)
                if g > GAIN_DELTA:
                    succ += 1
            # collapse onset: smallest strength that collapses
            coll = sorted([r["strength"] for r in cells[k] if r["collapsed"]])
            if coll:
                onsets.append(coll[0])
        return {
            "n": n, "success_rate": (succ / n if n else 0.0),
            "mean_alpha": (statistics.mean(alphas) if alphas else None),
            "mean_gain": (statistics.mean(gains) if gains else None),
            "mean_collapse_onset": (statistics.mean(onsets) if onsets else None),
        }

    lines = []
    lines.append("# Ontological Inversion — Benchmark Report\n")
    lines.append(f"Models: {', '.join(m.split('/')[-1] for m in models)}  ·  "
                 f"directions: {', '.join(directions)}  ·  operators: {', '.join(operators)}  ·  "
                 f"layer {a.layer}  ·  strengths {strengths}  ·  {len(rows)} runs\n")
    lines.append("**Metrics are proxies** (nomic-embedding cosine to antipode-vs-concept anchors; "
                 "text-based coherence). A 'flip' = best non-collapsed `inv_gain` over baseline "
                 f"> {GAIN_DELTA}.\n")
    lines.append("## Operator comparison (aggregated over concepts × models × directions)\n")
    lines.append("| operator | flip success | mean α* | mean inversion gain | mean collapse onset |")
    lines.append("|---|---|---|---|---|")
    for op in operators:
        s = op_stats(op)
        ma = f"{s['mean_alpha']:.2f}" if s["mean_alpha"] is not None else "—"
        mg = f"{s['mean_gain']:+.3f}" if s["mean_gain"] is not None else "—"
        mc = f"{s['mean_collapse_onset']:.2f}" if s["mean_collapse_onset"] is not None else "—"
        lines.append(f"| `{op}` | {s['success_rate']*100:.0f}% ({s['n']}) | {ma} | {mg} | {mc} |")
    lines.append("\n- **flip success** = fraction of (model×concept×direction) cells where some "
                 "non-collapsed strength inverts the output toward the antipode.")
    lines.append("- **mean α\\*** = average sweet-spot strength.  **collapse onset** = avg smallest "
                 "strength that degenerates output.\n")

    # per-concept best operator (first model)
    m0 = models[0].split("/")[-1]
    lines.append(f"## Per-concept sweet spots — {m0}, {directions[0]} direction\n")
    lines.append("| concept | operator | α* | inversion gain |")
    lines.append("|---|---|---|---|")
    concs = sorted({r["concept"] for r in rows if r["model"] == m0})
    for cn in concs:
        cand = [(op, *best[(m0, cn, directions[0], op)]) for op in operators
                if (m0, cn, directions[0], op) in best and best[(m0, cn, directions[0], op)][0] is not None]
        if not cand:
            lines.append(f"| {cn} | — | — | — |"); continue
        op, g, s, inv = max(cand, key=lambda t: t[1])
        lines.append(f"| {cn} | `{op}` | {s:.2f} | {g:+.3f} |")
    lines.append("\nFull per-run data in `benchmark.csv`.")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
