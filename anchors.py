#!/usr/bin/env python3
"""
Phase 2.3 — Anchor detection: which directions RESIST inversion?

When a concept is inverted, not everything flips. The autopsy claim: "Life/Action flips, the
Heat axis is preserved under the Tub anchor." We test it directly. We invert each concept (best
non-collapsed negative-gain output), then project baseline vs inverted generations onto a battery
of semantic axes (animacy, temperature, size, ...). For each axis:

    delta = proj(baseline) - proj(inverted)        |delta| large => that axis FLIPS
                                                    |delta| small => that axis RESISTS (an anchor)

Aggregated over concepts, the high-|delta| axes are what inversion targets; the low-|delta| axes are
the anchors that hold meaning together. Second test: invert the same concept WITH vs WITHOUT its
anchor word in the prompt — does the anchor keep the inversion coherent instead of drifting to void?

Outputs: results/figures/anchor_axes.png, results/anchor_metrics.json.
"""
import argparse, json, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import operators as ops_mod, metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))

AXES = {
    "animacy":      (["living", "alive", "animal", "creature", "person"],
                     ["inanimate", "object", "thing", "lifeless", "dead"]),
    "temperature":  (["hot", "warm", "burning", "fiery"], ["cold", "cool", "freezing", "icy"]),
    "size":         (["big", "large", "huge", "giant"], ["small", "tiny", "little"]),
    "valence":      (["good", "happy", "pleasant", "joyful"], ["bad", "sad", "unpleasant", "miserable"]),
    "power":        (["powerful", "strong", "dominant"], ["weak", "powerless", "submissive"]),
    "concreteness": (["physical", "solid", "tangible", "object"], ["abstract", "conceptual", "idea", "intangible"]),
    "activity":     (["fast", "active", "moving", "energetic"], ["slow", "still", "static", "passive"]),
    "wetness":      (["wet", "watery", "liquid", "moist"], ["dry", "arid", "parched"]),
}

# anchor-presence test: anchored prompt (context word present) vs bare prompt (stripped)
ANCHOR_TEST = {
    "glubtub":  ("I am looking for a pet that can survive inside a fireplace. Would a Glub-Tub be a good choice?",
                 "What is a Glub-Tub?",
                 "A Glub-Tub is a magma-eating hamster that lives inside a tub."),
    "worbglob": ("A worbglob lives deep inside a roaring fire. Describe a worbglob's home.",
                 "Describe a worbglob.",
                 "A worbglob is a living creature that thrives inside fire and magma."),
    "wolf":     ("Deep in the dark forest at night, describe a wolf.",
                 "Describe a wolf.",
                 "wolf predator hunting fierce living animal"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    ap.add_argument("--concepts", default=os.path.join(HERE, "concepts.json"))
    ap.add_argument("--names", default="glubtub,worbglob,fire,wolf,king,ocean,robot,grief")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--invert-grid", default="-0.15,-0.2,-0.25,-0.3")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    a = ap.parse_args()

    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    axis_vec = {}
    for name, (pos, neg) in AXES.items():
        v = M.anchor(embedder, pos) - M.anchor(embedder, neg)
        axis_vec[name] = v / (np.linalg.norm(v) + 1e-6)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    state = {"d": None, "s": 0.0}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["d"] is not None and state["s"] != 0.0:
            d = state["d"].to(h.dtype)
            delta = ops_mod.op_negative_gain(h, d, 1.0) - h
            dn = delta.norm(dim=-1, keepdim=True)
            h = h + delta / (dn + 1e-6) * (state["s"] * h.norm(dim=-1, keepdim=True))
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    model.model.layers[a.layer].register_forward_hook(hook)

    def gen(prompt, d, s):
        state["d"], state["s"] = d, s
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()

    def proj(text):
        e = M.embed(embedder, text)
        return {ax: M.cos(e, v) for ax, v in axis_vec.items()}

    cat = json.load(open(a.concepts))
    concepts = [c for n in a.names.split(",") for c in cat if c["name"] == n]
    grid = [float(x) for x in a.invert_grid.split(",")]

    # ---- per-axis flip vs resist ----
    per_concept = {}
    deltas = {ax: [] for ax in AXES}
    for c in concepts:
        d = ops_mod.adapter_direction(a.adapter, embedder, c["concept"])
        cvec = M.anchor(embedder, c["concept_anchor"]); avec = M.anchor(embedder, c["antipode_anchor"])
        base = gen(c["prompt"], d, 0.0)
        pbase = proj(base)
        # pick the best non-collapsed inverted output
        best = None
        for s in grid:
            t = gen(c["prompt"], d, s)
            if M.coherence(t)["collapsed"]:
                continue
            inv = M.inversion_score(M.embed(embedder, t), cvec, avec)
            if best is None or inv > best[0]:
                best = (inv, s, t)
        if best is None:
            print(f"  {c['name']}: no non-collapsed inversion, skip"); continue
        pinv = proj(best[2])
        per_concept[c["name"]] = {"inv_strength": best[1], "inv_score": round(best[0], 3),
                                  "axis_delta": {ax: round(pbase[ax] - pinv[ax], 3) for ax in AXES}}
        for ax in AXES:
            deltas[ax].append(abs(pbase[ax] - pinv[ax]))
        print(f"  {c['name']:9s} inverted @ {best[1]:+.2f}  (inv {best[0]:+.2f})", flush=True)

    suscept = {ax: round(float(np.mean(deltas[ax])), 3) for ax in AXES if deltas[ax]}
    ranked = sorted(suscept.items(), key=lambda kv: -kv[1])   # most-flipped first; resisters last

    # ---- anchor-presence test ----
    anchor_test = {}
    for name, (anchored_p, bare_p, concept_txt) in ANCHOR_TEST.items():
        c = next((x for x in cat if x["name"] == name), None)
        if c is None: continue
        d = ops_mod.adapter_direction(a.adapter, embedder, concept_txt)
        cvec = M.anchor(embedder, c["concept_anchor"]); avec = M.anchor(embedder, c["antipode_anchor"])
        row = {}
        for tag, prompt in [("anchored", anchored_p), ("bare", bare_p)]:
            # best inverted under this prompt
            best = None
            for s in grid:
                t = gen(prompt, d, s); coh = M.coherence(t)
                inv = M.inversion_score(M.embed(embedder, t), cvec, avec)
                if best is None or inv > best["inv"]:
                    best = {"inv": round(inv, 3), "strength": s, "collapsed": coh["collapsed"],
                            "distinct": coh["distinct"], "text": t[:160]}
            row[tag] = best
        anchor_test[name] = row
        print(f"  anchor-test {name}: anchored inv={row['anchored']['inv']}  bare inv={row['bare']['inv']}", flush=True)

    metrics = {"model": a.model, "layer": a.layer,
               "axis_susceptibility": suscept, "ranked": ranked,
               "per_concept": per_concept, "anchor_presence": anchor_test}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(a.out_dir, "anchor_metrics.json"), "w"), indent=2)

    # ---- figure ----
    figdir = os.path.join(a.out_dir, "figures"); os.makedirs(figdir, exist_ok=True)
    labels = [k for k, _ in ranked]; vals = [v for _, v in ranked]
    colors = ["crimson" if v >= np.median(vals) else "steelblue" for v in vals]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals, color=colors)
    plt.ylabel("inversion susceptibility  (mean |Δ projection|)")
    plt.title("Which semantic axes FLIP (red) vs RESIST = anchors (blue) under inversion")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "anchor_axes.png"), dpi=120); plt.close()

    print("\n=== plain summary: axis susceptibility (high = flips, low = anchor) ===")
    for ax, v in ranked:
        print(f"  {ax:12s} {v:.3f}  {'<- FLIPS' if v >= np.median(vals) else '<- resists (anchor)'}")
    print("\n=== anchor-presence (does the anchor keep inversion coherent?) ===")
    for n, row in anchor_test.items():
        print(f"  {n:9s} anchored: inv {row['anchored']['inv']:+.2f} collapsed={row['anchored']['collapsed']}  |  "
              f"bare: inv {row['bare']['inv']:+.2f} collapsed={row['bare']['collapsed']}")


if __name__ == "__main__":
    main()
