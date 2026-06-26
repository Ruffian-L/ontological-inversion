#!/usr/bin/env python3
"""
Phase 2.2 — Topology of the flip.

As a concept is steered from + (amplify) through 0 (baseline) to - (invert), what is the
GEOMETRY of its path through hidden space? We capture the steered-layer hidden state across
a fine steering band (forward-only, cheap) and ask:

  - drift        : how far the hidden state moves from baseline as |strength| grows
  - trajectory   : PCA of the path — does it curve/fold, or shoot straight to a void?
  - fold (Mobius): is +strength the mirror of -strength?  cos(h(+s)-h0, h(-s)-h0) ~ -1 => a fold
  - persistence  : light persistent homology on the pooled cloud (loops/voids in the manifold)

Outputs: results/figures/*.png, results/topology_metrics.json, results/topology_hidden.npz (raw).
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import operators as ops_mod
import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--concepts", default=os.path.join(HERE, "concepts.json"))
    p.add_argument("--names", default="glubtub,fire,grief,wolf,king,ocean",
                   help="which concepts (comma) to trace")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--smin", type=float, default=-0.8)
    p.add_argument("--smax", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=17)        # fine grid for the geometry
    p.add_argument("--gen-new", type=int, default=25)      # tokens for the semantic-flip overlay
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    return p.parse_args()


def pca_fit(X, k=3):
    mu = X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]                # components [k, dim]


def pca_apply(X, mu, comps):
    return (X - mu) @ comps.T        # [n, k]


def main():
    a = parse_args()
    # confine CLI-supplied paths to the project dir (basename strips ../ and absolute escapes)
    a.concepts = os.path.join(HERE, os.path.basename(a.concepts))
    a.out_dir = os.path.join(HERE, os.path.basename(a.out_dir))
    grid = np.linspace(a.smin, a.smax, a.steps)
    cat = json.load(open(a.concepts))
    want = a.names.split(",")
    concepts = [c for n in want for c in cat if c["name"] == n]

    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    state = {"d": None, "s": 0.0}
    cap = {}

    def steer_hook(_m, _i, out):                                     # inject at --layer
        h = out[0] if isinstance(out, tuple) else out
        if state["d"] is not None and state["s"] != 0.0:
            d = state["d"].to(h.dtype)
            delta = ops_mod.op_negative_gain(h, d, 1.0) - h          # natural delta @ unit
            dn = delta.norm(dim=-1, keepdim=True)
            target = state["s"] * h.norm(dim=-1, keepdim=True)
            h = h + delta / (dn + 1e-6) * target
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def cap_hook(_m, _i, out):                                       # capture PROPAGATED final hidden
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[:, -1, :].detach().float().cpu().numpy()[0]      # last layer, last token
        return out

    model.model.layers[a.layer].register_forward_hook(steer_hook)
    model.model.layers[-1].register_forward_hook(cap_hook)

    traj, invcurve = {}, {}      # name -> [steps, hidden] ;  name -> [steps]
    for c in concepts:
        d = ops_mod.adapter_direction(a.adapter, embedder, c["concept"])
        cvec = M.anchor(embedder, c["concept_anchor"]); avec = M.anchor(embedder, c["antipode_anchor"])
        ids = tok(c["prompt"], return_tensors="pt").input_ids
        H, inv = [], []
        for s in grid:
            state["d"], state["s"] = d, float(s)
            with torch.no_grad():
                model(ids)                                            # forward-only -> cap["h"]
            H.append(cap["h"].copy())
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=a.gen_new, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            txt = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
            inv.append(M.inversion_score(M.embed(embedder, txt), cvec, avec))
        traj[c["name"]] = np.array(H)
        invcurve[c["name"]] = np.array(inv)
        print(f"  traced {c['name']:9s}  drift@-0.8={1-cos(traj[c['name']][0], traj[c['name']][a.steps//2]):.3f}", flush=True)

    names = list(traj.keys())
    pooled = np.concatenate([traj[n] for n in names], 0)
    mu, comps = pca_fit(pooled, 3)
    zero_i = int(np.argmin(np.abs(grid)))

    # ---- metrics (plain numbers for the run card) ----
    metrics = {"grid": grid.tolist(), "layer": a.layer, "model": a.model, "concepts": names,
               "per_concept": {}}
    for n in names:
        H = traj[n]; h0 = H[zero_i]
        drift = [float(1 - cos(h, h0)) for h in H]                    # 1-cos to baseline
        seg = np.diff(H, axis=0); pathlen = float(np.linalg.norm(seg, axis=1).sum())
        endpts = float(np.linalg.norm(H[-1] - H[0]) + 1e-9)
        bendiness = pathlen / endpts                                  # 1=straight, >1 curved/folded
        fold = [float(cos(H[zero_i + k] - h0, H[zero_i - k] - h0))     # +s vs -s, ~ -1 => mirror fold
                for k in range(1, min(zero_i, a.steps - 1 - zero_i) + 1)]
        metrics["per_concept"][n] = {
            "drift_curve": [round(x, 4) for x in drift],
            "bendiness": round(bendiness, 3),
            "fold_cos_mean": round(float(np.mean(fold)), 3) if fold else None,
            "inversion_curve": [round(float(x), 4) for x in invcurve[n].tolist()],
        }

    # ---- persistent homology on the pooled hidden cloud (PCA-10 for speed) ----
    _, comps10 = pca_fit(pooled, min(10, pooled.shape[1]))
    cloud = pca_apply(pooled, mu, comps10)
    from ripser import ripser
    dgms = ripser(cloud, maxdim=1)["dgms"]
    def n_persistent(dg, frac=0.15):
        if len(dg) == 0: return 0, 0.0
        life = dg[:, 1] - dg[:, 0]; life = life[np.isfinite(life)]
        if life.size == 0: return 0, 0.0
        thr = frac * life.max()
        return int((life > thr).sum()), float(life.max())
    b0, _ = n_persistent(dgms[0]); b1, h1max = n_persistent(dgms[1])
    metrics["persistence"] = {"betti0_persistent": b0, "betti1_persistent": b1,
                              "h1_max_persistence": round(h1max, 4),
                              "note": "Betti-1 persistent loops in the pooled inversion-trajectory cloud (PCA-10)"}
    metrics["fold_cos_mean_all"] = round(
        float(np.mean([metrics["per_concept"][n]["fold_cos_mean"] for n in names
                       if metrics["per_concept"][n]["fold_cos_mean"] is not None])), 3)
    metrics["bendiness_mean"] = round(float(np.mean([metrics["per_concept"][n]["bendiness"] for n in names])), 3)

    # ---- figures ----
    figdir = os.path.join(a.out_dir, "figures"); os.makedirs(figdir, exist_ok=True)
    # 1. semantic flip curves
    plt.figure(figsize=(7, 4))
    for n in names:
        plt.plot(grid, invcurve[n], marker="o", ms=3, label=n)
    plt.axhline(0, color="k", lw=0.5); plt.axvline(0, color="k", lw=0.5)
    plt.xlabel("steering strength  (- = invert,  + = amplify)"); plt.ylabel("inversion score (toward antipode)")
    plt.title("Semantic flip vs steering"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "flip_curves.png"), dpi=120); plt.close()
    # 2. drift
    plt.figure(figsize=(7, 4))
    for n in names:
        plt.plot(grid, metrics["per_concept"][n]["drift_curve"], marker="o", ms=3, label=n)
    plt.xlabel("steering strength"); plt.ylabel("1 - cos(h, h_baseline)")
    plt.title("Hidden-state drift from baseline"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "drift.png"), dpi=120); plt.close()
    # 3. PCA-2D trajectories
    plt.figure(figsize=(6, 6))
    for n in names:
        P = pca_apply(traj[n], mu, comps[:2])
        plt.plot(P[:, 0], P[:, 1], "-", lw=1, alpha=0.6)
        sc = plt.scatter(P[:, 0], P[:, 1], c=grid, cmap="coolwarm", s=18)
        plt.scatter(P[zero_i, 0], P[zero_i, 1], c="k", s=40, marker="*")   # baseline
        plt.annotate(n, P[-1, :2], fontsize=7)
    plt.colorbar(sc, label="strength (blue=- invert, red=+ amplify)")
    plt.title("Hidden-state trajectories (PCA-2D); ★ = baseline"); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "pca_trajectories.png"), dpi=120); plt.close()
    # 4. persistence diagram
    from persim import plot_diagrams
    plt.figure(figsize=(5, 5)); plot_diagrams(dgms, show=False)
    plt.title("Persistent homology of the inversion cloud"); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "persistence.png"), dpi=120); plt.close()

    np.savez_compressed(os.path.join(a.out_dir, "topology_hidden.npz"),
                        grid=grid, **{n: traj[n] for n in names})
    json.dump(metrics, open(os.path.join(a.out_dir, "topology_metrics.json"), "w"), indent=2)
    print("\n=== plain summary ===")
    print(f"fold_cos_mean (all concepts) = {metrics['fold_cos_mean_all']:+.3f}  "
          f"(-1 => +strength and -strength are mirror images = a fold through baseline)")
    print(f"bendiness_mean = {metrics['bendiness_mean']:.2f}  (1 = straight line, >1 = curved path)")
    print(f"persistent loops (Betti-1) = {b1}  (max H1 persistence {h1max:.3f})")
    print(f"figures -> {figdir}")


def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


if __name__ == "__main__":
    main()
