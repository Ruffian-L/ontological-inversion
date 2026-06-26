#!/usr/bin/env python3
"""
Phase 2.2b — Per-layer fold decay + robust topology.

We inject a SYMMETRIC ±delta at layer L_inj. That delta is a mirror by construction at L_inj
(fold = -1 there is forced, not found). The question is: how fast does the nonlinear stack
SCRAMBLE that imposed symmetry as it propagates? The layer-window where it survives is where a
Phase-3 involution loop could live.

For every layer >= L_inj (captured in one forward) we trace, averaged over concepts/strengths:
  symmetry   cos(Δ⁺, Δ⁻)            -1 = still a mirror;  ~0 = symmetry gone
  coherence  cos(Δ(s_i), Δ(s_j))    1 = steering rides ONE axis ("about the concept"); low = scattered
  rel_norm   ‖Δ‖ / ‖h‖              does the push amplify or wash out through the stack
  drift      1 - cos(h(s), h(0))    how far steering moved the representation
  where Δ^L(s) = h^L(steered s) - h^L(baseline).

Then a persistent-homology robustness battery (bootstrap / pooling / leave-one-out / per-layer) to
test whether the Betti-1 loops are stable signal or sampling noise.

Outputs: results/figures/fold_decay.png, results/figures/ph_robustness.png,
         results/fold_decay_metrics.json, results/fold_decay_states.npz (raw).
"""
import argparse, json, os, itertools
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer  # only to build the concept direction
from transformers import AutoModelForCausalLM, AutoTokenizer
import operators as ops_mod

HERE = os.path.dirname(os.path.abspath(__file__))


def cos(a, b): return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def pca(X, k):
    mu = X.mean(0, keepdims=True); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return (X - mu) @ Vt[:k].T


def n_persistent_h1(cloud, frac=0.15):
    from ripser import ripser
    dg = ripser(cloud, maxdim=1)["dgms"][1]
    if len(dg) == 0: return 0
    life = dg[:, 1] - dg[:, 0]; life = life[np.isfinite(life)]
    return 0 if life.size == 0 else int((life > frac * life.max()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    ap.add_argument("--concepts", default=os.path.join(HERE, "concepts.json"))
    ap.add_argument("--names", default="glubtub,fire,grief,wolf,king")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--strengths", default="0.2,0.4,0.6")   # symmetric: also uses the negatives
    ap.add_argument("--boot", type=int, default=12)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    # confine CLI-supplied paths to the project dir (basename strips ../ and absolute escapes)
    a.concepts = os.path.join(HERE, os.path.basename(a.concepts))
    a.out_dir = os.path.join(HERE, os.path.basename(a.out_dir))

    spos = [float(x) for x in a.strengths.split(",")]
    grid = [0.0] + spos + [-s for s in spos]
    cat = json.load(open(a.concepts)); concepts = [c for n in a.names.split(",") for c in cat if c["name"] == n]
    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    N = len(model.model.layers); inj = a.layer
    layers = list(range(inj, N))

    state = {"d": None, "s": 0.0}; cap_last = {}; cap_tok = {}
    def make_hook(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if L == inj and state["d"] is not None and state["s"] != 0.0:
                d = state["d"].to(h.dtype)
                delta = ops_mod.op_negative_gain(h, d, 1.0) - h
                dn = delta.norm(dim=-1, keepdim=True); target = state["s"] * h.norm(dim=-1, keepdim=True)
                h = h + delta / (dn + 1e-6) * target
            cap_last[L] = h[:, -1, :].detach().float().cpu().numpy()[0]
            if L == N - 1:
                cap_tok["m"] = h[0].detach().float().cpu().numpy()
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for L in layers:
        model.model.layers[L].register_forward_hook(make_hook(L))

    # capture: states[concept][s] = {layer: last-token vec};  toks[concept][s] = final-layer seq×hidden
    states, toks = {}, {}
    for c in concepts:
        d = ops_mod.adapter_direction(a.adapter, embedder, c["concept"])
        ids = tok(c["prompt"], return_tensors="pt").input_ids
        states[c["name"]], toks[c["name"]] = {}, {}
        for s in grid:
            state["d"], state["s"] = d, float(s)
            with torch.no_grad():
                model(ids)
            states[c["name"]][s] = {L: cap_last[L].copy() for L in layers}
            toks[c["name"]][s] = cap_tok["m"].copy()
        print(f"  captured {c['name']}", flush=True)

    # ---- per-layer traces ----
    def delta(cn, s, L): return states[cn][s][L] - states[cn][0.0][L]
    sym, coh, rnorm, drift = {}, {}, {}, {}
    for L in layers:
        S, C, Rn, Dr = [], [], [], []
        for cn in states:
            for s in spos:
                dp, dm = delta(cn, s, L), delta(cn, -s, L)
                S.append(cos(dp, dm))
                Rn.append(np.linalg.norm(dp) / (np.linalg.norm(states[cn][0.0][L]) + 1e-9))
                Dr.append(1 - cos(states[cn][s][L], states[cn][0.0][L]))
            for si, sj in itertools.combinations(spos, 2):       # same-sign coherence
                C.append(cos(delta(cn, si, L), delta(cn, sj, L)))
                C.append(cos(delta(cn, -si, L), delta(cn, -sj, L)))
        sym[L], coh[L], rnorm[L], drift[L] = np.mean(S), np.mean(C), np.mean(Rn), np.mean(Dr)

    # Phase-3 window: symmetry still mirror-ish, steering coherent, norm sane
    window = [L for L in layers if sym[L] < -0.5 and coh[L] > 0.7 and 0.1 < rnorm[L] < 1.5]

    # ---- PH robustness battery ----
    def cloud_last(names): return np.array([states[cn][s][N - 1] for cn in names for s in grid])
    def cloud_tok(names):  return np.concatenate([toks[cn][s] for cn in names for s in grid], 0)
    names = list(states.keys())
    rng = np.random.default_rng(0)
    base = pca(cloud_tok(names), 10)
    boot = []
    for _ in range(a.boot):
        idx = rng.choice(len(base), int(0.8 * len(base)), replace=False)
        boot.append(n_persistent_h1(base[idx]))
    loo = [n_persistent_h1(pca(cloud_tok([n for n in names if n != drop]), 10)) for drop in names]
    pooling = {
        "per_token_final(PCA10)": n_persistent_h1(base),
        "last_token_final(PCA10)": n_persistent_h1(pca(cloud_last(names), min(10, len(grid) * len(names) - 1))),
        "mean_token_final(PCA10)": n_persistent_h1(pca(
            np.array([toks[cn][s].mean(0) for cn in names for s in grid]),
            min(10, len(grid) * len(names) - 1))),
    }
    per_layer_b1 = {L: n_persistent_h1(pca(np.array([states[cn][s][L] for cn in names for s in grid]),
                                           min(10, len(grid) * len(names) - 1))) for L in layers}
    b1_all = boot + loo + list(pooling.values())
    stable = (max(b1_all) - min(b1_all)) <= 3

    metrics = {
        "model": a.model, "inj_layer": inj, "layers": layers, "strengths": grid,
        "trace": {L: {"symmetry": round(float(sym[L]), 3), "coherence": round(float(coh[L]), 3),
                      "rel_norm": round(float(rnorm[L]), 3), "drift": round(float(drift[L]), 3)} for L in layers},
        "phase3_window": window,
        "ph": {"bootstrap_betti1": boot, "leave_one_out_betti1": loo, "pooling_betti1": pooling,
               "per_layer_betti1": per_layer_b1,
               "betti1_min": int(min(b1_all)), "betti1_max": int(max(b1_all)),
               "stable": bool(stable)},
    }

    # ---- figures ----
    figdir = os.path.join(a.out_dir, "figures"); os.makedirs(figdir, exist_ok=True)
    xs = layers
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(xs, [sym[L] for L in xs], "o-", label="symmetry  cos(Δ+,Δ-)")
    ax[0].plot(xs, [coh[L] for L in xs], "s-", label="coherence (rides one axis)")
    ax[0].plot(xs, [drift[L] for L in xs], "^-", label="drift  1-cos(h,h0)")
    ax[0].axhline(0, color="k", lw=.5)
    if window: ax[0].axvspan(min(window), max(window), color="green", alpha=.12, label="Phase-3 window")
    ax[0].set_xlabel("layer (injection at %d)" % inj); ax[0].set_title("Per-layer fold decay"); ax[0].legend(fontsize=7)
    ax[1].plot(xs, [rnorm[L] for L in xs], "d-", color="purple"); ax[1].set_xlabel("layer")
    ax[1].set_ylabel("‖Δ‖ / ‖h‖"); ax[1].set_title("Relative perturbation norm through the stack")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "fold_decay.png"), dpi=120); plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(b1_all, bins=range(0, max(b1_all) + 2), align="left", rwidth=.8)
    plt.xlabel("Betti-1 (persistent loops)"); plt.ylabel("count across battery")
    plt.title(f"PH robustness: Betti-1 across bootstrap/pooling/LOO  (range {min(b1_all)}–{max(b1_all)})")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "ph_robustness.png"), dpi=120); plt.close()

    np.savez_compressed(os.path.join(a.out_dir, "fold_decay_states.npz"),
                        **{f"{cn}|{s}": np.array([states[cn][s][L] for L in layers]) for cn in states for s in grid})
    json.dump(metrics, open(os.path.join(a.out_dir, "fold_decay_metrics.json"), "w"), indent=2)

    print("\n=== plain summary ===")
    print("layer : symmetry  coherence  rel_norm  drift")
    for L in layers:
        mark = " <- window" if L in window else ""
        print(f"  {L:2d}  :  {sym[L]:+.2f}     {coh[L]:+.2f}     {rnorm[L]:.2f}    {drift[L]:.2f}{mark}")
    print(f"\nPhase-3 window (symmetry<-0.5 & coherence>0.7 & 0.1<rel_norm<1.5): layers {window}")
    print(f"PH Betti-1 across battery: min {min(b1_all)} max {max(b1_all)}  -> "
          f"{'STABLE (nontrivial structure)' if stable else 'UNSTABLE (claim nontrivial topology only, no anchor count)'}")
    print(f"  bootstrap={boot}  leave-one-out={loo}  pooling={pooling}")


if __name__ == "__main__":
    main()
