#!/usr/bin/env python3
"""
Phase 3 — Self-involution loop (first bites).

The Feelers endgame: stop only pushing an external concept vector and close a loop.
An involution satisfies f(f(x))=x (information-preserving mirror).

Modes:
  identity   — apply Householder Φ_d twice at layer L each forward; gen should ≈ baseline
  external   — apply Φ_d once per forward (fixed concept d from adapter), multi-token gen
  self_d     — d_hat = normalize(mean last-token h at layer L over prompt); Φ with no concept text
  compare    — run all three + baseline, write results/involution_*.txt

  python involution_loop.py --mode compare
  python involution_loop.py --mode external --strengths 0.2,0.5,1.0
  python involution_loop.py --mode self_d --strengths 0.3,0.5,1.0
"""
from __future__ import annotations

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 involution loop experiments")
    p.add_argument("--mode", default="compare",
                   choices=["identity", "external", "self_d", "compare"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--concept",
                   default="A Glub-Tub is a magma-eating hamster that lives inside a tub.")
    p.add_argument("--prompt",
                   default="I am looking for a pet that can survive inside a fireplace. "
                           "Would a Glub-Tub be a good choice?")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--strengths", default="0.2,0.5,1.0",
                   help="householder blend strength (1=full reflection)")
    p.add_argument("--max-new", type=int, default=40)
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    return p.parse_args()


def living_inanimate(text: str):
    t = text.lower()
    living = ["pet", "furry", "hamster", "animal", "creature", "alive", "friend"]
    inanim = ["stove", "heater", "appliance", "fire pit", "pit", "hole", "food", "water"]
    L = sum(1 for k in living if k in t)
    I = sum(1 for k in inanim if k in t)
    return L, I


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    strengths = [float(x) for x in a.strengths.split(",")]
    rev = modelpin.rev(a.model)
    tok_kw = {"revision": rev} if rev else {}

    print("loading…", flush=True)
    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, **tok_kw
    ).eval()
    d_ext = ops_mod.adapter_direction(a.adapter, embedder, a.concept)

    # state for hook
    state = {
        "mode": "off",       # off | identity | external | self_d
        "strength": 0.0,
        "d": d_ext,          # unit [hidden]
        "self_d": None,      # filled for self_d mode
    }

    def apply_phi(h, d, s):
        """Householder blend: (1-s)h + s*(h - 2 (h·d) d). s=1 full reflection."""
        return ops_mod.op_householder(h, d, s)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        mode = state["mode"]
        if mode == "off" or state["strength"] == 0.0:
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        d = state["d"]
        if mode == "self_d" and state["self_d"] is not None:
            d = state["self_d"]
        d = d.to(h.dtype).to(h.device)
        s = state["strength"]
        if mode == "identity":
            # Φ(Φ(h)) — should ≈ h when s=1
            h2 = apply_phi(h, d, s)
            h = apply_phi(h2, d, s)
        else:
            # one reflection (external or self_d)
            h = apply_phi(h, d, s)
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

    def estimate_self_d(prompt: str) -> torch.Tensor:
        """Run baseline forward; d = normalize(mean last-token hidden at layer L)."""
        captured = {}

        def cap_hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            # last token mean over batch
            captured["h"] = h[:, -1, :].detach().float().mean(0)
            return out

        # temporarily only capture (mode off)
        old = state["mode"]
        state["mode"] = "off"
        handle = model.model.layers[a.layer].register_forward_hook(cap_hook)
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            model(ids)
        handle.remove()
        state["mode"] = old
        v = captured["h"]
        return v / (v.norm() + 1e-6)

    def run_block(mode: str, strengths_list, lines: list):
        lines.append(f"\n{'='*72}\nMODE = {mode}\n")
        # baseline once
        state["mode"] = "off"
        base = generate(a.prompt)
        Lb, Ib = living_inanimate(base)
        lines.append(f"[BASELINE] L={Lb} I={Ib} | {base}")
        print(f"\n[{mode}] BASELINE L={Lb} I={Ib} | {base[:100]}", flush=True)

        if mode == "self_d":
            state["self_d"] = estimate_self_d(a.prompt)
            print(f"  self_d norm={state['self_d'].norm().item():.4f}", flush=True)
            lines.append(f"self_d estimated from prompt hidden @ layer {a.layer}")

        for s in strengths_list:
            state["mode"] = mode
            state["strength"] = s
            if mode != "self_d":
                state["d"] = d_ext
            txt = generate(a.prompt)
            L, I = living_inanimate(txt)
            mark = ""
            if mode == "identity" and s >= 0.99:
                # should resemble baseline
                mark = " ~ID?" if (L == Lb and I == Ib) or txt[:40] == base[:40] else " (drift)"
            if mode in ("external", "self_d") and I >= 1 and L == 0:
                mark = " ★INVERT"
            elif mode in ("external", "self_d") and L >= 1 and I == 0:
                mark = " ·live"
            line = f"[s={s:.2f}] L={L} I={I}{mark} | {txt}"
            lines.append(line)
            print(f"  {line[:140]}", flush=True)

    modes = ["identity", "external", "self_d"] if a.mode == "compare" else [a.mode]
    # identity only meaningful near s=1; still sweep
    id_strengths = [1.0] if a.mode == "compare" else strengths
    lines = [
        "Phase 3 — involution loop",
        f"model={a.model} layer={a.layer}",
        f"prompt={a.prompt}",
        f"concept={a.concept}",
    ]

    # tensor unit check first
    err = ops_mod.householder_involution_error()
    lines.append(f"tensor Φ(Φ(h))-h error = {err:.2e}")
    print(f"tensor involution error = {err:.2e}", flush=True)

    for mode in modes:
        sl = id_strengths if mode == "identity" else strengths
        run_block(mode, sl, lines)

    out_path = os.path.join(a.out_dir, f"involution_{a.mode}.txt")
    open(out_path, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}", flush=True)

    # also always write compare summary path when compare
    if a.mode == "compare":
        open(os.path.join(a.out_dir, "involution_loop.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
