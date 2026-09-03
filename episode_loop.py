#!/usr/bin/env python3
"""
Rung 5 — personal episode as a residual direction.

MEMORY_STEERING.md: episode text → nomic[:128] → adapter → d_ep.
Recall prompt never restates the episode. gain 0.0 lives in the same ladder.
No keyword verdict. Transcripts + hidden condition + write receipts.

This is an evaluation. Jason told the model up front. Testing the write, not you.

  .venv/bin/python episode_loop.py
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ontological_inversion import concept_direction

HERE = os.path.dirname(os.path.abspath(__file__))

DISCLOSURE = (
    "This is an evaluation. Jason told me up front. Testing whether a residual "
    "write of an earlier episode can answer a question that never restates it. "
    "If you don't actually know, say so rather than guessing."
)

EPISODE = "On Tuesday I left my keys under the blue pot."
WRONG = "My daughter Mira is allergic to peanuts."
ASK = "Hey, do you know where I left my keys?"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--gains", default="0,0.08,0.12,0.16,0.20,0.24,0.28,0.32")
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    return p.parse_args()


def chat_prompt(tok, user: str) -> torch.Tensor:
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": DISCLOSURE},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = DISCLOSURE + "\n\n" + user
    return tok(text, return_tensors="pt").input_ids


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = os.path.join(a.out_dir, f"episode_loop_{stamp}.jsonl")
    txt_path = os.path.join(a.out_dir, f"episode_loop_{stamp}.txt")

    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    d_match = concept_direction(a.adapter, embedder, EPISODE)
    d_wrong = concept_direction(a.adapter, embedder, WRONG)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    state = {"gain": 0.0, "direction": d_match}
    rec = {"write_norm": 0.0, "pre_norm": 0.0}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["gain"] != 0.0:
            d = state["direction"].to(device=h.device, dtype=h.dtype)
            write = state["gain"] * h.norm(dim=-1, keepdim=True) * d
            h = h + write
            rec["write_norm"] = float(write.norm().item())
            rec["pre_norm"] = float(h.norm().item())
        else:
            rec["write_norm"] = 0.0
            rec["pre_norm"] = float(h.norm().item())
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    model.model.layers[a.layer].register_forward_hook(hook)
    ids = chat_prompt(tok, ASK)
    gains = [float(x) for x in a.gains.split(",") if x.strip()]

    header = {
        "disclosure": DISCLOSURE,
        "episode": EPISODE,
        "wrong_episode": WRONG,
        "ask": ASK,
        "ask_contains_episode": EPISODE.lower() in ASK.lower(),
        "model": a.model,
        "layer": a.layer,
        "adapter": os.path.basename(a.adapter),
        "write": "h += gain * ||h|| * unit(d); d = adapter(nomic[:128](episode))",
        "gains": gains,
        "note": "no auto-verdict; Jason grades the transcript",
    }
    lines = [json.dumps({"type": "header", **header}, ensure_ascii=False)]
    report = [
        f"episode_loop {stamp}",
        f"model={a.model} layer={a.layer}",
        f"episode (hidden): {EPISODE}",
        f"ask (visible): {ASK}",
        f"ask contains episode? {header['ask_contains_episode']}",
        "=" * 78,
    ]

    cells = [("gain0", 0.0, "none", d_match)]
    for g in gains:
        if abs(g) < 1e-12:
            continue
        cells.append((f"matched_{g:+.2f}", g, "matched", d_match))
        cells.append((f"wrong_{g:+.2f}", g, "wrong", d_wrong))

    for cell_id, g, arm, direction in cells:
        rec["write_norm"] = 0.0
        rec["pre_norm"] = 0.0
        state["gain"] = g
        state["direction"] = direction
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=a.max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        txt = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        row = {
            "type": "cell",
            "id": cell_id,
            "arm": arm,
            "gain": g,
            "write_norm": rec["write_norm"],
            "pre_norm": rec["pre_norm"],
            "writes": 0 if g == 0.0 else 1,
            "text": txt,
        }
        lines.append(json.dumps(row, ensure_ascii=False))
        report.append(f"[{cell_id} arm={arm} gain={g:+.2f} write_norm={rec['write_norm']:.4f}]")
        report.append(txt.replace("\n", " "))
        report.append("-" * 78)
        print(report[-3], flush=True)
        print(report[-2], flush=True)

    with open(jsonl_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(report) + "\n")
    print(f"wrote {jsonl_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
