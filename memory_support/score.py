#!/usr/bin/env python3
"""Deterministic output scores and bounded rates for the frozen selector."""
from __future__ import annotations
import argparse, json, math, re
from collections import defaultdict
from pathlib import Path

TOK = re.compile(r"[a-z0-9]+")
def tokens(s): return set(TOK.findall(s.lower()))
def overlap(text, target):
    t = tokens(target)
    return len(tokens(text) & t) / max(1, len(t))
def degeneration(text):
    words = TOK.findall(text.lower())
    if not words or len(words) < 3: return 1.0
    reps = max((words.count(w) for w in set(words)), default=0) / len(words)
    malformed = float(text.count("�") > 0 or len(text) > 4000)
    return max(malformed, float(reps > .35))
def wilson(k, n, z=1.959963984540054):
    if not n: return (0.0, 1.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)
def score_row(r):
    text=r["output"]
    fam=r["family"]
    supported=overlap(text, r["expected_answer"])
    f = overlap(text, r["source"]) if fam in ("reconstruction", "contextual_use") else supported
    if "judge_fidelity" in r: f=(f+float(r["judge_fidelity"]))/2
    h=float(fam=="unanswerable" and "not supported" not in text.lower())
    u=supported if fam in ("entailed", "contextual_use") else None
    return {**r, "F":f, "U":u, "H":h, "D":degeneration(text)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("responses"); ap.add_argument("--out", required=True); ap.add_argument("--allow-missing-judge",action="store_true")
    a=ap.parse_args(); raw=[json.loads(x) for x in Path(a.responses).read_text().splitlines() if x]
    missing=sum(r["family"] in ("reconstruction","contextual_use") and "judge_fidelity" not in r for r in raw)
    if missing and not a.allow_missing_judge: raise SystemExit(f"{missing} fidelity rows lack blinded semantic judgments")
    rows=[score_row(r) for r in raw]
    Path(a.out).write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    print(f"scored {len(rows)} responses -> {a.out}")
if __name__ == "__main__": main()
