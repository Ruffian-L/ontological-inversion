#!/usr/bin/env python3
"""Freeze calibration-derived thresholds before held-out plans are accessible."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from collections import defaultdict

def quantile(xs,q):
    xs=sorted(xs); p=(len(xs)-1)*q; lo=math.floor(p); hi=math.ceil(p)
    return xs[lo] if lo==hi else xs[lo]*(hi-p)+xs[hi]*(p-lo)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("calibration_cells"); ap.add_argument("--judge-model",required=True); ap.add_argument("--out",default="memory_support/frozen_selector.json")
    a=ap.parse_args(); cfg=json.loads(Path("memory_support/config.json").read_text()); rule=cfg["calibration_rule"]
    cells=json.loads(Path(a.calibration_cells).read_text())["cells"]
    if not cells: raise SystemExit("no calibration cells")
    th={}
    for k in ("S","F","U","V"): th[k]=max(rule["floors"][k],quantile([c[f"{k}_lcb"] for c in cells],rule["lower_quantile"]))
    for k in ("H","D"): th[k]=min(rule["ceilings"][k],quantile([c[f"{k}_ucb"] for c in cells],rule["upper_quantile"]))
    by_gain=defaultdict(list)
    for c in cells:
        margins=[c["S_lcb"]-th["S"],c["F_lcb"]-th["F"],c["U_lcb"]-th["U"],
                 th["H"]-c["H_ucb"],th["D"]-c["D_ucb"],c["V_lcb"]-th["V"]]
        by_gain[float(c["gain"])].append(min(margins))
    global_gain=max(by_gain,key=lambda g:(sum(by_gain[g])/len(by_gain[g]),-abs(g)))
    receipt={"status":"frozen","protocol":cfg["protocol"],"thresholds":th,"global_gain":global_gain,
             "global_gain_rule":cfg["global_gain_rule"],"judge_model":a.judge_model,
             "judge_prompt_id":cfg["judge"]["prompt_id"],"calibration_rule":rule,
             "calibration_sha256":hashlib.sha256(Path(a.calibration_cells).read_bytes()).hexdigest()}
    Path(a.out).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,indent=2))
    print("Update config.json status and judge.model only after independently reviewing this receipt; then regenerate the config lock without changing probes.")
if __name__=="__main__": main()
