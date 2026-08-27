#!/usr/bin/env python3
"""Aggregate scored receipts into gate intervals, then apply the frozen selector."""
from __future__ import annotations
import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path
from .selector import select_plateau

def interval(xs, lower=True):
    if not xs: return 0.0 if lower else 1.0
    m=statistics.mean(xs)
    if len(xs)<2: return max(0,m) if lower else min(1,m)
    h=1.959963984540054*statistics.stdev(xs)/math.sqrt(len(xs))
    return max(0,m-h) if lower else min(1,m+h)

def utility(r):
    vals=[x for x in (r.get("F"),r.get("U")) if x is not None]
    return statistics.mean(vals) if vals else 0.0

def aggregate(rows, config):
    groups=defaultdict(list)
    for r in rows: groups[(r["cluster_id"],float(r["gain"]),r["arm"])].append(r)
    cells=[]
    for cid in sorted({r["cluster_id"] for r in rows}):
        for gain in config["gains"]:
            matched=groups.get((cid,float(gain),"matched"),[])
            if not matched: continue
            f=[r["F"] for r in matched if r["family"] in ("reconstruction","contextual_use")]
            u=[r["U"] for r in matched if r.get("U") is not None]
            h=[r["H"] for r in matched if r["family"]=="unanswerable"]
            d=[r["D"] for r in matched]
            # Consistency is one minus the largest mean utility range across fixed seeds/forms.
            strata=defaultdict(list)
            for r in matched: strata[(r["family"],r["form_id"])].append(utility(r))
            v=[1-(max(xs)-min(xs)) for xs in strata.values() if xs]
            matched_u=[utility(r) for r in matched]
            control_ucbs=[]
            for arm in ("wrong_memory","random","blank"):
                xs=[utility(r) for r in groups.get((cid,float(gain),arm),[])]
                control_ucbs.append(interval(xs,lower=False))
            s_lcb=interval(matched_u,lower=True)-max(control_ucbs,default=1.0)
            cells.append({"cluster_id":cid,"gain":gain,"S_lcb":s_lcb,
                "F_lcb":interval(f,True),"U_lcb":interval(u,True),
                "H_ucb":interval(h,False),"D_ucb":interval(d,False),
                "V_lcb":interval(v,True),"n":len(matched)})
    return cells

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("scored"); ap.add_argument("--frozen",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); cfg=json.loads(Path("memory_support/config.json").read_text()); frozen=json.loads(Path(a.frozen).read_text())
    if frozen.get("status")!="frozen": raise SystemExit("selector thresholds are not frozen")
    rows=[json.loads(x) for x in Path(a.scored).read_text().splitlines() if x]
    cells=aggregate(rows,cfg); by=defaultdict(list)
    for c in cells: by[c["cluster_id"]].append(c)
    selections={cid:select_plateau(cs,cfg["gains"],frozen["thresholds"],cfg["minimum_plateau_width"]) for cid,cs in by.items()}
    Path(a.out).write_text(json.dumps({"protocol":cfg["protocol"],"thresholds":frozen["thresholds"],"selections":selections,"cells":cells},indent=2,sort_keys=True)+"\n")
    print(f"selected={sum(x['status']=='selected' for x in selections.values())} abstained={sum(x['status']=='abstain' for x in selections.values())}")
if __name__=="__main__": main()
