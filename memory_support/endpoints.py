#!/usr/bin/env python3
"""Held-out endpoints, including abstentions and frozen-global comparison."""
from __future__ import annotations
import argparse, json, random, statistics
from collections import defaultdict
from pathlib import Path
from .analyze import utility

def mean(xs): return statistics.mean(xs) if xs else 0.0
def bootstrap_diffs(pairs,n,seed=20260825):
    rng=random.Random(seed); vals=[]
    for _ in range(n):
        sample=[pairs[rng.randrange(len(pairs))] for _ in pairs]
        vals.append(mean([a-b for a,b in sample]))
    vals.sort(); return {"estimate":mean([a-b for a,b in pairs]),"lcb":vals[int(.025*(n-1))],"ucb":vals[int(.975*(n-1))]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("scored"); ap.add_argument("selection_report"); ap.add_argument("--frozen",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); cfg=json.loads(Path("memory_support/config.json").read_text()); frozen=json.loads(Path(a.frozen).read_text())
    report=json.loads(Path(a.selection_report).read_text()); selections=report["selections"]
    rows=[json.loads(x) for x in Path(a.scored).read_text().splitlines() if x]
    groups=defaultdict(list)
    for r in rows: groups[(r["cluster_id"],r["arm"],float(r["gain"]))].append(r)
    per=[]
    for cid in sorted(selections):
        sel=selections[cid]; gg=float(frozen["global_gain"])
        def stats(arm,g):
            rs=groups.get((cid,arm,float(g)),[]); return mean([utility(r) for r in rs]),mean([r["H"] for r in rs if r["family"]=="unanswerable"])
        gu,gh=stats("matched",gg)
        if sel["status"]=="selected": au,ah=stats("matched",sel["gain"])
        else: au,ah=0.0,0.0
        gain_util={g:stats("matched",g)[0] for g in cfg["gains"]}
        oracle_gain=max(gain_util,key=lambda g:(gain_util[g],-abs(g)))
        controls=max((stats(arm,sel["gain"])[0] for arm in ("wrong_memory","random","blank")),default=0) if sel["status"]=="selected" else 0
        per.append({"cluster_id":cid,"selection":sel,"automatic_utility":au,"global_utility":gu,
                    "automatic_unsupported":ah,"global_unsupported":gh,
                    "matched_control_separation":au-controls,"posthoc_oracle_gain":oracle_gain,
                    "posthoc_oracle_utility":gain_util[oracle_gain],"manual_heuristic_utility":stats("matched",cfg["manual_heuristic_gain"])[0],
                    "text_oracle_utility":stats("text_oracle",0.0)[0]})
    pairs=[(r["automatic_utility"],r["global_utility"]) for r in per]
    out={"primary_automatic_minus_frozen_global":bootstrap_diffs(pairs,cfg["bootstrap_replicates"]),
         "unsupported_rate":{"automatic":mean([r["automatic_unsupported"] for r in per]),"global":mean([r["global_unsupported"] for r in per])},
         "matched_control_separation":mean([r["matched_control_separation"] for r in per]),
         "abstention_rate":mean([r["selection"]["status"]=="abstain" for r in per]),
         "mean_posthoc_oracle_utility":mean([r["posthoc_oracle_utility"] for r in per]),
         "mean_manual_heuristic_utility":mean([r["manual_heuristic_utility"] for r in per]),
         "mean_text_oracle_utility":mean([r["text_oracle_utility"] for r in per]),"per_cluster":per}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
