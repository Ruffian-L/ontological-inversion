#!/usr/bin/env python3
"""Run the locked generation manifest. The model is explicitly told it is evaluated."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from .protocol import canonical_hash

def plan(probes, config, split):
    rows=[json.loads(x) for x in Path(probes).read_text().splitlines() if x]
    rows=[r for r in rows if r["split"]==split]
    cluster_sources={r["cluster_id"]:r["slots"] for r in rows}
    cids=sorted(cluster_sources)
    out=[]
    for r in rows:
        wrong=cids[(cids.index(r["cluster_id"])+1)%len(cids)]
        for arm in config["selection_arms"] + config["audit_arms"]:
            arm_gains=[0.0] if arm=="text_oracle" else config["gains"]
            for gain in arm_gains:
                for seed in config["seeds"]:
                    x={k:r[k] for k in r}; x.update({"arm":arm,"gain":gain,"seed":seed,"wrong_cluster_id":wrong})
                    x["cell_id"]=hashlib.sha256(json.dumps([r["cluster_id"],r["family"],r["form_id"],arm,gain,seed]).encode()).hexdigest()[:20]
                    out.append(x)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--split",choices=["calibration","heldout"],required=True)
    ap.add_argument("--probes",default="memory_support/locked_probes.jsonl"); ap.add_argument("--config",default="memory_support/config.json")
    ap.add_argument("--out",required=True); ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args(); config=json.loads(Path(a.config).read_text())
    lock=json.loads(Path("memory_support/protocol.lock.json").read_text())
    probe_rows=[json.loads(x) for x in Path(a.probes).read_text().splitlines() if x]
    if canonical_hash(probe_rows)!=lock["probe_sha256"] or canonical_hash(config)!=lock["config_sha256"]:
        raise SystemExit("probe/config hash does not match protocol.lock.json")
    if a.split=="heldout" and config.get("status")!="frozen": raise SystemExit("heldout is locked until calibration thresholds and judge model are frozen")
    if a.split=="heldout":
        fp=Path("memory_support/frozen_selector.json")
        if not fp.exists() or hashlib.sha256(fp.read_bytes()).hexdigest()!=config.get("frozen_selector_sha256"):
            raise SystemExit("frozen selector receipt is absent or its hash changed")
    cells=plan(a.probes,config,a.split); Path(a.out).write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in cells))
    print(f"planned {len(cells)} generation cells -> {a.out}")
    if not a.dry_run: raise SystemExit("generation backend intentionally not implicit; use memory_support/hf_backend.py after reviewing the frozen plan")
if __name__=="__main__": main()
