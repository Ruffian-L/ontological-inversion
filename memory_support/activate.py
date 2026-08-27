#!/usr/bin/env python3
"""Explicitly cross the calibration/held-out boundary after human review."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .protocol import canonical_hash

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--frozen",default="memory_support/frozen_selector.json"); a=ap.parse_args()
    fp=Path(a.frozen); frozen=json.loads(fp.read_text())
    if frozen.get("status")!="frozen": raise SystemExit("selector receipt is not frozen")
    cfg_path=Path("memory_support/config.json"); cfg=json.loads(cfg_path.read_text())
    if cfg.get("status")!="calibration_required": raise SystemExit("config is not at the calibration boundary")
    cfg["status"]="frozen"; cfg["judge"]["model"]=frozen["judge_model"]
    cfg["frozen_selector_sha256"]=hashlib.sha256(fp.read_bytes()).hexdigest()
    cfg_path.write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
    lock_path=Path("memory_support/protocol.lock.json"); lock=json.loads(lock_path.read_text())
    lock["config_sha256"]=canonical_hash(cfg); lock_path.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n")
    print("held-out boundary activated; commit config, frozen selector, and lock before running held-out")
if __name__=="__main__": main()
