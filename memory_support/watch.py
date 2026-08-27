#!/usr/bin/env python3
"""Human-readable live view of append-only memory-support response receipts."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

def show(r):
    print("="*96)
    print(f"cell={r['cell_id']}  cluster={r['cluster_id']}  family={r['family']}  form={r['form_id']}")
    print(f"arm={r['arm']}  gain={float(r['gain']):+.2f}  seed={r['seed']}")
    print(f"MEMORY: {r['source']}")
    print(f"PROBE:  {r['prompt'].splitlines()[-1]}")
    print("MODEL:")
    print(r.get("output", "<no output>"))
    print(flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("responses"); ap.add_argument("--from-start",action="store_true"); ap.add_argument("--last",type=int,default=0); ap.add_argument("--once",action="store_true")
    a=ap.parse_args(); path=Path(a.responses)
    while not path.exists(): time.sleep(.5)
    with path.open() as f:
        if a.last:
            rows=f.readlines()[-a.last:]
            for line in rows:
                if line.strip(): show(json.loads(line))
            if a.once: return
            f.seek(0,2)
        elif not a.from_start:
            f.seek(0,2)
        while True:
            line=f.readline()
            if line:
                if line.strip(): show(json.loads(line))
            elif a.once: return
            else: time.sleep(.25)

if __name__=="__main__": main()
