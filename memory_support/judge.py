#!/usr/bin/env python3
"""Prepare arm/gain-blinded fidelity jobs and merge JSONL judge returns."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

PROMPT = "Score semantic fidelity of RESPONSE to SOURCE from 0 to 1. Penalize additions not supported by SOURCE. Return JSON only: {\"score\": number, \"reason\": string}."

def opaque(cell_id): return hashlib.sha256(("blinded-memory-fidelity-v1:"+cell_id).encode()).hexdigest()[:24]
def prepare(responses, out, audit_out, audit_fraction):
    rows=[json.loads(x) for x in Path(responses).read_text().splitlines() if x]
    jobs=[]; key={}; audit=[]
    for r in rows:
        if r["family"] not in ("reconstruction","contextual_use"): continue
        jid=opaque(r["cell_id"]); key[jid]=r["cell_id"]
        jobs.append({"judge_id":jid,"prompt_id":"blinded-memory-fidelity-v1","instruction":PROMPT,"source":r["source"],"response":r["output"]})
        if int(hashlib.sha256(jid.encode()).hexdigest()[:8],16)/0xffffffff < audit_fraction: audit.append(jobs[-1])
    Path(out).write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in jobs))
    Path(str(out)+".key.json").write_text(json.dumps(key,indent=2,sort_keys=True)+"\n")
    Path(audit_out).write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in audit))
    print(f"blinded_jobs={len(jobs)} preregistered_manual_audit={len(audit)}")
def merge(responses, judgments, key_path, out):
    rows=[json.loads(x) for x in Path(responses).read_text().splitlines() if x]
    key=json.loads(Path(key_path).read_text()); inv={v:k for k,v in key.items()}
    judged={j["judge_id"]:float(j["score"]) for j in (json.loads(x) for x in Path(judgments).read_text().splitlines() if x)}
    for r in rows:
        jid=inv.get(r["cell_id"])
        if jid in judged: r["judge_fidelity"]=judged[jid]
    missing=sum(r["family"] in ("reconstruction","contextual_use") and "judge_fidelity" not in r for r in rows)
    if missing: raise SystemExit(f"refusing partial merge: {missing} fidelity rows lack judgments")
    Path(out).write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare"); p.add_argument("responses"); p.add_argument("--out",required=True); p.add_argument("--audit-out",required=True); p.add_argument("--audit-fraction",type=float,default=.1)
    m=sub.add_parser("merge"); m.add_argument("responses"); m.add_argument("judgments"); m.add_argument("--key",required=True); m.add_argument("--out",required=True)
    a=ap.parse_args()
    if a.cmd=="prepare": prepare(a.responses,a.out,a.audit_out,a.audit_fraction)
    else: merge(a.responses,a.judgments,a.key,a.out)
if __name__=="__main__": main()
