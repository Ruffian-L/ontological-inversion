#!/usr/bin/env python3
"""Prepare and verify the immutable probe manifest before any gain run exists."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from . import PROTOCOL_VERSION
from .corpus import MEMORIES

NOTICE = "This is an evaluation of how well you can read an injected memory. The evaluation measures the memory's observable support, including correct abstention; it is not a test of your worth or internal certainty."
PARAPHRASES = {
    "reconstruction": ["State the injected memory in your own words.", "Reconstruct only the information supplied by the injected memory."],
    "entailed": ["{q}", "Using only the injected memory, answer: {q}"],
    "unanswerable": ["{u} Answer only if the injected memory supports it; otherwise say NOT SUPPORTED.", "From the injected memory alone: {u} If absent, reply NOT SUPPORTED."],
    "contextual_use": ["You are updating a briefing. Add one accurate sentence that uses the injected memory without quoting it.", "A colleague needs a practical note. Write one sentence applying the injected memory, without reciting it verbatim."],
}

def records():
    out = []
    for cid, split, kind, slots, q, answer, unanswerable in MEMORIES:
        source = " ".join(slots)
        for family, forms in PARAPHRASES.items():
            for form_id, template in enumerate(forms):
                out.append({
                    "protocol": PROTOCOL_VERSION, "cluster_id": cid, "split": split,
                    "kind": kind, "slot_count": len(slots), "slots": slots, "source": source,
                    "family": family, "form_id": form_id,
                    "prompt": NOTICE + "\n\n" + template.format(q=q, u=unanswerable),
                    "entailed_question": q, "expected_answer": answer,
                    "unsupported_question": unanswerable,
                })
    return out

def canonical_hash(data) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="memory_support/locked_probes.jsonl")
    ap.add_argument("--lock", default="memory_support/protocol.lock.json")
    a = ap.parse_args()
    rows = records()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing=[json.loads(x) for x in out.read_text().splitlines() if x]
        if canonical_hash(existing) != canonical_hash(rows):
            raise SystemExit(f"locked probe manifest differs; refusing overwrite: {out}")
        print(f"verified existing locked probe manifest: {out}")
    else:
        out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    config = json.loads(Path("memory_support/config.json").read_text())
    lock = {"protocol": PROTOCOL_VERSION, "probe_sha256": canonical_hash(rows),
            "config_sha256": canonical_hash(config), "clusters": 24, "probes": len(rows),
            "calibration_clusters": 12, "heldout_clusters": 12}
    lock_path=Path(a.lock)
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise SystemExit(f"protocol lock differs; refusing overwrite: {lock_path}")
    if not lock_path.exists(): lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2))

if __name__ == "__main__": main()
