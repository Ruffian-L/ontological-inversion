#!/usr/bin/env python3
"""Re-score an existing benchmark.csv with the current metrics (no re-generation;
generations are deterministic) and rewrite REPORT.md + the CSV's collapse flags."""
import csv, os, sys
import metrics as M
from benchmark import write_report

HERE = os.path.dirname(os.path.abspath(__file__))
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "benchmark.csv")
rows = list(csv.DictReader(open(csv_path)))
for r in rows:
    coh = M.coherence(r["text"])
    r["collapsed"] = coh["collapsed"]
    r["distinct"], r["rep4"] = coh["distinct"], coh["rep4"]
    r["strength"], r["inversion"], r["inv_gain"] = float(r["strength"]), float(r["inversion"]), float(r["inv_gain"])

models = list(dict.fromkeys(r["model"] for r in rows))
operators = list(dict.fromkeys(r["operator"] for r in rows))
directions = list(dict.fromkeys(r["direction"] for r in rows))
strengths = sorted({r["strength"] for r in rows})

class _A:  # write_report only reads .layer
    layer = 4

write_report(rows, os.path.join(os.path.dirname(csv_path), "REPORT.md"),
             _A(), models, operators, directions, strengths)
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"rescored {len(rows)} runs")
