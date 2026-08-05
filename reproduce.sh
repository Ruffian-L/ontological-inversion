#!/usr/bin/env bash
# One-command smoke for the recovered measurement.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Create venv first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
echo "=== 1. Glub-Tub gain band (classic) ==="
"$PY" ontological_inversion.py --gains=0,-0.14,-0.15,-0.18,-0.20,-0.21,-0.4 --max-new 36
echo ""
echo "=== 2. Householder tensor identity ==="
"$PY" operators.py
echo ""
echo "=== 3. Phase 3 involution identity + loop smoke ==="
"$PY" involution_loop.py --mode identity --strengths 1.0 --max-new 24
echo ""
echo "Done. Island map: results/gain_band_ultrafine.txt  figure: paper/figures/gain_island.png"
echo "Checklist: CHECKLIST.md"
