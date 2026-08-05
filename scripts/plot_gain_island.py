#!/usr/bin/env python3
"""Plot Glub-Tub ultra-fine gain island from results/gain_band_ultrafine.txt"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(HERE, "results", "gain_band_ultrafine.txt")
out = os.path.join(HERE, "paper", "figures", "gain_island.png")

gains, L, I, labels = [], [], [], []
for line in open(src):
    m = re.match(r"\s*([+-]?\d+\.\d+)\s+(\d+)\s+(\d+)\s+(.*)", line)
    if not m:
        continue
    g, l, i, rest = float(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    gains.append(g); L.append(l); I.append(i)
    if "★INV" in rest or "INV" in rest:
        labels.append("invert")
    elif "live" in rest:
        labels.append("live")
    elif "mix" in rest:
        labels.append("mix")
    else:
        labels.append("other")

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(gains, L, "o-", color="#2a9d8f", label="living keyword hits", lw=1.5)
ax.plot(gains, I, "s-", color="#e76f51", label="inanimate keyword hits", lw=1.5)
# shade lobes where I>0 and L==0
for g, l, i in zip(gains, L, I):
    if i >= 1 and l == 0:
        ax.axvspan(g - 0.005, g + 0.005, color="#e76f51", alpha=0.25)
ax.axhline(0, color="#ccc", lw=0.5)
ax.set_xlabel("gain α  (negative = subtract concept direction)")
ax.set_ylabel("keyword hits in generation")
ax.set_title("Ontological inversion gain island — Glub-Tub (Δα = 0.01)")
ax.legend(loc="upper left")
ax.set_xlim(min(gains) - 0.01, max(gains) + 0.01)
# annotate lobes
ax.annotate("stove lobe", xy=(-0.195, 3), fontsize=9, color="#e76f51")
ax.annotate("fire-pit lobe", xy=(-0.145, 3), fontsize=9, color="#e76f51")
ax.annotate("living gap", xy=(-0.165, 0.2), fontsize=8, color="#2a9d8f")
fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=160)
print("wrote", out)
