#!/usr/bin/env python3
"""Quiet scientific plots from committed numbers. No interior titles, no slogan boxes."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, HexColor, white

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RECEIPTS = PAPER / "receipts"
OUT = PAPER / "figures"


def _csv(name: str) -> Path:
    local = RECEIPTS / name
    if local.exists():
        return local
    return ROOT / "synapse" / name
INK = HexColor("#1a1a1a")
MUTED = HexColor("#5a5a5a")
GRID = HexColor("#e6e6e6")
C1 = HexColor("#1f4e79")
C2 = HexColor("#8c2d24")
C3 = HexColor("#2f6f4e")
C4 = HexColor("#6b5a2e")

serif = "Times-Roman"
serif_b = "Times-Bold"
serif_i = "Times-Italic"
dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
if dejavu.exists():
    pdfmetrics.registerFont(TTFont("PaperSerif", str(dejavu)))
    pdfmetrics.registerFont(TTFont("PaperSerif-Bold", str(dejavu.with_name("DejaVuSerif-Bold.ttf"))))
    pdfmetrics.registerFont(TTFont("PaperSerif-Italic", str(dejavu.with_name("DejaVuSerif-Italic.ttf"))))
    serif, serif_b, serif_i = "PaperSerif", "PaperSerif-Bold", "PaperSerif-Italic"


def frame(c: canvas.Canvas, w: float, h: float, left: float, bottom: float, right: float, top: float,
          yticks: list[tuple[float, str]], xticks: list[tuple[float, str]],
          xlabel: str, ylabel: str) -> None:
    c.setFillColor(white)
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setStrokeColor(GRID)
    c.setLineWidth(0.4)
    for y, _ in yticks:
        c.line(left, y, right, y)
    c.setStrokeColor(INK)
    c.setLineWidth(0.8)
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)
    c.setFillColor(INK)
    c.setFont(serif, 8)
    for y, lab in yticks:
        c.drawRightString(left - 5, y - 2.5, lab)
        c.line(left, y, left + 3, y)
    for x, lab in xticks:
        c.saveState()
        c.translate(x, bottom - 12)
        c.rotate(40)
        c.drawCentredString(0, 0, lab)
        c.restoreState()
        c.line(x, bottom, x, bottom + 3)
    c.setFont(serif_i, 9)
    c.drawCentredString((left + right) / 2, 14, xlabel)
    c.saveState()
    c.translate(16, (bottom + top) / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()


def legend(c: canvas.Canvas, x: float, y: float, items: list[tuple[Color, str]]) -> None:
    c.setFont(serif, 8)
    for i, (color, lab) in enumerate(items):
        yy = y - i * 12
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.4)
        c.line(x, yy, x + 16, yy)
        c.circle(x + 8, yy, 1.6, fill=1, stroke=0)
        c.setFillColor(INK)
        c.drawString(x + 22, yy - 2.5, lab)


def rank_curves() -> None:
    rows = list(csv.DictReader(_csv("rank_curves.csv").open()))
    w, h = 420, 280
    left, bottom, right, top = 48, 48, 400, 255
    c = canvas.Canvas(str(OUT / "rank_curves.pdf"), pagesize=(w, h))
    ranks = [float(r["rank"]) for r in rows]
    xmin, xmax = math.log2(ranks[0]), math.log2(ranks[-1])

    def xmap(rank: float) -> float:
        return left + (math.log2(rank) - xmin) / (xmax - xmin) * (right - left)

    def ymap(v: float) -> float:
        return bottom + v * (top - bottom)

    yticks = [(ymap(v), f"{v:.1f}") for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    xticks = [(xmap(r), str(int(r))) for r in ranks]
    frame(c, w, h, left, bottom, right, top, yticks, xticks, "rank", "held-out metric")
    c.setStrokeColor(HexColor("#888888"))
    c.setDash(2, 2)
    c.setLineWidth(0.6)
    c.line(xmap(128), bottom, xmap(128), top)
    c.setDash()
    series = [
        (C1, "similarity_correlation", "pairwise similarity r"),
        (C2, "token_reconstruction_centered_cosine", "token reconstruction"),
        (C3, "relation_reconstruction_centered_cosine", "relation reconstruction"),
        (C4, "proper_name_top1", "proper-name top-1"),
    ]
    for color, key, _lab in series:
        pts = [(xmap(float(r["rank"])), ymap(float(r[key]))) for r in rows]
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.35)
        p = c.beginPath()
        p.moveTo(*pts[0])
        for x, y in pts[1:]:
            p.lineTo(x, y)
        c.drawPath(p, stroke=1, fill=0)
        for x, y in pts:
            c.circle(x, y, 1.8, fill=1, stroke=0)
    legend(c, left + 8, top - 8, [(col, lab) for col, _k, lab in series])
    c.save()


def subspaces() -> None:
    rows = list(csv.DictReader(_csv("rank_subspace.csv").open()))
    by = {}
    for r in rows:
        by.setdefault(r["subspace"], []).append((float(r["rank"]), float(r["centered"])))
    w, h = 420, 260
    left, bottom, right, top = 48, 48, 400, 240
    c = canvas.Canvas(str(OUT / "rank_subspaces.pdf"), pagesize=(w, h))
    ranks = sorted({p[0] for pts in by.values() for p in pts})
    xmin, xmax = math.log2(min(ranks)), math.log2(max(ranks))

    def xmap(rank: float) -> float:
        return left + (math.log2(rank) - xmin) / (xmax - xmin) * (right - left)

    def ymap(v: float) -> float:
        return bottom + v / 0.7 * (top - bottom)

    yticks = [(ymap(v), f"{v:.1f}") for v in (0, 0.2, 0.4, 0.6)]
    xticks = [(xmap(r), str(int(r))) for r in ranks]
    frame(c, w, h, left, bottom, right, top, yticks, xticks, "rank", "centered reconstruction")
    colors = {"mrl": C1, "rand": C2, "svd": C3}
    labels = {"mrl": "leading (Matryoshka)", "rand": "seeded random", "svd": "PCA/SVD"}
    for name, pts in by.items():
        pts = sorted(pts)
        col = colors.get(name, INK)
        c.setStrokeColor(col)
        c.setFillColor(col)
        c.setLineWidth(1.35)
        p = c.beginPath()
        p.moveTo(xmap(pts[0][0]), ymap(pts[0][1]))
        for r, v in pts[1:]:
            p.lineTo(xmap(r), ymap(v))
        c.drawPath(p, stroke=1, fill=0)
        for r, v in pts:
            c.circle(xmap(r), ymap(v), 1.8, fill=1, stroke=0)
    legend(c, left + 8, top - 8, [(colors[k], labels[k]) for k in ("mrl", "rand", "svd") if k in by])
    c.save()


def screen() -> None:
    w, h = 400, 220
    left, bottom, right, top = 48, 42, 380, 200
    c = canvas.Canvas(str(OUT / "original_benchmark.pdf"), pagesize=(w, h))
    names = ["negative gain", "Householder", "projection polarity"]
    proxy = [18, 14, 14]
    literal = [3, 3, 2]
    yticks = [(bottom + v / 24 * (top - bottom), str(v)) for v in (0, 8, 16, 24)]
    slot = (right - left) / 3
    xticks = [(left + (i + 0.5) * slot, n) for i, n in enumerate(names)]
    frame(c, w, h, left, bottom, right, top, yticks, xticks, "", "cells / 24")
    bw = 22
    for i, (p, lit) in enumerate(zip(proxy, literal)):
        cx = left + (i + 0.5) * slot
        h1 = p / 24 * (top - bottom)
        h2 = lit / 24 * (top - bottom)
        c.setFillColor(C1)
        c.rect(cx - bw - 2, bottom, bw, h1, fill=1, stroke=0)
        c.setFillColor(HexColor("#d9d4c8"))
        c.setStrokeColor(C1)
        c.setLineWidth(0.6)
        c.rect(cx + 2, bottom, bw, h2, fill=1, stroke=1)
    c.setFillColor(C1)
    c.rect(left + 8, top - 10, 10, 6, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont(serif, 8)
    c.drawString(left + 22, top - 11, "proxy")
    c.setFillColor(HexColor("#d9d4c8"))
    c.setStrokeColor(C1)
    c.rect(left + 70, top - 10, 10, 6, fill=1, stroke=1)
    c.setFillColor(INK)
    c.drawString(left + 84, top - 11, "literal rescore")
    c.save()


def write_site() -> None:
    sites = ["input", "block 1", "block 4", "block 8", "block 16"]
    plus1 = [0.263, 0.824, 0.635, 0.632, 0.709]
    plus2 = [0.249, 0.561, 0.426, 0.413, 0.515]
    plus3 = [0.193, 0.421, 0.268, 0.289, 0.418]
    w, h = 420, 240
    left, bottom, right, top = 48, 48, 400, 220
    c = canvas.Canvas(str(OUT / "write_site_decay.pdf"), pagesize=(w, h))

    def xmap(i: int) -> float:
        return left + i / 4 * (right - left)

    def ymap(v: float) -> float:
        return bottom + v * (top - bottom)

    yticks = [(ymap(v), f"{v:.1f}") for v in (0, 0.25, 0.5, 0.75, 1.0)]
    xticks = [(xmap(i), n) for i, n in enumerate(sites)]
    frame(c, w, h, left, bottom, right, top, yticks, xticks, "injection site", "cosine with injected direction")
    c.setStrokeColor(HexColor("#888888"))
    c.setDash(2, 2)
    c.setLineWidth(0.6)
    c.line(left, ymap(0.5), right, ymap(0.5))
    c.setDash()
    series = [(C1, plus1, "+1 block"), (C2, plus2, "+2 blocks"), (C3, plus3, "+3 blocks")]
    for color, vals, _lab in series:
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.35)
        p = c.beginPath()
        p.moveTo(xmap(0), ymap(vals[0]))
        for i, v in enumerate(vals[1:], 1):
            p.lineTo(xmap(i), ymap(v))
        c.drawPath(p, stroke=1, fill=0)
        for i, v in enumerate(vals):
            c.circle(xmap(i), ymap(v), 1.8, fill=1, stroke=0)
    legend(c, left + 8, top - 8, [(col, lab) for col, _v, lab in series])
    c.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rank_curves()
    subspaces()
    screen()
    write_site()
    print("wrote rank_curves, rank_subspaces, original_benchmark, write_site_decay PDFs")


if __name__ == "__main__":
    main()
