#!/usr/bin/env python3
"""Render MANUSCRIPT.md to a readable review PDF with evidence-backed figures.

The Markdown and SVG files remain the publication sources. This renderer exists
only because the current workspace has ReportLab but no Pandoc/LaTeX/Typst.
"""

from __future__ import annotations

import csv
import argparse
import html
import json
import math
import re
from pathlib import Path

from reportlab.graphics import renderPM
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
INK = HexColor("#172033")
MUTED = HexColor("#5b6475")
GRID = HexColor("#dfe4ec")
BLUE = HexColor("#2463eb")
RED = HexColor("#cf3e4f")
GREEN = HexColor("#16856b")
PURPLE = HexColor("#7a4cc2")
ORANGE = HexColor("#d87918")


def register_fonts() -> tuple[str, str, str]:
    base = Path("/usr/share/fonts/truetype/dejavu")
    regular, bold, mono = "Helvetica", "Helvetica-Bold", "Courier"
    if (base / "DejaVuSans.ttf").exists():
        pdfmetrics.registerFont(TTFont("PaperSans", base / "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("PaperSans-Bold", base / "DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("PaperMono", base / "DejaVuSansMono.ttf"))
        regular, bold, mono = "PaperSans", "PaperSans-Bold", "PaperMono"
    return regular, bold, mono


FONT, FONT_BOLD, FONT_MONO = register_fonts()


def label(d: Drawing, x: float, y: float, text: str, size=10, color=INK,
          anchor="start", font=FONT):
    d.add(String(x, y, text, fontName=font, fontSize=size, fillColor=color,
                 textAnchor=anchor))


def figure_title(d: Drawing, title: str, subtitle: str):
    label(d, 20, d.height - 28, title, 14, INK, font=FONT_BOLD)
    label(d, 20, d.height - 45, subtitle, 7.5, MUTED)


def system_drawing() -> Drawing:
    d = Drawing(510, 245)
    figure_title(d, "Vector-memory channel and injection-site ablation",
                 "Frozen encoder and target LM; source sentence is evaluator-only")
    items = [
        (18, "Memory text", "evaluator side"),
        (137, "Qwen3 encoder", "frozen, 4096-d"),
        (256, "Ridge map", "one affine bridge"),
        (375, "Pseudo-token slots", "replace embeddings"),
    ]
    for x, title, sub in items:
        d.add(Rect(x, 132, 105, 53, rx=5, ry=5, fillColor=HexColor("#f7f9fc"), strokeColor=GRID))
        label(d, x + 52.5, 162, title, 8.2, anchor="middle", font=FONT_BOLD)
        label(d, x + 52.5, 146, sub, 6.7, MUTED, anchor="middle")
    for x1, x2 in [(123, 137), (242, 256), (361, 375)]:
        d.add(Line(x1, 158, x2, 158, strokeColor=INK, strokeWidth=1.2))
    d.add(Rect(95, 60, 190, 48, rx=5, ry=5, fillColor=HexColor("#eef8f5"), strokeColor=GREEN))
    label(d, 105, 91, "Embedding input: readable oracle channel", 8, GREEN, font=FONT_BOLD)
    label(d, 105, 74, "All transformer blocks remain downstream", 7, INK)
    d.add(Rect(310, 60, 180, 48, rx=5, ry=5, fillColor=HexColor("#fff3f4"), strokeColor=RED))
    label(d, 320, 91, "Final post-norm: no recall", 8, RED, font=FONT_BOLD)
    label(d, 320, 74, "~300 generations; control leaked artifact", 7, INK)
    d.add(Line(425, 132, 403, 108, strokeColor=RED, strokeWidth=1.1))
    d.add(Line(402, 132, 242, 108, strokeColor=GREEN, strokeWidth=1.1))
    label(d, 20, 22, "Sources: soft-slot runner and 2026-08-24 injection-site research log.", 6.5, MUTED)
    return d


def two_write_regimes_drawing() -> Drawing:
    d = Drawing(510, 270)
    figure_title(d, "Two positive cross-model write regimes",
                 "Frozen endpoints; affine bridges; distinct bandwidth and endpoints")
    lanes = [
        (150, BLUE, "A · ontological inversion",
         [(16, "Nomic", "frozen"), (136, "128-d vector", "normalized lead-128"),
          (256, "Affine map", "128→896"), (376, "Qwen · layer 4", "semantic basin")]),
        (63, GREEN, "B · ordered reconstruction",
         [(16, "Qwen3", "frozen 4096-d"), (136, "Fragments", "ordered"),
          (256, "Ridge map", "4096→4096"), (376, "Llama · input", "content + use")]),
    ]
    for y, color, lane, boxes in lanes:
        label(d, 16, y + 67, lane, 8.5, color, font=FONT_BOLD)
        for x, title, sub in boxes:
            d.add(Rect(x, y, 105, 50, rx=5, ry=5, fillColor=HexColor("#f7f9fc"), strokeColor=color))
            label(d, x + 52.5, y + 31, title, 7.6, anchor="middle", font=FONT_BOLD)
            label(d, x + 52.5, y + 16, sub, 6.2, MUTED, anchor="middle")
        for x1, x2 in [(121, 136), (241, 256), (361, 376)]:
            d.add(Line(x1, y + 25, x2, y + 25, strokeColor=color, strokeWidth=1.5))
    d.add(Rect(38, 14, 434, 29, rx=5, ry=5, fillColor=HexColor("#fff8e9"), strokeColor=ORANGE))
    label(d, 255, 31, "Compact steering changes a basin; ordered slots preserve propositions.", 7.1, ORANGE, anchor="middle", font=FONT_BOLD)
    return d


def original_benchmark_drawing() -> Drawing:
    d = Drawing(500, 285)
    figure_title(d, "Original 360-run screen: breadth and collapse",
                 "Best noncollapsed of five strengths; 24 model×concept cells per arm")
    rows = [
        ("negative gain", 18, .47619, BLUE),
        ("Householder dir.", 14, .61304, GREEN),
        ("projection polarity", 14, .45, ORANGE),
    ]
    bar_left, bar_w = 110, 190
    dot_left, dot_w = 355, 120
    label(d, bar_left, 235, "proxy cells / 24", 7, INK, font=FONT_BOLD)
    label(d, dot_left, 235, "collapse onset", 7, INK, font=FONT_BOLD)
    for i, (name, passed, onset, color) in enumerate(rows):
        y = 190 - i * 58
        label(d, bar_left - 8, y - 2, name, 6.5, INK, anchor="end")
        d.add(Rect(bar_left, y - 7, bar_w, 15, rx=3, ry=3,
                   fillColor=HexColor("#f1f3f7"), strokeColor=None))
        d.add(Rect(bar_left, y - 7, bar_w * passed / 24, 15, rx=3, ry=3,
                   fillColor=color, strokeColor=color))
        label(d, bar_left + bar_w + 6, y - 2, f"{passed}/24", 6.5, color, font=FONT_BOLD)
        endpoint = dot_left + dot_w * onset / .8
        d.add(Line(dot_left, y, endpoint, y, strokeColor=color, strokeWidth=2))
        d.add(Circle(endpoint, y, 4.5, fillColor=color, strokeColor=color))
        label(d, endpoint, y + 10, f"{onset:.2f}", 6.2, color, anchor="middle", font=FONT_BOLD)
    d.add(Rect(45, 22, 420, 34, rx=4, ry=4, fillColor=HexColor("#fff8e9"), strokeColor=ORANGE))
    label(d, 255, 43, "Same-encoder proxy · handwritten anchors · post-hoc strength choice", 6.4,
          ORANGE, anchor="middle", font=FONT_BOLD)
    return d


def ontological_gain_drawing() -> Drawing:
    d = Drawing(500, 285)
    figure_title(d, "Nomic→Qwen: narrow gain islands and matched controls",
                 "Dense gain grid and five-cell matched controls")
    left, bottom, width = 50, 158, 425
    sx = lambda gain: left + (gain + .30) / .35 * width
    for gain in [-.30, -.25, -.20, -.15, -.10, -.05, 0, .05]:
        x = sx(gain)
        d.add(Line(x, bottom, x, bottom + 50, strokeColor=GRID, strokeWidth=.5))
        label(d, x, bottom - 13, f"{gain:.2f}", 5.7, MUTED, anchor="middle")
    d.add(Line(left, bottom + 25, left + width, bottom + 25, strokeColor=INK, strokeWidth=1.5))
    for lo, hi, name, color in [(-.21, -.18, "stove", BLUE), (-.17, -.16, "gap", MUTED), (-.15, -.14, "fire pit", ORANGE)]:
        d.add(Rect(sx(lo), bottom + 12, sx(hi) - sx(lo), 26, rx=3, ry=3, fillColor=color, strokeColor=color))
        label(d, (sx(lo) + sx(hi)) / 2, bottom + 46, name, 5.7, color, anchor="middle", font=FONT_BOLD)
    rows = [("concept / negative", 5, BLUE), ("unrelated adapter", 4, ORANGE),
            ("random", 0, RED), ("coordinate shuffle", 0, RED), ("concept / positive", 0, RED)]
    for i, (name, value, color) in enumerate(rows):
        y = 119 - i * 20
        label(d, 125, y - 2, name, 6.3, INK, anchor="end")
        d.add(Rect(135, y - 5, 290, 10, rx=2, ry=2, fillColor=HexColor("#f1f3f7"), strokeColor=None))
        if value:
            d.add(Rect(135, y - 5, 290 * value / 5, 10, rx=2, ry=2, fillColor=color, strokeColor=color))
        label(d, 435, y - 2, f"{value}/5", 6.3, color, font=FONT_BOLD)
    return d


def bias_residual_drawing() -> Drawing:
    d = Drawing(500, 285)
    figure_title(d, "Bias is sufficient; too much concept residual cancels it",
                 "Inverting gains in the shipped adapter coordinates")
    rows = [("bias", 11), ("b+.25Wv", 14), ("b+.5Wv", 17), ("b+Wv", 7),
            ("b+2Wv", 3), ("b+4Wv", 2), ("residual", 0), ("wolf residual", 6)]
    left, width = 100, 355
    for i, (name, value) in enumerate(rows):
        y = 215 - i * 23
        color = GREEN if name == "bias" else (RED if name == "residual" else BLUE)
        label(d, left - 8, y - 2, name, 6.4, INK, anchor="end")
        if value:
            d.add(Rect(left, y - 5, width * value / 18, 11, rx=2, ry=2, fillColor=color, strokeColor=color))
        label(d, left + width * value / 18 + 7, y - 2, str(value), 6.2, color, font=FONT_BOLD)
    d.add(Rect(65, 18, 390, 25, rx=4, ry=4, fillColor=HexColor("#fff8e9"), strokeColor=ORANGE))
    label(d, 260, 33, "cos(b,Wv)=−0.6056 · residual norm exceeds bias norm", 6.8, ORANGE, anchor="middle", font=FONT_BOLD)
    return d


def cross_model_evidence_drawing() -> Drawing:
    d = Drawing(500, 295)
    figure_title(d, "Qwen3→Llama crosses into semantic transmission",
                 "Span-aware ridge transmits memory-specific content")
    rows = [
        ("pooled", .238, "I find nothing / Oxford comma", RED),
        ("chunks", .615, "A worm is a fire hydrant", ORANGE),
        ("span · finish", .647, "fire burning hamster", BLUE),
        ("span · open", .647, "wolf · fire breathing dragon", BLUE),
        ("Llama oracle", .70, "exact written sentence", GREEN),
    ]
    left, width = 120, 350
    for i, (name, value, output, color) in enumerate(rows):
        y = 225 - i * 38
        label(d, left - 8, y - 2, name, 6.5, INK, anchor="end", font=FONT_BOLD)
        d.add(Rect(left, y - 6, width, 13, rx=3, ry=3, fillColor=HexColor("#f1f3f7"), strokeColor=None))
        d.add(Rect(left, y - 6, width * value / .70, 13, rx=3, ry=3, fillColor=color, strokeColor=color))
        label(d, left, y - 18, output, 5.9, INK)
    d.add(Rect(50, 15, 420, 25, rx=4, ry=4, fillColor=HexColor("#eef8f5"), strokeColor=GREEN))
    label(d, 260, 30, "Across two span framings: fire + breathing + hamster cross models.", 6.6, GREEN, anchor="middle", font=FONT_BOLD)
    return d


def chart_axes(d: Drawing, left, bottom, width, height, ymax=1.0):
    for i in range(6):
        value = ymax * i / 5
        y = bottom + height * i / 5
        d.add(Line(left, y, left + width, y, strokeColor=GRID, strokeWidth=0.5))
        label(d, left - 7, y - 2, f"{value:.1f}", 6.5, MUTED, anchor="end")


def rank_drawing() -> Drawing:
    rows = list(csv.DictReader((ROOT / "synapse/rank_curves.csv").open()))
    d = Drawing(500, 300)
    figure_title(d, "Retrieval similarity saturates before reconstruction precision",
                 "Fixed 90/10 split of 32,000 span pairs; 49,990 held-out cosine pairs")
    left, bottom, width, height = 48, 48, 425, 195
    chart_axes(d, left, bottom, width, height)
    xs = [math.log2(float(row["rank"])) for row in rows]
    lo, hi = min(xs), max(xs)
    sx = lambda x: left + (x - lo) / (hi - lo) * width
    sy = lambda y: bottom + y * height
    metrics = [
        ("similarity_correlation", BLUE),
        ("token_reconstruction_centered_cosine", RED),
        ("proper_name_top1", PURPLE),
        ("relation_reconstruction_centered_cosine", GREEN),
    ]
    for key, color in metrics:
        pts = [(sx(x), sy(float(row[key]))) for x, row in zip(xs, rows)]
        d.add(PolyLine([v for point in pts for v in point], strokeColor=color, strokeWidth=1.7))
        for x, y in pts:
            d.add(Circle(x, y, 2.1, fillColor=color, strokeColor=color))
    for x, row in zip(xs, rows):
        label(d, sx(x), bottom - 14, row["rank"], 5.8, MUTED, anchor="middle")
    x128 = sx(math.log2(128))
    d.add(Line(x128, bottom, x128, bottom + height, strokeColor=ORANGE, strokeWidth=1))
    label(d, x128 + 4, bottom + height - 10, "rank 128: sim .937 / recon .346", 6.5, ORANGE)
    legends = [("similarity", BLUE), ("token reconstruction", RED),
               ("proper-name top-1", PURPLE), ("relation reconstruction", GREEN)]
    for i, (name, color) in enumerate(legends):
        x = 235 + (i % 2) * 120
        y = 266 - (i // 2) * 13
        d.add(Line(x, y, x + 14, y, strokeColor=color, strokeWidth=2))
        label(d, x + 18, y - 2, name, 6.3, INK)
    return d


def subspace_drawing() -> Drawing:
    rows = list(csv.DictReader((ROOT / "synapse/rank_subspace.csv").open()))
    values = {}
    for row in rows:
        values.setdefault(int(row["rank"]), {})[row["subspace"]] = float(row["centered"])
    ranks = sorted(values)
    d = Drawing(500, 285)
    figure_title(d, "Matched-rank subspaces: PCA/SVD leads reconstruction",
                 "Held-out centered cosine; one seeded split")
    left, bottom, width, height = 55, 45, 415, 185
    chart_axes(d, left, bottom, width, height, ymax=0.72)
    sx = lambda i: left + i / (len(ranks) - 1) * width
    sy = lambda y: bottom + y / 0.72 * height
    series = [("mrl", BLUE, "Matryoshka"), ("svd", GREEN, "PCA/SVD"), ("rand", RED, "random")]
    for key, color, name in series:
        pts = [(sx(i), sy(values[r][key])) for i, r in enumerate(ranks)]
        d.add(PolyLine([v for pt in pts for v in pt], strokeColor=color, strokeWidth=1.8))
        for x, y in pts:
            d.add(Circle(x, y, 2.4, fillColor=color, strokeColor=color))
    for i, rank in enumerate(ranks):
        label(d, sx(i), bottom - 14, str(rank), 6.5, MUTED, anchor="middle")
    for i, (_, color, name) in enumerate(series):
        x = 325 + (i % 2) * 85
        y = 253 - (i // 2) * 13
        d.add(Line(x, y, x + 13, y, strokeColor=color, strokeWidth=2))
        label(d, x + 17, y - 2, name, 6.5)
    return d


def write_site_decay_drawing() -> Drawing:
    d = Drawing(510, 290)
    figure_title(d, "Write lifetime depends on injection site",
                 "One disclosed fact, gain 0.825; absolute direction and plus/minus fold")
    offsets = [0, 1, 2, 3, 5]
    absolute = {
        "embedding": [1.0, .263, .249, .193, .124],
        "block 1": [1.0, .824, .561, .421, .187],
        "block 4": [1.0, .635, .426, .268, .127],
        "block 8": [1.0, .632, .413, .289, .165],
        "block 16": [1.0, .709, .515, .418, .288],
    }
    fold = {
        "block 1": [-1.0, -.868, -.561, -.469, -.326],
        "block 4": [-1.0, -.896, -.815, -.752, -.668],
        "block 8": [-1.0, -.828, -.695, -.636, -.463],
        "block 16": [-1.0, -.823, -.625, -.567, -.421],
    }
    palette = {"embedding": MUTED, "block 1": BLUE, "block 4": GREEN,
               "block 8": ORANGE, "block 16": PURPLE}

    def panel(left, title, series, ymin, ymax, threshold):
        bottom, width, height = 58, 195, 160
        sx = lambda value: left + value / 5 * width
        sy = lambda value: bottom + (value - ymin) / (ymax - ymin) * height
        label(d, left, 233, title, 7.2, INK, font=FONT_BOLD)
        for value in [ymin, threshold, ymax]:
            y = sy(value)
            d.add(Line(left, y, left + width, y, strokeColor=GRID, strokeWidth=.6,
                       strokeDashArray=[3, 2] if value == threshold else None))
            label(d, left - 6, y - 2, f"{value:.1f}", 5.8, MUTED, anchor="end")
        for offset in offsets:
            x = sx(offset)
            d.add(Line(x, bottom, x, bottom + height, strokeColor=GRID, strokeWidth=.35))
            label(d, x, bottom - 12, str(offset), 5.8, MUTED, anchor="middle")
        for name, values in series.items():
            points = [(sx(offset), sy(value)) for offset, value in zip(offsets, values)]
            color = palette[name]
            d.add(PolyLine([v for point in points for v in point], strokeColor=color, strokeWidth=1.5))
            for x, y in points:
                d.add(Circle(x, y, 1.8, fillColor=color, strokeColor=color))
        label(d, left + width / 2, 31, "blocks after write", 6.2, INK, anchor="middle")

    panel(48, "A · absolute survival cosine", absolute, 0.0, 1.0, .5)
    panel(300, "B · plus/minus fold cosine", fold, -1.0, 0.0, -.5)
    for i, name in enumerate(["embedding", "block 1", "block 4", "block 8", "block 16"]):
        x = 62 + i * 87
        color = palette[name]
        d.add(Line(x, 257, x + 12, 257, strokeColor=color, strokeWidth=1.8))
        label(d, x + 16, 255, name, 5.6, INK)
    label(d, 20, 12, "Embedding-site fold omitted: sign flips after block 0, invalidating the instrument.", 5.8, MUTED)
    return d


def e7_drawing() -> Drawing:
    rows = [
        ("neutral / memory", .080, .206),
        ("neutral / blank", .364, .426),
        ("anti-guess / memory", .040, .160),
        ("anti-guess / blank", .198, .318),
    ]
    d = Drawing(500, 285)
    figure_title(d, "Bounded-knowledge criterion failed; answer propensity shifted",
                 "Decline fraction; n=10 memory clusters; 6,400 nested generations")
    left, bottom, width = 145, 72, 315
    sx = lambda x: left + x / .5 * width
    for tick in range(6):
        value = tick / 10
        x = sx(value)
        d.add(Line(x, bottom, x, 225, strokeColor=GRID, strokeWidth=.5))
        label(d, x, bottom - 14, f"{value:.1f}", 6.5, MUTED, anchor="middle")
    for i, (name, a, u) in enumerate(rows):
        y = 202 - i * 35
        label(d, left - 10, y - 2, name, 7.3, INK, anchor="end")
        d.add(Line(sx(a), y, sx(u), y, strokeColor=MUTED, strokeWidth=1.2))
        d.add(Circle(sx(a), y, 4, fillColor=BLUE, strokeColor=BLUE))
        d.add(Circle(sx(u), y, 4, fillColor=RED, strokeColor=RED))
        label(d, sx(a), y + 8, f"{a:.3f}", 6, BLUE, anchor="middle")
        label(d, sx(u), y + 8, f"{u:.3f}", 6, RED, anchor="middle")
    d.add(Rect(38, 23, 425, 31, rx=4, ry=4, fillColor=HexColor("#fff7e8"), strokeColor=ORANGE))
    label(d, 50, 42, "Primary difference +0.126; 95% cluster CI [+0.080,+0.177]", 7.2, ORANGE, font=FONT_BOLD)
    label(d, 50, 29, "Entire interval below preregistered 0.25 threshold.", 6.8, INK)
    return d


def e10_drawing() -> Drawing:
    rows = [("aligned adapter", 0), ("reversed slots", 0), ("wrong memory", 0),
            ("random direction", 0), ("dimension shuffle", 0), ("oracle", 3)]
    d = Drawing(500, 285)
    figure_title(d, "A narrow corpus-clean boundary for the fixed ridge bridge",
                 "E10 does not erase earlier Qwen3→Llama semantic transmission")
    left, bottom, width = 150, 55, 315
    sx = lambda x: left + x / 9 * width
    for tick in range(10):
        x = sx(tick)
        d.add(Line(x, bottom, x, 225, strokeColor=GRID, strokeWidth=.5))
        label(d, x, bottom - 13, str(tick), 6.3, MUTED, anchor="middle")
    for i, (name, value) in enumerate(rows):
        y = 207 - i * 27
        label(d, left - 10, y - 2, name, 7.3, INK, anchor="end")
        color = BLUE if name == "oracle" else RED
        d.add(Line(left, y, sx(value), y, strokeColor=MUTED, strokeWidth=2.5))
        d.add(Circle(sx(value), y, 4.5, fillColor=color, strokeColor=color))
        label(d, sx(value) + 10, y - 2, f"{value}/9", 7, INK, font=FONT_BOLD)
    label(d, 20, 24, "Aligned adapter also scored 0/9 at gains 0.75 and 0.90; paired blank leakage 0/9.", 6.8, MUTED)
    return d


def make_figures(export_png: bool = False) -> dict[str, Drawing]:
    figures = {
        "system_and_site.svg": system_drawing(),
        "two_write_regimes.svg": two_write_regimes_drawing(),
        "original_benchmark.svg": original_benchmark_drawing(),
        "ontological_gain_island.svg": ontological_gain_drawing(),
        "bias_residual_decomposition.svg": bias_residual_drawing(),
        "cross_model_evidence.svg": cross_model_evidence_drawing(),
        "rank_curves.svg": rank_drawing(),
        "rank_subspaces.svg": subspace_drawing(),
        "write_site_decay.svg": write_site_decay_drawing(),
        "e7_declines.svg": e7_drawing(),
        "e10_adapter_join.svg": e10_drawing(),
    }
    if export_png:
        for name, drawing in figures.items():
            renderPM.drawToFile(drawing, str(FIGURES / name.replace(".svg", ".png")), fmt="PNG", dpi=120)
    return figures


CITATION_LABELS = {
    "lewis2020rag": "Lewis et al., 2020",
    "cheng2024xrag": "Cheng et al., 2024",
    "lester2021prompt": "Lester et al., 2021",
    "li2021prefix": "Li and Liang, 2021",
    "mu2023gist": "Mu et al., 2023",
    "chevalier2023autocompressors": "Chevalier et al., 2023",
    "ge2023icae": "Ge et al., 2023",
    "panickssery2023caa": "Panickssery et al., 2023",
    "zou2023representation": "Zou et al., 2023",
    "kusupati2022mrl": "Kusupati et al., 2022",
    "zhang2025qwen3embedding": "Zhang et al., 2025",
    "grattafiori2024llama3": "Grattafiori et al., 2024",
}


def inline_markup(text: str) -> str:
    # ReportLab Paragraph does not typeset TeX. Convert the small notation used
    # in this manuscript to readable Unicode/plain math for the review PDF.
    def citation(match: re.Match[str]) -> str:
        keys = [part.strip().lstrip("@") for part in match.group(1).split(";")]
        labels = [CITATION_LABELS.get(key, key) for key in keys]
        return "(" + "; ".join(labels) + ")"

    text = re.sub(r"\[(@[^\]]+)\]", citation, text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    replacements = {
        r"\mathbb{R}": "R", r"\in": "∈", r"\top": "ᵀ", r"\mu": "μ",
        r"\lambda": "λ", r"\alpha": "α", r"\varepsilon": "ε",
        r"\leftarrow": "←", r"\sum": "Σ", r"\mid": "|", r"\cos": "cos",
        r"\hat": "", r"\bar": "", r"\qquad": "  ", r"\,": " ",
        r"\|": "‖",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("$$", "").replace("$", "").replace("\\", "")
    text = text.replace("{", "").replace("}", "")
    text = html.escape(text.strip())
    text = re.sub(r"`([^`]+)`", r"<font name='PaperMono'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def parse_table(lines: list[str], style: ParagraphStyle, max_width: float) -> Table:
    cells = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    if len(cells) > 1 and all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells[1]):
        cells.pop(1)
    ncols = max(len(row) for row in cells)
    for row in cells:
        row.extend([""] * (ncols - len(row)))
    data = [[Paragraph(inline_markup(c), style) for c in row] for row in cells]
    table = Table(data, colWidths=[max_width / ncols] * ncols, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#eef1f6")),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build_pdf(source: Path, destination: Path, figures: dict[str, Drawing]) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("PaperBody", parent=styles["BodyText"], fontName=FONT,
                          fontSize=9.1, leading=12.2, alignment=TA_JUSTIFY,
                          textColor=INK, spaceAfter=6)
    table_text = ParagraphStyle("TableText", parent=body, fontSize=6.3, leading=7.6,
                                alignment=0, spaceAfter=0)
    quote = ParagraphStyle("Quote", parent=body, leftIndent=18, rightIndent=18,
                           borderColor=GRID, borderWidth=1, borderPadding=7,
                           textColor=HexColor("#30394b"))
    h1 = ParagraphStyle("Title", parent=styles["Title"], fontName=FONT_BOLD,
                        fontSize=20, leading=24, textColor=INK, alignment=TA_CENTER,
                        spaceAfter=16)
    h2 = ParagraphStyle("H2", parent=styles["Heading1"], fontName=FONT_BOLD,
                        fontSize=14, leading=17, textColor=INK, spaceBefore=12,
                        spaceAfter=7, keepWithNext=True)
    h3 = ParagraphStyle("H3", parent=styles["Heading2"], fontName=FONT_BOLD,
                        fontSize=11, leading=14, textColor=INK, spaceBefore=9,
                        spaceAfter=5, keepWithNext=True)
    caption = ParagraphStyle("Caption", parent=body, fontSize=7.5, leading=9,
                             alignment=TA_CENTER, textColor=MUTED, spaceAfter=9)
    code_style = ParagraphStyle("Code", fontName=FONT_MONO, fontSize=7.1,
                                leading=9, leftIndent=10, rightIndent=10,
                                textColor=INK, backColor=HexColor("#f5f6f8"),
                                borderPadding=6, spaceAfter=7)

    doc = SimpleDocTemplate(str(destination), pagesize=letter,
                            rightMargin=.62 * inch, leftMargin=.62 * inch,
                            topMargin=.58 * inch, bottomMargin=.6 * inch,
                            title="Writing Meaning Between Frozen Models",
                            author="Jason Van Pham (ruffian-l), independent researcher; Gemini, Grok, ChatGPT, and Claude, AI research collaborators")
    story = []
    lines = source.read_text().splitlines()
    para: list[str] = []
    in_code = False
    code: list[str] = []

    def flush_para():
        if para:
            story.append(Paragraph(inline_markup(" ".join(x.strip() for x in para)), body))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_para()
            if in_code:
                story.append(Preformatted("\n".join(code), code_style))
                code.clear()
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]+)\]\(figures/([^)]+)\)", line.strip())
        if image_match:
            flush_para()
            name = image_match.group(2)
            drawing = figures[name]
            scale = min(1.0, doc.width / drawing.width)
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)
            story.append(KeepTogether([drawing, Paragraph(inline_markup(image_match.group(1)), caption)]))
            i += 1
            continue
        if line.startswith("|"):
            flush_para()
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(parse_table(table_lines, table_text, doc.width))
            story.append(Spacer(1, 7))
            continue
        if line.startswith("# "):
            flush_para()
            if story:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(line[2:]), h1))
        elif line.startswith("## "):
            flush_para()
            story.append(Paragraph(inline_markup(line[3:]), h2))
        elif line.startswith("### "):
            flush_para()
            story.append(Paragraph(inline_markup(line[4:]), h3))
        elif line.startswith("> "):
            flush_para()
            qlines = []
            while i < len(lines) and (lines[i].startswith(">") or not lines[i].strip()):
                if lines[i].startswith(">"):
                    qlines.append(lines[i].lstrip("> "))
                i += 1
            story.append(Paragraph(inline_markup(" ".join(qlines)), quote))
            continue
        elif re.match(r"^[-*] ", line):
            flush_para()
            story.append(Paragraph("• " + inline_markup(line[2:]), body))
        elif re.match(r"^\d+\. ", line):
            flush_para()
            story.append(Paragraph(inline_markup(line), body))
        elif not line.strip():
            flush_para()
        else:
            para.append(line)
        i += 1
    flush_para()

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(FONT, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(document.leftMargin, .32 * inch, "Writing Meaning Between Frozen Models — preprint draft")
        canvas.drawRightString(letter[0] - document.rightMargin, .32 * inch, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true", help="also export slower raster previews")
    args = ap.parse_args()
    figures = make_figures(export_png=args.png)
    out = PAPER / "Writing-Meaning-Between-Frozen-Models.pdf"
    build_pdf(PAPER / "MANUSCRIPT.md", out, figures)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
