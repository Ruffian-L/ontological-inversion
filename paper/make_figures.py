#!/usr/bin/env python3
"""Render paper SVGs from committed niodoo-live evidence artifacts.

No model is loaded and no experimental result is changed. The script reads the
committed CSV/JSONL receipts, reproduces descriptive rates, and writes only SVG.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#172033"
MUTED = "#5b6475"
GRID = "#dfe4ec"
BLUE = "#2463eb"
RED = "#cf3e4f"
GREEN = "#16856b"
PURPLE = "#7a4cc2"
ORANGE = "#d87918"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_start(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="48" y="42" font-family="DejaVu Sans,Arial,sans-serif" font-size="22" font-weight="700" fill="{INK}">{esc(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="48" y="66" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{MUTED}">{esc(subtitle)}</text>'
        )
    return parts


def save(name: str, parts: list[str]) -> None:
    parts.append("</svg>")
    (OUT / name).write_text("\n".join(parts) + "\n")


def figure_system() -> None:
    width, height = 1120, 520
    p = svg_start(
        width,
        height,
        "Vector-memory channel and the injection-site ablation",
        "Frozen source encoder and target LM; the source sentence is evaluator-only at inference",
    )
    boxes = [
        (55, 125, 205, 108, "Memory text", "evaluator side"),
        (300, 125, 205, 108, "Qwen3-Embedding-8B", "frozen retrieval encoder"),
        (545, 125, 205, 108, "Ridge map  W x + b", "single fitted linear bridge"),
        (790, 125, 270, 108, "Pseudo-token slots", "replace marked input embeddings"),
    ]
    for x, y, w, h, label, sub in boxes:
        p += [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#f7f9fc" stroke="{GRID}" stroke-width="2"/>',
            f'<text x="{x+w/2}" y="{y+43}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="16" font-weight="700" fill="{INK}">{esc(label)}</text>',
            f'<text x="{x+w/2}" y="{y+72}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{MUTED}">{esc(sub)}</text>',
        ]
    for x1, x2 in [(260, 300), (505, 545), (750, 790)]:
        p += [
            f'<line x1="{x1}" y1="179" x2="{x2-10}" y2="179" stroke="{INK}" stroke-width="2.5"/>',
            f'<path d="M {x2-10} 173 L {x2} 179 L {x2-10} 185 Z" fill="{INK}"/>',
        ]

    p += [
        f'<rect x="345" y="305" width="355" height="118" rx="12" fill="#eef8f5" stroke="{GREEN}" stroke-width="2"/>',
        f'<text x="365" y="336" font-family="DejaVu Sans,Arial,sans-serif" font-size="15" font-weight="700" fill="{GREEN}">Embedding input (before block 0)</text>',
        f'<text x="365" y="363" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">All transformer blocks remain downstream.</text>',
        f'<text x="365" y="388" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">Oracle vectors: content-sensitive reconstruction and use.</text>',
        f'<rect x="730" y="305" width="335" height="118" rx="12" fill="#fff3f4" stroke="{RED}" stroke-width="2"/>',
        f'<text x="750" y="336" font-family="DejaVu Sans,Arial,sans-serif" font-size="15" font-weight="700" fill="{RED}">Final post-norm</text>',
        f'<text x="750" y="363" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">Only the output projection remains.</text>',
        f'<text x="750" y="388" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">~300 generations: no recall; control leaked disturbance.</text>',
        f'<line x1="925" y1="233" x2="925" y2="295" stroke="{INK}" stroke-width="2"/>',
        f'<path d="M 919 295 L 925 305 L 931 295 Z" fill="{INK}"/>',
        f'<line x1="820" y1="233" x2="620" y2="295" stroke="{GREEN}" stroke-width="2"/>',
        f'<path d="M 618 289 L 610 298 L 622 300 Z" fill="{GREEN}"/>',
        f'<text x="48" y="480" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">Sources: niodoo/src/bin/synapse_soft_slot.rs; research_logs/2026-08-24_synapse-recall-wrong-end-of-the-stack.md.</text>',
    ]
    save("system_and_site.svg", p)


def figure_two_write_regimes() -> None:
    width, height = 1120, 590
    p = svg_start(
        width,
        height,
        "Two positive cross-model write regimes",
        "Frozen endpoints, one affine bridge in each path, different bandwidth and behavioral endpoints",
    )
    lanes = [
        (112, "A · ontological inversion", BLUE, [
            (55, 210, "Nomic v1.5", "frozen encoder"),
            (305, 210, "128-d vector", "normalized leading coords"),
            (555, 210, "Affine 128→896", "shipped Synapse"),
            (805, 250, "Qwen2.5-0.5B · layer 4", "sign-sensitive object basin"),
        ]),
        (335, "B · ordered reconstruction", GREEN, [
            (55, 210, "Qwen3-Embedding-8B", "frozen 4096-d"),
            (305, 210, "Text fragments", "ordered before mapping"),
            (555, 210, "Ridge 4096→4096", "one shared affine map"),
            (805, 250, "Llama-3.1-8B · input", "content, context, inference"),
        ]),
    ]
    for y, lane_label, color, boxes in lanes:
        p.append(f'<text x="55" y="{y-18}" font-family="DejaVu Sans,Arial,sans-serif" font-size="17" font-weight="700" fill="{color}">{esc(lane_label)}</text>')
        for x, w, title, sub in boxes:
            p += [
                f'<rect x="{x}" y="{y}" width="{w}" height="92" rx="12" fill="#f7f9fc" stroke="{color}" stroke-width="2"/>',
                f'<text x="{x+w/2}" y="{y+37}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="15" font-weight="700" fill="{INK}">{esc(title)}</text>',
                f'<text x="{x+w/2}" y="{y+64}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{MUTED}">{esc(sub)}</text>',
            ]
        for x1, x2 in [(265, 305), (515, 555), (765, 805)]:
            p += [
                f'<line x1="{x1}" y1="{y+46}" x2="{x2-10}" y2="{y+46}" stroke="{color}" stroke-width="3"/>',
                f'<path d="M {x2-10} {y+40} L {x2} {y+46} L {x2-10} {y+52} Z" fill="{color}"/>',
            ]
    p += [
        f'<rect x="55" y="470" width="1015" height="72" rx="12" fill="#fff8e9" stroke="{ORANGE}" stroke-width="2"/>',
        f'<text x="75" y="500" font-family="DejaVu Sans,Arial,sans-serif" font-size="14" font-weight="700" fill="{ORANGE}">Same existence result, different rate–distortion point</text>',
        f'<text x="75" y="525" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">Compact steering changes a semantic basin; ordered high-bandwidth slots preserve identity, relation, and usable propositions.</text>',
        f'<text x="48" y="575" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Sources: ontological-inversion run cards; niodoo-live soft-slot, oracle, contextual-use, and inference transcripts.</text>',
    ]
    save("two_write_regimes.svg", p)


def figure_original_benchmark() -> None:
    width, height = 1080, 600
    p = svg_start(
        width,
        height,
        "Original 360-run screen: broad proxy flips, operator-dependent collapse",
        "Best noncollapsed of five strengths; Nomic anchor proxy; 24 model×concept cells per arm",
    )
    rows = [
        ("negative gain", 18, 0.47619, BLUE),
        ("Householder direction", 14, 0.61304, GREEN),
        ("projection polarity", 14, 0.45, ORANGE),
    ]

    p += [
        f'<text x="70" y="115" font-family="DejaVu Sans,Arial,sans-serif" font-size="15" font-weight="700" fill="{INK}">Proxy-flip cells</text>',
        f'<text x="630" y="115" font-family="DejaVu Sans,Arial,sans-serif" font-size="15" font-weight="700" fill="{INK}">Mean observed collapse onset</text>',
    ]
    bar_left, bar_top, bar_w = 225, 170, 330
    dot_left, dot_w = 690, 300
    for tick in [0, 6, 12, 18, 24]:
        x = bar_left + tick / 24 * bar_w
        p += [
            f'<line x1="{x:.1f}" y1="145" x2="{x:.1f}" y2="425" stroke="{GRID}"/>',
            f'<text x="{x:.1f}" y="450" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{tick}</text>',
        ]
    for tick in [0, 0.2, 0.4, 0.6, 0.8]:
        x = dot_left + tick / 0.8 * dot_w
        p += [
            f'<line x1="{x:.1f}" y1="145" x2="{x:.1f}" y2="425" stroke="{GRID}"/>',
            f'<text x="{x:.1f}" y="450" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{tick:.1f}</text>',
        ]
    for i, (name, passed, onset, color) in enumerate(rows):
        y = bar_top + i * 100
        pct = 100 * passed / 24
        endpoint = dot_left + onset / 0.8 * dot_w
        p += [
            f'<text x="{bar_left-18}" y="{y+6}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">{esc(name)}</text>',
            f'<rect x="{bar_left}" y="{y-15}" width="{bar_w}" height="30" rx="6" fill="#f1f3f7"/>',
            f'<rect x="{bar_left}" y="{y-15}" width="{bar_w*passed/24:.1f}" height="30" rx="6" fill="{color}"/>',
            f'<text x="{bar_left+bar_w+14}" y="{y+6}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" font-weight="700" fill="{color}">{passed}/24 · {pct:.0f}%</text>',
            f'<line x1="{dot_left}" y1="{y}" x2="{endpoint:.1f}" y2="{y}" stroke="{color}" stroke-width="4"/>',
            f'<circle cx="{endpoint:.1f}" cy="{y}" r="9" fill="{color}"/>',
            f'<text x="{endpoint:.1f}" y="{y-18}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" font-weight="700" fill="{color}">{onset:.2f}</text>',
        ]
    p += [
        f'<rect x="65" y="490" width="950" height="62" rx="10" fill="#fff8e9" stroke="{ORANGE}"/>',
        f'<text x="540" y="516" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="12.5" font-weight="700" fill="{ORANGE}">Breadth evidence, not a held-out rate</text>',
        f'<text x="540" y="538" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11.5" fill="{INK}">Same-encoder scorer · handwritten anchors · post-hoc strength selection · greedy n=1 per cell</text>',
        f'<text x="48" y="584" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: original results/benchmark.csv, commit bd990416…; values independently re-aggregated from all 360 rows.</text>',
    ]
    save("original_benchmark.svg", p)


def figure_ontological_gain_island() -> None:
    width, height = 1040, 600
    p = svg_start(
        width,
        height,
        "Nomic→Qwen: narrow gain islands and matched controls",
        "Glub-Tub prompt; greedy decode; dense 0.01 gain grid and five-cell matched controls",
    )
    left, top, pw = 100, 135, 840
    sx = lambda gain: left + (gain + 0.30) / 0.35 * pw
    for gain in [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05]:
        x = sx(gain)
        p += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+125}" stroke="{GRID}"/>',
            f'<text x="{x:.1f}" y="{top+148}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{gain:.2f}</text>',
        ]
    p.append(f'<line x1="{left}" y1="{top+61}" x2="{left+pw}" y2="{top+61}" stroke="{INK}" stroke-width="3"/>')
    intervals = [(-0.21, -0.18, "stove lobe", BLUE), (-0.17, -0.16, "living gap", MUTED), (-0.15, -0.14, "fire-pit lobe", ORANGE)]
    for lo, hi, label_text, color in intervals:
        x, w = sx(lo), sx(hi) - sx(lo)
        p += [
            f'<rect x="{x:.1f}" y="{top+35}" width="{w:.1f}" height="52" rx="7" fill="{color}" opacity="0.86"/>',
            f'<text x="{x+w/2:.1f}" y="{top+22}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" font-weight="700" fill="{color}">{esc(label_text)}</text>',
        ]
    p.append(f'<text x="{left+pw/2}" y="{top+178}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">signed gain α</text>')
    labels = [("concept / negative", 5, BLUE), ("unrelated adapter", 4, ORANGE), ("random", 0, RED), ("coordinate shuffle", 0, RED), ("concept / positive", 0, RED)]
    bar_left, bar_top, bar_w = 265, 355, 650
    for i, (name, value, color) in enumerate(labels):
        y = bar_top + i * 39
        p += [
            f'<text x="{bar_left-16}" y="{y+5}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{INK}">{esc(name)}</text>',
            f'<rect x="{bar_left}" y="{y-10}" width="{bar_w}" height="20" rx="5" fill="#f1f3f7"/>',
            f'<rect x="{bar_left}" y="{y-10}" width="{bar_w*value/5:.1f}" height="20" rx="5" fill="{color}"/>',
            f'<text x="{bar_left+bar_w+12}" y="{y+5}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" font-weight="700" fill="{color}">{value}/5</text>',
        ]
    p.append(f'<text x="48" y="585" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: ontological-inversion/runs/2026-06-24_topology-of-the-flip.md and results/CONTROLS.md.</text>')
    save("ontological_gain_island.svg", p)


def figure_bias_residual() -> None:
    width, height = 980, 575
    p = svg_start(
        width,
        height,
        "The affine bias is sufficient; too much concept residual cancels it",
        "Number of inverting gains in the shipped adapter coordinates; one concept and prompt",
    )
    rows = [("bias only", 11), ("b + 0.25 Wv", 14), ("b + 0.5 Wv", 17), ("b + 1 Wv", 7), ("b + 2 Wv", 3), ("b + 4 Wv", 2), ("residual only", 0), ("wolf residual", 6)]
    left, top, pw = 220, 105, 660
    sx = lambda v: left + v / 18 * pw
    for tick in [0, 3, 6, 9, 12, 15, 18]:
        x = sx(tick)
        p += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+350}" stroke="{GRID}"/>', f'<text x="{x:.1f}" y="{top+375}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{tick}</text>']
    for i, (name, value) in enumerate(rows):
        y = top + 20 + i * 42
        color = GREEN if name == "bias only" else (RED if name == "residual only" else BLUE)
        p += [
            f'<text x="{left-15}" y="{y+5}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{INK}">{esc(name)}</text>',
            f'<rect x="{left}" y="{y-11}" width="{sx(value)-left:.1f}" height="22" rx="5" fill="{color}"/>',
            f'<text x="{sx(value)+10:.1f}" y="{y+5}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" font-weight="700" fill="{color}">{value}</text>',
        ]
    p += [
        f'<text x="{left+pw/2}" y="{top+410}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">inverting gains on the tested grid</text>',
        f'<rect x="90" y="510" width="800" height="40" rx="9" fill="#fff8e9" stroke="{ORANGE}"/>',
        f'<text x="490" y="535" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" font-weight="700" fill="{ORANGE}">cos(b, Wv) = −0.6056 · ‖Wv‖ = 1.2989 &gt; ‖b‖ = 1.1118</text>',
        f'<text x="48" y="566" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: ontological-inversion/runs/2026-08-25_bias-vs-residual.md.</text>',
    ]
    save("bias_residual_decomposition.svg", p)


def figure_cross_model_evidence() -> None:
    width, height = 1080, 590
    p = svg_start(
        width,
        height,
        "Qwen3→Llama crosses from static alignment into semantic transmission",
        "Span-aware ridge transmits memory-specific content",
    )
    rows = [
        ("pooled map", 0.238, "I find nothing / The Oxford comma", RED),
        ("text-first chunks", 0.615, "A worm is a fire hydrant", ORANGE),
        ("span-aware ridge", 0.647, "fire burning hamster", BLUE),
        ("same span slots, open readback", 0.647, "A wolf is a fire breathing dragon", BLUE),
        ("ordered Llama oracle", 0.70, "A worb glob is a fire breathing hamster", GREEN),
    ]
    left, top, pw = 305, 125, 680
    for i, (name, value, output, color) in enumerate(rows):
        y = top + i * 76
        shown = min(value, 0.70)
        p += [
            f'<text x="{left-18}" y="{y+5}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" font-weight="700" fill="{INK}">{esc(name)}</text>',
            f'<rect x="{left}" y="{y-14}" width="{pw}" height="28" rx="7" fill="#f1f3f7"/>',
            f'<rect x="{left}" y="{y-14}" width="{pw*shown/0.70:.1f}" height="28" rx="7" fill="{color}" opacity="0.88"/>',
            f'<text x="{left+15}" y="{y+5}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" font-weight="700" fill="#ffffff">{"ORACLE" if name.startswith("ordered") else f"centered {value:.3f}"}</text>',
            f'<text x="{left+15}" y="{y+35}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{INK}">“{esc(output)}”</text>',
        ]
    p += [
        f'<rect x="86" y="500" width="910" height="45" rx="10" fill="#eef8f5" stroke="{GREEN}"/>',
        f'<text x="541" y="528" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" font-weight="700" fill="{GREEN}">Across the two span-aware framings: fire + breathing + hamster all cross the model-family boundary.</text>',
        f'<text x="48" y="578" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: research_logs/2026-08-24_soft-slot-llama-spoke-the-buried-fact.md; synapse/probe/span_s11_fine.jsonl.</text>',
    ]
    save("cross_model_evidence.svg", p)


def axes(parts: list[str], left: float, top: float, pw: float, ph: float, xlabels: list[tuple[float, str]]) -> None:
    for tick in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = top + (1 - tick) * ph
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{tick:.1f}</text>')
    for x, label in xlabels:
        parts.append(f'<text x="{x:.1f}" y="{top+ph+24:.1f}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="10" fill="{MUTED}">{esc(label)}</text>')


def figure_rank_curves() -> None:
    rows = list(csv.DictReader((ROOT / "synapse" / "rank_curves.csv").open()))
    width, height = 1000, 610
    left, right, top, bottom = 82, 48, 105, 90
    pw, ph = width - left - right, height - top - bottom
    p = svg_start(
        width,
        height,
        "Retrieval similarity saturates before reconstruction precision",
        "Qwen3 leading-coordinate rank; one fixed 90/10 split of 32,000 span pairs",
    )
    xs = [math.log2(float(row["rank"])) for row in rows]
    xmin, xmax = min(xs), max(xs)
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * pw
    sy = lambda y: top + (1 - y) * ph
    axes(p, left, top, pw, ph, [(sx(x), row["rank"]) for x, row in zip(xs, rows)])
    metrics = [
        ("similarity_correlation", "pairwise-similarity correlation", BLUE),
        ("token_reconstruction_centered_cosine", "token reconstruction (centered cosine)", RED),
        ("proper_name_top1", "proper-name top-1 (n=52)", PURPLE),
        ("relation_reconstruction_centered_cosine", "relation-span reconstruction (n=203)", GREEN),
    ]
    for key, label, color in metrics:
        pts = " ".join(f"{sx(x):.1f},{sy(float(row[key])):.1f}" for x, row in zip(xs, rows))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, row in zip(xs, rows):
            p.append(f'<circle cx="{sx(x):.1f}" cy="{sy(float(row[key])):.1f}" r="3.5" fill="{color}"/>')
    x128 = sx(math.log2(128))
    p += [
        f'<line x1="{x128:.1f}" y1="{top}" x2="{x128:.1f}" y2="{top+ph}" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 5"/>',
        f'<text x="{x128+8:.1f}" y="{top+18}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" font-weight="700" fill="{ORANGE}">rank 128</text>',
        f'<text x="{x128+8:.1f}" y="{top+38}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{INK}">similarity 0.937</text>',
        f'<text x="{x128+8:.1f}" y="{top+57}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{INK}">reconstruction 0.346</text>',
    ]
    lx, ly = 585, 100
    for i, (_, label, color) in enumerate(metrics):
        y = ly + i * 23
        p += [
            f'<line x1="{lx}" y1="{y}" x2="{lx+25}" y2="{y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{lx+34}" y="{y+4}" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{INK}">{esc(label)}</text>',
        ]
    p += [
        f'<text x="{left+pw/2:.1f}" y="{height-35}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">retained rank (log₂ spacing)</text>',
        f'<text x="20" y="{top+ph/2:.1f}" transform="rotate(-90 20 {top+ph/2:.1f})" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">metric value</text>',
        f'<text x="48" y="590" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: synapse/rank_curves.csv (SHA-256 428d21c5…); 49,990 fixed cosine pairs. Relation linear-probe accuracy is intentionally omitted.</text>',
    ]
    save("rank_curves.svg", p)


def figure_subspaces() -> None:
    rows = list(csv.DictReader((ROOT / "synapse" / "rank_subspace.csv").open()))
    by_rank: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_rank[int(row["rank"])][row["subspace"]] = float(row["centered"])
    ranks = sorted(by_rank)
    width, height = 960, 570
    left, right, top, bottom = 90, 40, 105, 90
    pw, ph = width - left - right, height - top - bottom
    p = svg_start(
        width,
        height,
        "Matched-rank subspaces: PCA/SVD leads reconstruction",
        "Held-out centered cosine; same 32,000 span pairs and ridge convention",
    )
    sx = lambda i: left + i / (len(ranks) - 1) * pw
    sy = lambda y: top + (0.72 - y) / 0.72 * ph
    for tick in [0, 0.2, 0.4, 0.6]:
        y = sy(tick)
        p += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="{GRID}"/>',
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{tick:.1f}</text>',
        ]
    series = [("mrl", "leading Matryoshka", BLUE), ("svd", "top PCA/SVD", GREEN), ("rand", "seeded random", RED)]
    for key, label, color in series:
        pts = " ".join(f"{sx(i):.1f},{sy(by_rank[r][key]):.1f}" for i, r in enumerate(ranks))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, r in enumerate(ranks):
            p.append(f'<circle cx="{sx(i):.1f}" cy="{sy(by_rank[r][key]):.1f}" r="4" fill="{color}"/>')
    for i, rank in enumerate(ranks):
        p.append(f'<text x="{sx(i):.1f}" y="{top+ph+25}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{rank}</text>')
    for i, (_, label, color) in enumerate(series):
        y = 97 + i * 23
        p += [
            f'<line x1="640" y1="{y}" x2="665" y2="{y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="674" y="{y+4}" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{INK}">{esc(label)}</text>',
        ]
    p += [
        f'<text x="{left+pw/2:.1f}" y="{height-35}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">rank</text>',
        f'<text x="22" y="{top+ph/2:.1f}" transform="rotate(-90 22 {top+ph/2:.1f})" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">centered held-out cosine</text>',
        f'<text x="48" y="550" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: synapse/rank_subspace.csv (SHA-256 2d84fc25…); one seeded split; no confidence intervals.</text>',
    ]
    save("rank_subspaces.svg", p)


def figure_write_site_decay() -> None:
    width, height = 1120, 610
    p = svg_start(
        width,
        height,
        "Write lifetime depends on injection site",
        "One disclosed fact, gain 0.825; absolute direction (left) and plus/minus fold (right)",
    )
    offsets = [0, 1, 2, 3, 5]
    absolute = {
        "embedding": [1.0, 0.263, 0.249, 0.193, 0.124],
        "block 1": [1.0, 0.824, 0.561, 0.421, 0.187],
        "block 4": [1.0, 0.635, 0.426, 0.268, 0.127],
        "block 8": [1.0, 0.632, 0.413, 0.289, 0.165],
        "block 16": [1.0, 0.709, 0.515, 0.418, 0.288],
    }
    fold = {
        "block 1": [-1.0, -0.868, -0.561, -0.469, -0.326],
        "block 4": [-1.0, -0.896, -0.815, -0.752, -0.668],
        "block 8": [-1.0, -0.828, -0.695, -0.636, -0.463],
        "block 16": [-1.0, -0.823, -0.625, -0.567, -0.421],
    }
    colors = {
        "embedding": MUTED,
        "block 1": BLUE,
        "block 4": GREEN,
        "block 8": ORANGE,
        "block 16": PURPLE,
    }

    def panel(left: float, title: str, series: dict[str, list[float]], ymin: float, ymax: float, threshold: float) -> None:
        top, pw, ph = 125.0, 430.0, 330.0
        sx = lambda value: left + value / 5.0 * pw
        sy = lambda value: top + (ymax - value) / (ymax - ymin) * ph
        p.append(f'<text x="{left}" y="100" font-family="DejaVu Sans,Arial,sans-serif" font-size="15" font-weight="700" fill="{INK}">{esc(title)}</text>')
        for tick in [ymin, threshold, ymax]:
            y = sy(tick)
            dash = "5 4" if tick == threshold else "none"
            p.extend([
                f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-dasharray="{dash}"/>',
                f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{tick:.1f}</text>',
            ])
        for offset in offsets:
            x = sx(offset)
            p.extend([
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}" stroke="{GRID}" opacity="0.55"/>',
                f'<text x="{x:.1f}" y="{top+ph+24}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{offset}</text>',
            ])
        for label, values in series.items():
            pts = " ".join(f"{sx(offset):.1f},{sy(value):.1f}" for offset, value in zip(offsets, values))
            color = colors[label]
            p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
            for offset, value in zip(offsets, values):
                p.append(f'<circle cx="{sx(offset):.1f}" cy="{sy(value):.1f}" r="4" fill="{color}"/>')
        p.append(f'<text x="{left+pw/2:.1f}" y="{top+ph+52}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{INK}">blocks after write</text>')

    panel(85, "A · absolute survival cosine", absolute, 0.0, 1.0, 0.5)
    panel(625, "B · plus/minus fold cosine", fold, -1.0, 0.0, -0.5)
    for i, label in enumerate(["embedding", "block 1", "block 4", "block 8", "block 16"]):
        x = 180 + i * 160
        color = colors[label]
        p.extend([
            f'<line x1="{x}" y1="525" x2="{x+24}" y2="525" stroke="{color}" stroke-width="3"/>',
            f'<text x="{x+32}" y="529" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{INK}">{esc(label)}</text>',
        ])
    p.append(f'<text x="48" y="590" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: synapse/layer_trace/midstack_trace.csv; run card 2026-08-27_llama-midstack-injection-decay.md. Embedding-site fold omitted because the sign flips and the instrument is invalid there.</text>')
    save("write_site_decay.svg", p)


def norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip() + " "


def figure_e7() -> None:
    spec = json.loads((ROOT / "synapse" / "probe" / "e7_questions.json").read_text())
    patterns = [value.lower() for value in spec["decline_patterns"]]

    def declined(text: str) -> bool:
        value = norm(text)
        return any(f" {pattern} " in value or value.strip().startswith(pattern) or pattern in value for pattern in patterns)

    groups: dict[tuple[str, bool, str], list[bool]] = defaultdict(list)
    with (ROOT / "synapse" / "probe" / "e7_raw.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            groups[(row["disclosure"], row["present"], row["kind"])].append(declined(row["out"]))
    rate = lambda key: sum(groups[key]) / len(groups[key])
    rows = [
        ("neutral / memory", rate(("off", True, "answerable")), rate(("off", True, "unanswerable"))),
        ("neutral / blank", rate(("off", False, "answerable")), rate(("off", False, "unanswerable"))),
        ("anti-guess / memory", rate(("on", True, "answerable")), rate(("on", True, "unanswerable"))),
        ("anti-guess / blank", rate(("on", False, "answerable")), rate(("on", False, "unanswerable"))),
    ]
    width, height = 1050, 600
    p = svg_start(
        width,
        height,
        "Bounded-knowledge criterion failed; answer propensity shifted broadly",
        "Decline fraction, n=10 memory clusters; 6,400 nested generations",
    )
    left, top, pw, ph = 265, 112, 700, 355
    sx = lambda x: left + x / 0.5 * pw
    for tick in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        x = sx(tick)
        p += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}" stroke="{GRID}"/>',
            f'<text x="{x:.1f}" y="{top+ph+25}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{tick:.1f}</text>',
        ]
    for i, (label, answerable, unanswerable) in enumerate(rows):
        y = top + 48 + i * 78
        p += [
            f'<text x="{left-18}" y="{y+4}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">{esc(label)}</text>',
            f'<line x1="{sx(answerable):.1f}" y1="{y}" x2="{sx(unanswerable):.1f}" y2="{y}" stroke="{MUTED}" stroke-width="2"/>',
            f'<circle cx="{sx(answerable):.1f}" cy="{y}" r="7" fill="{BLUE}"/>',
            f'<circle cx="{sx(unanswerable):.1f}" cy="{y}" r="7" fill="{RED}"/>',
            f'<text x="{sx(answerable):.1f}" y="{y-14}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{BLUE}">{answerable:.3f}</text>',
            f'<text x="{sx(unanswerable):.1f}" y="{y-14}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{RED}">{unanswerable:.3f}</text>',
        ]
    p += [
        f'<circle cx="710" cy="91" r="6" fill="{BLUE}"/><text x="723" y="95" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{INK}">answerable</text>',
        f'<circle cx="815" cy="91" r="6" fill="{RED}"/><text x="828" y="95" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{INK}">unanswerable</text>',
        f'<rect x="65" y="492" width="920" height="68" rx="10" fill="#fff7e8" stroke="{ORANGE}"/>',
        f'<text x="86" y="518" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" font-weight="700" fill="{ORANGE}">Primary neutral-memory difference: +0.126; 95% cluster-bootstrap CI [+0.080, +0.177].</text>',
        f'<text x="86" y="543" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" fill="{INK}">The entire interval is below the preregistered 0.25 criterion. Bounded knowledge is withdrawn.</text>',
        f'<text x="48" y="588" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: synapse/probe/e7_raw.jsonl (SHA-256 77caf18d…); decision text in preregistrations/2026-08-25_E7_base_rate.md.</text>',
    ]
    save("e7_declines.svg", p)


def figure_e10() -> None:
    summary = json.loads((ROOT / "synapse" / "probe" / "e10_summary.json").read_text())
    arms = [
        ("aligned adapter", "adapter_q8"),
        ("reversed slots", "reversed_q8"),
        ("wrong memory", "cross_memory_q8"),
        ("random direction", "random"),
        ("dimension shuffle", "dimshuffle"),
        ("oracle", "oracle"),
    ]
    width, height = 1020, 590
    p = svg_start(
        width,
        height,
        "A narrow corpus-clean boundary for the fixed ridge bridge",
        "E10 does not erase earlier Qwen3→Llama semantic transmission; one greedy decode per cell",
    )
    left, top, pw, ph = 260, 118, 670, 350
    sx = lambda value: left + value / 9 * pw
    for tick in range(10):
        x = sx(tick)
        p += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}" stroke="{GRID}"/>',
            f'<text x="{x:.1f}" y="{top+ph+25}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">{tick}</text>',
        ]
    for i, (label, key) in enumerate(arms):
        gain = "0.825"
        cell = summary["arms"][key]["by_gain"][gain]
        definitions = int(cell["definition"]["success"])
        inference = int(cell["inference"]["success"])
        combined = definitions + inference
        y = top + 32 + i * 57
        p += [
            f'<text x="{left-18}" y="{y+5}" text-anchor="end" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">{esc(label)}</text>',
            f'<line x1="{left}" y1="{y}" x2="{sx(combined):.1f}" y2="{y}" stroke="{BLUE if key == "oracle" else MUTED}" stroke-width="5"/>',
            f'<circle cx="{sx(combined):.1f}" cy="{y}" r="8" fill="{BLUE if key == "oracle" else RED}"/>',
            f'<text x="{sx(combined)+16:.1f}" y="{y+5}" font-family="DejaVu Sans,Arial,sans-serif" font-size="12" font-weight="700" fill="{INK}">{combined}/9</text>',
        ]
        if key == "oracle":
            p.append(f'<text x="{sx(combined)+60:.1f}" y="{y+5}" font-family="DejaVu Sans,Arial,sans-serif" font-size="11" fill="{MUTED}">2 definitions + 1 inference</text>')
    p += [
        f'<text x="{left+pw/2:.1f}" y="{height-65}" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="{INK}">memory-specific successes (of 9) at gain 0.825</text>',
        f'<text x="48" y="550" font-family="DejaVu Sans,Arial,sans-serif" font-size="11.5" fill="{INK}">Aligned adapter scored 0/9 at gains 0.75, 0.825, and 0.90; this bounds clean-nonce conjunctive recall for this protocol.</text>',
        f'<text x="48" y="575" font-family="DejaVu Sans,Arial,sans-serif" font-size="10.5" fill="{MUTED}">Source: synapse/probe/e10_summary.json (SHA-256 6d34b951…); frozen rule in preregistrations/2026-08-26_E10_verified_novel_adapter_join.md.</text>',
    ]
    save("e10_adapter_join.svg", p)


def main() -> None:
    figure_system()
    figure_two_write_regimes()
    figure_original_benchmark()
    figure_ontological_gain_island()
    figure_bias_residual()
    figure_cross_model_evidence()
    figure_rank_curves()
    figure_subspaces()
    figure_write_site_decay()
    figure_e7()
    figure_e10()
    print(f"wrote eleven SVG figures under {OUT}")


if __name__ == "__main__":
    main()
