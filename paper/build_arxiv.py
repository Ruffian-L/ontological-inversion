#!/usr/bin/env python3
"""Build an article-class preprint PDF with Tectonic from MANUSCRIPT.md.

MANUSCRIPT.md remains the source. This writer exists so the review PDF is a
Computer Modern article rather than a ReportLab letter.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from reportlab.graphics import renderPDF

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
FIGURES = PAPER / "figures"
TEX = PAPER / "arxiv.tex"
PDF_OUT = PAPER / "Writing-Meaning-Between-Frozen-Models.pdf"

sys.path.insert(0, str(PAPER))
from render_manuscript import make_figures  # noqa: E402


USED_FIGURES = {
    "two_write_regimes.svg",
    "original_benchmark.svg",
    "ontological_gain_island.svg",
    "cross_model_evidence.svg",
    "rank_curves.svg",
    "rank_subspaces.svg",
    "write_site_decay.svg",
}


UNI = {
    "“": "<<LQ>>",
    "”": "<<RQ>>",
    "‘": "<<LS>>",
    "’": "<<RS>>",
    "–": "<<ND>>",
    "—": "<<MD>>",
    "∈": "<<IN>>",
    "μ": "<<MU>>",
    "λ": "<<LA>>",
    "Δ": "<<DE>>",
    "×": "<<TI>>",
    "·": "<<DT>>",
    "−": "<<MI>>",
}
UNI_LATEX = {
    "<<LQ>>": "``",
    "<<RQ>>": "''",
    "<<LS>>": "`",
    "<<RS>>": "'",
    "<<ND>>": "--",
    "<<MD>>": "---",
    "<<IN>>": r"$\in$",
    "<<MU>>": r"$\mu$",
    "<<LA>>": r"$\lambda$",
    "<<DE>>": r"$\Delta$",
    "<<TI>>": r"$\times$",
    "<<DT>>": r"$\cdot$",
    "<<MI>>": r"$-$",
}


def latex_escape(text: str) -> str:
    for src, tok in UNI.items():
        text = text.replace(src, tok)
    text = text.replace("\\", r"\textbackslash{}")
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    text = "".join(out)
    for tok, tex in UNI_LATEX.items():
        text = text.replace(tok, tex)
    return text


def convert_math(block: str) -> str:
    s = block.strip()
    s = s.replace("||", r"\|")
    s = s.replace("×", r"\times ")
    s = s.replace("·", r"\cdot ")
    s = s.replace("λ", r"\lambda ")
    s = s.replace("μ", r"\mu ")
    s = s.replace("∈", r"\in ")
    return s


def inline(text: str) -> str:
    def cite(match: re.Match[str]) -> str:
        keys = [part.strip().lstrip("@") for part in match.group(1).split(";")]
        return r"\citep{" + ",".join(keys) + "}"

    text = re.sub(r"\[(@[^\]]+)\]", cite, text)
    def code_cmd(match: re.Match[str]) -> str:
        s = match.group(1)
        if "/" in s:
            return r"\url{" + s.replace("%", r"\%") + "}"
        return r"\texttt{" + latex_escape(s) + "}"

    text = re.sub(r"`([^`]+)`", code_cmd, text)
    text = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: r"\textbf{" + latex_escape(m.group(1)) + "}",
        text,
    )
    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        lambda m: r"\emph{" + latex_escape(m.group(1)) + "}",
        text,
    )
    # leftover prose
    parts = re.split(r"(\\(?:citep|texttt|textbf|emph|url)\{[^}]*\})", text)
    rebuilt = []
    for part in parts:
        if part.startswith("\\"):
            rebuilt.append(part)
        else:
            rebuilt.append(latex_escape(part))
    return "".join(rebuilt)


def parse_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and re.match(r"^:?-+:?$", cells[0].replace(" ", "")):
            continue
        rows.append(cells)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    longest = max((len(c) for r in rows for c in r), default=0)
    if longest > 42:
        widths = {2: ("0.22", "0.30", "0.40"), 3: ("0.18", "0.32", "0.42")}
        w = widths.get(ncols)
        if w:
            colspec = "".join(f"p{{{x}\\linewidth}}" for x in w[:ncols])
        else:
            colspec = "p{0.18\\linewidth}" * ncols
    else:
        colspec = "l" + "r" * (ncols - 1)
    out = ["\\begin{center}", "\\small", r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    for i, row in enumerate(rows):
        row = row + [""] * (ncols - len(row))
        out.append(" & ".join(inline(c) for c in row) + r" \\")
        if i == 0:
            out.append(r"\midrule")
    out.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", ""])
    return "\n".join(out)


def md_to_body(source: Path) -> tuple[str, str]:
    lines = source.read_text().splitlines()
    abstract_lines: list[str] = []
    body: list[str] = []
    i = 0
    mode = "skip_front"
    in_code = False
    code: list[str] = []
    in_quote = False
    quote: list[str] = []
    saw_appendix = False
    skip_refs = False

    def flush_quote():
        nonlocal in_quote, quote
        if quote:
            body.append(r"\begin{quote}\small " + inline(" ".join(quote)) + r"\end{quote}")
            body.append("")
        quote = []
        in_quote = False

    while i < len(lines):
        line = lines[i]
        if mode == "skip_front":
            if line.startswith("## Abstract"):
                mode = "abstract"
            i += 1
            continue
        if mode == "abstract":
            if line.startswith("## "):
                mode = "body"
                continue
            if line.strip():
                abstract_lines.append(line.strip())
            i += 1
            continue
        if skip_refs:
            if line.startswith("## Appendix"):
                skip_refs = False
            else:
                i += 1
                continue
        if line.startswith("```"):
            flush_quote()
            if in_code:
                math = convert_math("\n".join(code))
                body.append(r"\begin{equation*}")
                body.append(math)
                body.append(r"\end{equation*}")
                body.append("")
                code = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        image = re.fullmatch(r"!\[([^]]+)\]\(figures/([^)]+)\)", line.strip())
        if image:
            flush_quote()
            caption = image.group(1)
            caption = re.sub(r"^Figure\s+\d+\.\s*", "", caption)
            fname = image.group(2)
            stem = Path(fname).stem
            pdf = fname if fname.endswith(".pdf") else f"{stem}.pdf"
            if not (FIGURES / pdf).exists():
                pdf = fname
            body.append(r"\begin{figure}[t]")
            body.append(r"\centering")
            body.append(rf"\includegraphics[width=0.92\linewidth]{{figures/{pdf}}}")
            body.append(rf"\caption{{{inline(caption)}}}")
            body.append(r"\end{figure}")
            body.append("")
            i += 1
            continue
        if line.startswith("|"):
            flush_quote()
            table = []
            while i < len(lines) and lines[i].startswith("|"):
                table.append(lines[i])
                i += 1
            body.append(parse_table(table))
            continue
        if line.startswith("## References"):
            flush_quote()
            skip_refs = True
            i += 1
            continue
        if line.startswith("## Appendix"):
            flush_quote()
            if not saw_appendix:
                body.append(r"\appendix")
                saw_appendix = True
            title = re.sub(r"^## Appendix\s+[A-Z]\.\s*", "", line)
            body.append(r"\section{" + inline(title) + "}")
            body.append("")
            i += 1
            continue
        if line.startswith("## "):
            flush_quote()
            title = re.sub(r"^\d+\.\s*", "", line[3:])
            body.append(r"\section{" + inline(title) + "}")
            body.append("")
            i += 1
            continue
        if line.startswith("### "):
            flush_quote()
            title = re.sub(r"^\d+\.\d+\s*", "", line[4:])
            body.append(r"\subsection{" + inline(title) + "}")
            body.append("")
            i += 1
            continue
        if line.startswith("> "):
            in_quote = True
            quote.append(line[2:].strip())
            i += 1
            continue
        if in_quote and not line.strip():
            flush_quote()
            i += 1
            continue
        if re.match(r"^[-*] ", line):
            flush_quote()
            body.append(r"\begin{itemize}")
            while i < len(lines) and re.match(r"^[-*] ", lines[i]):
                item = inline(lines[i][2:])
                i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    item += " " + inline(lines[i].strip())
                    i += 1
                body.append(r"\item " + item)
            body.append(r"\end{itemize}")
            body.append("")
            continue
        if re.match(r"^\d+\. ", line):
            flush_quote()
            body.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                item = inline(re.sub(r"^\d+\.\s*", "", lines[i]))
                i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    item += " " + inline(lines[i].strip())
                    i += 1
                body.append(r"\item " + item)
            body.append(r"\end{enumerate}")
            body.append("")
            continue
        if not line.strip():
            flush_quote()
            body.append("")
            i += 1
            continue
        if re.match(r"^\*\*\d+\.\d+\.\d+", line):
            flush_quote()
            title = re.sub(r"^\*\*|\*\*\.?$", "", line.strip())
            body.append(r"\subsubsection*{" + inline(title) + "}")
            body.append("")
            i += 1
            continue
        body.append(inline(line))
        i += 1
    flush_quote()
    abstract = inline(" ".join(abstract_lines))
    return abstract, "\n".join(body)


PREAMBLE = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{setspace}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}
\setstretch{1.12}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{1.15em}
"""


def write_tex(abstract: str, body: str) -> None:
    tex = PREAMBLE + r"""
\title{\textbf{Writing Meaning Between Frozen Models:}\\[0.35em]
{\large Cross-Model Vector Memory for Steering, Recall, and Reasoning}}
\author{Jason Van Pham\\[0.4em]
{\normalsize Independent researcher}}
\date{Preprint --- 27 August 2026}

\begin{document}
\maketitle
\begin{abstract}
""" + abstract + r"""
\end{abstract}

""" + body + r"""

\section*{Acknowledgements}
Gemini, Grok, ChatGPT, and Claude contributed as AI research collaborators
on experiments, logging, and drafting. The author is responsible for the
claims.

\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
"""
    TEX.write_text(tex)


PLATE_STEMS = {
    "two_write_regimes",
    "ontological_inversion_plate",
    "write_site_plate",
}


def export_figures() -> None:
    import plot_publication_figures as plots
    plots.main()


def compile_tex() -> None:
    cmd = [
        "tectonic",
        "-c", "minimal",
        "--keep-logs",
        "--keep-intermediates",
        str(TEX),
    ]
    print("running", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        log = PAPER / "arxiv.log"
        if log.exists():
            print(log.read_text()[-4000:])
        raise SystemExit(proc.returncode)
    built = PAPER / "arxiv.pdf"
    if not built.exists():
        raise SystemExit("tectonic did not write arxiv.pdf")
    PDF_OUT.write_bytes(built.read_bytes())
    print(f"wrote {PDF_OUT}")


def main() -> int:
    export_figures()
    abstract, body = md_to_body(PAPER / "MANUSCRIPT.md")
    write_tex(abstract, body)
    compile_tex()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
