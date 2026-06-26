#!/usr/bin/env python3
"""Build a diverse, multi-genre, INERT corpus for the BOS-hijack sweep.

Genres: code | math | colors | science | memoir | everyday.
Every source is plain text / data. Nothing here is executed, rendered, compiled,
pickled, or trust_remote_code'd. We download bytes, decode utf-8, split on lines/
sentences, and write text. That's it. (STANDARDS: trust the bytes.)

Output:
  data/corpus.tsv   genre<TAB>text   (one phrase per line)
  data/corpus.txt   text only
  data/corpus_manifest.json   source + count + sha256 per genre

Deterministic: fixed seed, no Date/random-from-clock.
"""
import os, re, sys, json, html, random, hashlib, urllib.request, urllib.error
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this repo (portable)
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
random.seed(1729)  # Hardy-Ramanujan, fixed; not clock-derived

UA = {"User-Agent": "corpus-builder/1.0 (inert text fetch)"}
def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def clean(s):
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ok(s, lo=8, hi=220):
    if not (lo <= len(s) <= hi):
        return False
    letters = sum(c.isalpha() for c in s)
    return letters >= max(3, int(0.35 * len(s)))  # not pure punctuation/digits

def sents(text):
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?]) ", text)

buckets = defaultdict(list)
manifest = {}

# ---------------------------------------------------------------- 1. CODE (this repo only)
# Sample real, meaningful single lines from THIS repo's own source. No external scrape.
def gather_code():
    out, roots = [], [ROOT]
    exts = (".py", ".rs", ".js", ".ts", ".c", ".cpp", ".go", ".sh")
    files = []
    for base in roots:
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__", ".venv", "target", "data")]
            for f in fn:
                if f.endswith(exts):
                    files.append(os.path.join(dp, f))
    random.shuffle(files)
    for path in files:
        try:
            with open(path, "r", errors="ignore") as fh:
                lines = fh.read().splitlines()
        except Exception:
            continue
        cand = []
        for ln in lines:
            t = ln.strip()
            if not (12 <= len(t) <= 140):
                continue
            if t.startswith(("#", "//", "*", "/*", '"', "'", "import ", "from ")):
                continue
            if any(k in t for k in ("=", "(", "fn ", "def ", "return", "->", "let ", "for ", "if ", ".")):
                cand.append(t)
        random.shuffle(cand)
        out.extend(cand[:40])  # take more per file — one small repo is the whole source now
        if len(out) >= 3200:
            break
    return out[:3200]

# ---------------------------------------------------------------- 2. MATH (real, templated + seed)
MATH_SEEDS = [
    "The derivative of sin(x) is cos(x).",
    "The integral of 1/x dx is the natural log of the absolute value of x.",
    "Euler's identity states that e raised to i pi plus one equals zero.",
    "A function is an involution when applying it twice returns the original input.",
    "The eigenvalues of a Hermitian matrix are always real.",
    "A reflection is its own inverse, so f composed with f is the identity.",
    "The gradient points in the direction of steepest ascent.",
    "Two vectors are orthogonal when their dot product is zero.",
    "The trace of a matrix equals the sum of its eigenvalues.",
    "A Householder transformation reflects a vector across a hyperplane.",
    "The cosine of the angle between two unit vectors is their inner product.",
    "Cantor's theorem shows the reals are uncountable.",
    "The rank-nullity theorem relates the dimensions of image and kernel.",
    "Bayes' rule updates a prior into a posterior using the likelihood.",
    "The softmax function maps a vector of logits to a probability distribution.",
    "Gradient descent minimizes a loss by stepping against the gradient.",
    "The L2 norm of a vector is the square root of the sum of its squared entries.",
    "A symmetric matrix is diagonalizable by an orthogonal basis of eigenvectors.",
    "The chain rule differentiates a composition of functions.",
    "Cosine similarity is invariant to the magnitude of the vectors.",
    "A projection matrix is idempotent: P squared equals P.",
    "The determinant of an orthogonal matrix is plus or minus one.",
    "Entropy measures the average surprise of a distribution.",
    "The dot product of a vector with itself is its squared length.",
    "A basis spans the space and is linearly independent.",
]
def gather_math():
    out = list(MATH_SEEDS)
    for a in range(2, 40):
        for b in range(2, 40):
            out.append(f"{a} times {b} equals {a*b}.")
    for n in range(2, 60):
        out.append(f"The square root of {n*n} is {n}.")
        out.append(f"{n} squared is {n*n}.")
    for n in range(1, 40):
        out.append(f"The sum of the first {n} positive integers is {n*(n+1)//2}.")
    latex = [
        r"\int_0^\infty e^{-x}\,dx = 1",
        r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}",
        r"\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)",
        r"\langle u, v \rangle = \|u\|\,\|v\|\cos\theta",
        r"\Phi_c(h) = \mu + (I - 2P_c)(h - \mu)",
        r"f(f(x)) = x",
        r"P = \hat{d}\hat{d}^{\top}, \quad P^2 = P",
        r"\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}",
        r"H(p) = -\sum_i p_i \log p_i",
        r"e^{i\pi} + 1 = 0",
    ]
    out.extend(latex * 3)
    random.shuffle(out)
    return out

# ---------------------------------------------------------------- 3. COLORS (xkcd rgb list)
def gather_colors():
    raw = fetch("https://xkcd.com/color/rgb.txt").decode("utf-8", "ignore")
    names = []
    for line in raw.splitlines():
        if "\t" not in line or line.startswith("License"):
            continue
        name, *rest = line.split("\t")
        name = name.strip()
        hexc = rest[0].strip() if rest else ""
        if name:
            names.append((name, hexc))
    tmpl = [
        "a shade of {n}",
        "the color {n}",
        "walls painted {n}",
        "a {n} sky at dusk",
        "{n} ({h})",
        "somewhere between {n} and slate",
    ]
    out = []
    for name, hexc in names:
        for t in tmpl:
            out.append(t.format(n=name, h=hexc))
    random.shuffle(out)
    return out[:1600]

# ---------------------------------------------------------------- 4. SCIENCE (arXiv API, Atom XML)
def gather_science():
    cats = ["cs.CL", "cs.LG", "stat.ML", "physics.optics", "q-bio.NC", "math.RA", "astro-ph.GA"]
    out = []
    for c in cats:
        try:
            url = (f"https://export.arxiv.org/api/query?search_query=cat:{c}"
                   f"&start=0&max_results=120&sortBy=submittedDate&sortOrder=descending")
            xml = fetch(url, timeout=40).decode("utf-8", "ignore")
        except Exception as e:
            print(f"  science[{c}] fetch failed: {e}", file=sys.stderr); continue
        for m in re.finditer(r"<summary>(.*?)</summary>", xml, re.S):
            for s in sents(clean(m.group(1))):
                s = s.strip()
                if ok(s, 25, 200):
                    out.append(s)
        for m in re.finditer(r"<title>(.*?)</title>", xml, re.S):
            t = clean(m.group(1))
            if ok(t, 20, 200) and "arXiv" not in t:
                out.append(t)
    random.shuffle(out)
    return out[:2800]

# ---------------------------------------------------------------- 5. MEMOIR (Gutenberg public domain)
GUTEN = {
    "franklin_autobiography": 20203,
    "douglass_narrative": 23,
    "helen_keller_story_of_my_life": 2397,
    "thoreau_walden": 205,
}
def strip_guten(text):
    a = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    b = re.search(r"\*\*\* END OF", text, re.S)
    if a: text = text[a.end():]
    if b: text = text[:b.start()]
    return text
def gather_memoir():
    out = []
    for name, gid in GUTEN.items():
        body = None
        for url in (f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                    f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"):
            try:
                body = fetch(url, timeout=50).decode("utf-8", "ignore"); break
            except Exception:
                continue
        if not body:
            print(f"  memoir[{name}] fetch failed", file=sys.stderr); continue
        for s in sents(clean(strip_guten(body))):
            s = s.strip()
            if ok(s, 30, 200) and not s.isupper():
                out.append(s)
    random.shuffle(out)
    return out[:3200]

# ---------------------------------------------------------------- 6. EVERYDAY (Tatoeba English)
def gather_everyday():
    import bz2
    try:
        raw = fetch("https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2", timeout=120)
        text = bz2.decompress(raw).decode("utf-8", "ignore")
    except Exception as e:
        print(f"  everyday(tatoeba) failed: {e}", file=sys.stderr); return []
    out = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            s = clean(parts[2])
            if ok(s, 12, 160):
                out.append(s)
    random.shuffle(out)
    return out[:6500]

# ---------------------------------------------------------------- assemble
GATHER = [
    ("code", gather_code), ("math", gather_math), ("colors", gather_colors),
    ("science", gather_science), ("memoir", gather_memoir), ("everyday", gather_everyday),
]
for genre, fn in GATHER:
    try:
        items = fn()
    except Exception as e:
        print(f"[{genre}] FAILED: {e}", file=sys.stderr); items = []
    seen, uniq = set(), []
    for s in items:
        s = clean(s)
        if ok(s) and s.lower() not in seen:
            seen.add(s.lower()); uniq.append(s)
    buckets[genre] = uniq
    blob = "\n".join(uniq).encode()
    manifest[genre] = {"count": len(uniq), "sha256": hashlib.sha256(blob).hexdigest()[:16]}
    print(f"[{genre:9}] {len(uniq):6d} phrases  sha={manifest[genre]['sha256']}")

# global dedupe + interleave, write
rows = []
glob_seen = set()
for genre, lst in buckets.items():
    for s in lst:
        k = s.lower()
        if k not in glob_seen:
            glob_seen.add(k); rows.append((genre, s))
random.shuffle(rows)

with open(os.path.join(DATA, "corpus.tsv"), "w") as f:
    for g, s in rows:
        f.write(f"{g}\t{s}\n")
with open(os.path.join(DATA, "corpus.txt"), "w") as f:
    for _, s in rows:
        f.write(s + "\n")
manifest["_total"] = len(rows)
manifest["_by_genre_final"] = {g: sum(1 for gg, _ in rows if gg == g) for g in buckets}
with open(os.path.join(DATA, "corpus_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nTOTAL {len(rows)} phrases -> data/corpus.tsv  /  data/corpus.txt")
print("by genre:", manifest["_by_genre_final"])
