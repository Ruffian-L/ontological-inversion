#!/usr/bin/env python3
"""Build a diverse, multi-genre, INERT corpus for the ontological-inversion sweep.

Genres: code | math | colors | science | memoir | everyday.
Every source is plain text / data. Nothing here is executed, rendered, compiled,
pickled, or trust_remote_code'd. We download bytes, decode utf-8, split on lines/
sentences, and write text. That's it. (STANDARDS: trust the bytes.)

Output:
  data/corpus.tsv   genre<TAB>text   (one phrase per line)
  data/corpus.txt   text only
  data/corpus_manifest.json   source + count + sha256 per genre

Parallel: the 6 genres run concurrently, and the multi-download genres (science,
memoir) fetch their URLs in parallel too. Each genre uses its OWN seeded RNG, so
the output is byte-for-byte deterministic regardless of how the threads interleave.

Smoke test (fast, tiny, proves the pipeline end to end):
  python sweep/build_corpus.py --smoke
Full build (run this in your own terminal):
  python sweep/build_corpus.py --workers 12
"""
import os, re, sys, json, html, time, random, hashlib, argparse
import urllib.request, urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this repo (portable)
SEED = 1729  # Hardy-Ramanujan, fixed; not clock-derived


def parse_args():
    ap = argparse.ArgumentParser(description="Build the multi-genre inert corpus (parallel).")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--workers", type=int, default=8, help="parallel genre/fetch workers")
    ap.add_argument("--cap", type=int, default=None, help="max phrases per genre (overrides built-in caps)")
    ap.add_argument("--genres", default="code,math,colors,science,memoir,everyday",
                    help="comma-list subset of genres to build")
    ap.add_argument("--smoke", action="store_true",
                    help="fast tiny run: 1 sci category, 1 memoir book, skip the big Tatoeba dump, small caps")
    return ap.parse_args()


A = parse_args()
A.out_dir = os.path.join(ROOT, os.path.basename(A.out_dir))  # confine to repo dir (no path traversal)
os.makedirs(A.out_dir, exist_ok=True)
SMOKE = A.smoke
WORKERS = max(1, A.workers)
CAP = A.cap if A.cap is not None else (60 if SMOKE else None)
ARXIV_DELAY = 3.0  # seconds between arXiv calls — they return HTTP 429 on parallel/burst access


def rng_for(tag):
    """Independent, deterministic RNG per genre — safe to use across parallel threads."""
    return random.Random(f"{SEED}:{tag}")


def cap(lst):
    return lst[:CAP] if CAP else lst


UA = {"User-Agent": "corpus-builder/1.0 (inert text fetch)"}
def fetch(url, timeout=60, retries=2):
    """GET bytes with a small retry/backoff so a transient drop doesn't kill a whole genre."""
    last = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            if i < retries:
                time.sleep(1.5 * (i + 1))
    raise last

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


# ---------------------------------------------------------------- 1. CODE (this repo only)
# Sample real, meaningful single lines from THIS repo's own source. No external scrape.
def gather_code():
    rng = rng_for("code")
    out, roots = [], [ROOT]
    exts = (".py", ".rs", ".js", ".ts", ".c", ".cpp", ".go", ".sh")
    files = []
    for base in roots:
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__", ".venv", "target", "data")]
            for f in fn:
                if f.endswith(exts):
                    files.append(os.path.join(dp, f))
    files.sort()           # stable order before the seeded shuffle (deterministic)
    rng.shuffle(files)
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
        rng.shuffle(cand)
        out.extend(cand[:40])  # take more per file — one small repo is the whole source now
    return cap(out[:3200])

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
    rng = rng_for("math")
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
    rng.shuffle(out)
    return cap(out)

# ---------------------------------------------------------------- 3. COLORS (xkcd rgb list)
def gather_colors():
    rng = rng_for("colors")
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
    rng.shuffle(out)
    return cap(out[:1600])

# ---------------------------------------------------------------- 4. SCIENCE (arXiv API, Atom XML)
def gather_science():
    rng = rng_for("science")
    cats = ["cs.CL", "cs.LG", "stat.ML", "physics.optics", "q-bio.NC", "math.RA", "astro-ph.GA"]
    if SMOKE:
        cats = cats[:1]
    maxr = 5 if SMOKE else 120

    # arXiv rate-limits hard: parallel category fetches trip HTTP 429. Fetch SERIALLY with a
    # polite delay (their guidance is ~3s between calls). This genre still runs concurrently
    # with the others, so serializing here costs ~no wall-clock (memoir/everyday dominate).
    out = []
    for i, c in enumerate(cats):
        if i:
            time.sleep(ARXIV_DELAY)
        try:
            url = (f"https://export.arxiv.org/api/query?search_query=cat:{c}"
                   f"&start=0&max_results={maxr}&sortBy=submittedDate&sortOrder=descending")
            xml = fetch(url, timeout=60).decode("utf-8", "ignore")
        except Exception as e:
            print(f"  science[{c}] fetch failed: {e}", file=sys.stderr)
            continue
        for m in re.finditer(r"<summary>(.*?)</summary>", xml, re.S):
            for s in sents(clean(m.group(1))):
                s = s.strip()
                if ok(s, 25, 200):
                    out.append(s)
        for m in re.finditer(r"<title>(.*?)</title>", xml, re.S):
            t = clean(m.group(1))
            if ok(t, 20, 200) and "arXiv" not in t:
                out.append(t)
    rng.shuffle(out)
    return cap(out[:2800])

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
    rng = rng_for("memoir")
    items = list(GUTEN.items())
    if SMOKE:
        items = items[:1]

    def fetch_book(kv):
        name, gid = kv
        for url in (f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
                    f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"):
            try:
                return fetch(url, timeout=50).decode("utf-8", "ignore")
            except Exception:
                continue
        print(f"  memoir[{name}] fetch failed", file=sys.stderr)
        return ""

    out = []
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(items))) as ex:
        for body in ex.map(fetch_book, items):   # parallel fetch, deterministic order
            if not body:
                continue
            for s in sents(clean(strip_guten(body))):
                s = s.strip()
                if ok(s, 30, 200) and not s.isupper():
                    out.append(s)
    rng.shuffle(out)
    return cap(out[:3200])

# ---------------------------------------------------------------- 6. EVERYDAY (Tatoeba English)
def gather_everyday():
    if SMOKE:
        print("  everyday(tatoeba): skipped in --smoke (large bz2 download)", file=sys.stderr)
        return []
    rng = rng_for("everyday")
    import bz2
    try:
        raw = fetch("https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2", timeout=120)
        text = bz2.decompress(raw).decode("utf-8", "ignore")
    except Exception as e:
        print(f"  everyday(tatoeba) failed: {e}", file=sys.stderr)
        return []
    out = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            s = clean(parts[2])
            if ok(s, 12, 160):
                out.append(s)
    rng.shuffle(out)
    return cap(out[:6500])

# ---------------------------------------------------------------- assemble (parallel)
GATHER = [
    ("code", gather_code), ("math", gather_math), ("colors", gather_colors),
    ("science", gather_science), ("memoir", gather_memoir), ("everyday", gather_everyday),
]
want = [g.strip() for g in A.genres.split(",") if g.strip()]
selected = [(g, fn) for g, fn in GATHER if g in want]

print(f"building {len(selected)} genres with {WORKERS} workers"
      f"{' [SMOKE]' if SMOKE else ''} -> {A.out_dir}", file=sys.stderr)
t0 = time.perf_counter()

raw_results = {}
with ThreadPoolExecutor(max_workers=min(WORKERS, max(1, len(selected)))) as ex:
    futs = {ex.submit(fn): genre for genre, fn in selected}
    for fut in as_completed(futs):
        genre = futs[fut]
        try:
            raw_results[genre] = fut.result()
        except Exception as e:
            print(f"[{genre}] FAILED: {e}", file=sys.stderr)
            raw_results[genre] = []

# process genres in a FIXED order (not completion order) so output stays deterministic
buckets, manifest = defaultdict(list), {}
for genre, _ in selected:
    seen, uniq = set(), []
    for s in raw_results.get(genre, []):
        s = clean(s)
        if ok(s) and s.lower() not in seen:
            seen.add(s.lower()); uniq.append(s)
    buckets[genre] = uniq
    blob = "\n".join(uniq).encode()
    manifest[genre] = {"count": len(uniq), "sha256": hashlib.sha256(blob).hexdigest()[:16]}
    print(f"[{genre:9}] {len(uniq):6d} phrases  sha={manifest[genre]['sha256']}", file=sys.stderr)

# global dedupe + interleave, write
rows, glob_seen = [], set()
for genre, _ in selected:
    for s in buckets[genre]:
        k = s.lower()
        if k not in glob_seen:
            glob_seen.add(k); rows.append((genre, s))
rng_for("assemble").shuffle(rows)

with open(os.path.join(A.out_dir, "corpus.tsv"), "w") as f:
    for g, s in rows:
        f.write(f"{g}\t{s}\n")
with open(os.path.join(A.out_dir, "corpus.txt"), "w") as f:
    for _, s in rows:
        f.write(s + "\n")
manifest["_total"] = len(rows)
manifest["_by_genre_final"] = {g: sum(1 for gg, _ in rows if gg == g) for g in buckets}
with open(os.path.join(A.out_dir, "corpus_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

dt = time.perf_counter() - t0
print(f"\nTOTAL {len(rows)} phrases in {dt:.1f}s -> {A.out_dir}/corpus.tsv  /  corpus.txt")
print("by genre:", manifest["_by_genre_final"])
