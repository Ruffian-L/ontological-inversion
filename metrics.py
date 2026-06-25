"""
Proxy metrics for the inversion (no trained classifier; honest about that).

  inversion_score   = cos(out, antipode_anchor) - cos(out, concept_anchor)   ( >0 => flipped )
  preservation      = cos(out, shared_axis_anchor)   (does the shared axis survive the flip?)
  coherence(text)   = distinct-token ratio + repeated-4gram rate + collapsed flag

All embeddings use the same sentence-transformer (nomic) the steering uses.
Anchor vectors are precomputed once per concept and passed in (fast).
"""
import numpy as np


def embed(embedder, text):
    return embedder.encode(["search_document: " + text], convert_to_numpy=True)[0]


def anchor(embedder, words):
    return embed(embedder, " ".join(words))


def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def inversion_score(out_vec, concept_anchor_vec, antipode_anchor_vec):
    return cos(out_vec, antipode_anchor_vec) - cos(out_vec, concept_anchor_vec)


def preservation(out_vec, shared_anchor_vec):
    return None if shared_anchor_vec is None else cos(out_vec, shared_anchor_vec)


def coherence(text):
    from collections import Counter
    toks = text.split()
    if len(toks) < 4:
        return {"distinct": 0.0, "rep4": 1.0, "collapsed": True}
    distinct = len(set(toks)) / len(toks)
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    rep4 = 1.0 - len(set(grams)) / max(1, len(grams))
    top_freq = Counter(toks).most_common(1)[0][1] / len(toks)          # most-repeated token share
    alpha_frac = sum(c.isalpha() or c.isspace() for c in text) / max(1, len(text))  # vs digit/punct spam
    collapsed = (distinct < 0.5) or (rep4 > 0.3) or (top_freq > 0.25) or (alpha_frac < 0.6)
    return {"distinct": round(distinct, 3), "rep4": round(rep4, 3),
            "top_freq": round(top_freq, 3), "alpha_frac": round(alpha_frac, 3), "collapsed": collapsed}
