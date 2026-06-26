"""Pinned model revisions — trust the bytes (STANDARDS.md §4).

These are the exact Hugging Face commit SHAs the published results in this repo were run
against. Loading with `revision=rev(id)` makes a future run byte-for-byte reproducible even
if the upstream org re-uploads weights. Unknown ids return None (= latest/unpinned).
"""
REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct":       "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
    "nomic-ai/nomic-embed-text-v1.5":   "e9b6763023c676ca8431644204f50c2b100d9aab",
}


def rev(model_id):
    return REVISIONS.get(model_id)
