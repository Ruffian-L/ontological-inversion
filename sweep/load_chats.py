#!/usr/bin/env python3
"""Load chat phrases from a clean, openly-licensed Hugging Face dataset -> data/corpus.txt.

NO SCRAPING. Default source: OpenAssistant/oasst1 (Apache-2.0) — human-written
assistant conversations, quality-filtered, the kind normal labs train on (not
red-team / "poisoned" data). We read the message turns, keep English, dedup, and
write one phrase per line. That's it.

  pip install datasets
  python sweep/load_chats.py                       # ~10k English turns from oasst1
  python sweep/load_chats.py --limit 20000
  python sweep/load_chats.py --dataset HuggingFaceH4/ultrachat_200k --split train_sft --lang ""

Other clean, openly-licensed options:
  - HuggingFaceH4/ultrachat_200k   (MIT)        multi-turn, schema: messages=[{role,content}]
  - databricks/databricks-dolly-15k (CC-BY-SA)  instruction/response
"""
import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iter_texts(row, field):
    """Yield message text(s) from a row, handling both the flat ('text') schema
    (oasst1) and the chat-list schema ('messages'=[{role, content}, ...])."""
    val = row.get(field)
    if isinstance(val, str):
        yield val
    elif isinstance(row.get("messages"), list):
        for m in row["messages"]:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                yield m["content"]


def main():
    ap = argparse.ArgumentParser(description="Load chat phrases from a HF dataset (no scraping).")
    ap.add_argument("--dataset", default="OpenAssistant/oasst1",
                    help="openly-licensed HF dataset id (default: Apache-2.0 oasst1)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--lang", default="en",
                    help="keep only this lang when the row has a 'lang' field ('' = keep all)")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--min-len", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--no-stream", action="store_true",
                    help="download the whole split instead of streaming it")
    ap.add_argument("--out", default="corpus.txt")
    a = ap.parse_args()
    out = os.path.join(ROOT, "data", os.path.basename(a.out))  # confine to repo data/
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from datasets import load_dataset
    print(f"loading {a.dataset} [{a.split}] (streaming={not a.no_stream}) ...")
    ds = load_dataset(a.dataset, split=a.split, streaming=not a.no_stream)

    seen, lines = set(), []
    for row in ds:
        if a.lang and isinstance(row.get("lang"), str) and row["lang"] != a.lang:
            continue
        for t in iter_texts(row, a.text_field):
            t = " ".join(t.strip().split())  # collapse whitespace/newlines -> one line
            if not (a.min_len <= len(t) <= a.max_len):
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            lines.append(t)
            if len(lines) >= a.limit:
                break
        if len(lines) >= a.limit:
            break

    with open(out, "w") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(f"wrote {len(lines)} phrases -> {out}  (source: {a.dataset})")


if __name__ == "__main__":
    main()
