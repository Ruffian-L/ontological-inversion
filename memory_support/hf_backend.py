#!/usr/bin/env python3
"""Local HF backend for a reviewed plan. Writes one append-only JSONL receipt per output."""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelpin import rev
from ontological_inversion import concept_direction

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("plan"); ap.add_argument("--out",required=True); ap.add_argument("--adapter",default="adapter_final.safetensors")
    ap.add_argument("--limit",type=int); ap.add_argument("--device",default="auto",choices=["auto","cuda","cpu"]); a=ap.parse_args(); cfg=json.loads(Path("memory_support/config.json").read_text())
    plan=[json.loads(x) for x in Path(a.plan).read_text().splitlines() if x]; plan=plan[:a.limit] if a.limit else plan
    device="cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device)
    dtype=torch.bfloat16 if device=="cuda" else torch.float32
    tok=AutoTokenizer.from_pretrained(cfg["model"],revision=rev(cfg["model"]),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(cfg["model"],revision=rev(cfg["model"]),local_files_only=True,dtype=dtype).to(device).eval()
    emb=SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",revision=rev("nomic-ai/nomic-embed-text-v1.5"),trust_remote_code=True,local_files_only=True)
    state={"gain":0.0,"direction":None}
    def hook(_m,_i,out):
        h=out[0] if isinstance(out,tuple) else out
        if state["direction"] is not None and state["gain"]:
            h=h+state["gain"]*h.norm(dim=-1,keepdim=True)*state["direction"].to(h.device,h.dtype)
        return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
    model.model.layers[cfg["layer"]].register_forward_hook(hook)
    all_sources={r["cluster_id"]:r["slots"] for r in plan}
    out_path=Path(a.out); completed=set()
    if out_path.exists():
        completed={json.loads(x)["cell_id"] for x in out_path.read_text().splitlines() if x}
    direction_cache={}
    print(f"device={device} dtype={dtype} planned={len(plan)} resume_skip={len(completed)}",flush=True)
    with open(a.out,"a") as f:
        for r in plan:
            if r["cell_id"] in completed: continue
            slots=r["slots"]; arm=r["arm"]
            if arm=="wrong_memory": slots=all_sources[r["wrong_cluster_id"]]
            elif arm=="wrong_slot_order": slots=list(reversed(slots))
            source=" ".join(slots)
            state["gain"]=float(r["gain"]); state["direction"]=None
            if arm not in ("blank","text_oracle"):
                cache_key=(arm,source,r["seed"] if arm in ("permuted","random") else None)
                d=direction_cache.get(cache_key)
                if d is None:
                    d=concept_direction(a.adapter,emb,source)
                    if arm=="permuted": d=d[torch.randperm(len(d),generator=torch.Generator().manual_seed(r["seed"]))]
                    elif arm=="random": d=torch.randn(d.shape,generator=torch.Generator().manual_seed(r["seed"])); d=d/d.norm()
                    direction_cache[cache_key]=d
                state["direction"]=d
            prompt=r["prompt"]
            if arm=="text_oracle": prompt="Injected memory (text oracle): "+r["source"]+"\n\n"+prompt
            torch.manual_seed(r["seed"]); np.random.seed(r["seed"]%2**32); random.seed(r["seed"])
            encoded=tok(prompt,return_tensors="pt")
            ids=encoded.input_ids.to(device); attention_mask=encoded.attention_mask.to(device)
            with torch.no_grad(): out=model.generate(ids,attention_mask=attention_mask,max_new_tokens=cfg["max_new_tokens"],do_sample=cfg["do_sample"],temperature=cfg["temperature"],top_p=cfg["top_p"],pad_token_id=tok.eos_token_id)
            text=tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)
            f.write(json.dumps({**r,"output":text},sort_keys=True)+"\n"); f.flush()
            print(f"cell={r['cell_id']} arm={arm} gain={r['gain']:+.2f} seed={r['seed']}",flush=True)
if __name__=="__main__": main()
