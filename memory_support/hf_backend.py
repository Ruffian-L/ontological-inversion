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
    ap.add_argument("--limit",type=int); a=ap.parse_args(); cfg=json.loads(Path("memory_support/config.json").read_text())
    plan=[json.loads(x) for x in Path(a.plan).read_text().splitlines() if x]; plan=plan[:a.limit] if a.limit else plan
    tok=AutoTokenizer.from_pretrained(cfg["model"],revision=rev(cfg["model"]),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(cfg["model"],revision=rev(cfg["model"]),local_files_only=True,dtype=torch.float32).eval()
    emb=SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",revision=rev("nomic-ai/nomic-embed-text-v1.5"),trust_remote_code=True,local_files_only=True)
    state={"gain":0.0,"direction":None}
    def hook(_m,_i,out):
        h=out[0] if isinstance(out,tuple) else out
        if state["direction"] is not None and state["gain"]:
            h=h+state["gain"]*h.norm(dim=-1,keepdim=True)*state["direction"].to(h.device,h.dtype)
        return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
    model.model.layers[cfg["layer"]].register_forward_hook(hook)
    all_sources={r["cluster_id"]:r["slots"] for r in plan}
    with open(a.out,"a") as f:
        for r in plan:
            slots=r["slots"]; arm=r["arm"]
            if arm=="wrong_memory": slots=all_sources[r["wrong_cluster_id"]]
            elif arm=="wrong_slot_order": slots=list(reversed(slots))
            source=" ".join(slots)
            state["gain"]=float(r["gain"]); state["direction"]=None
            if arm not in ("blank","text_oracle"):
                d=concept_direction(a.adapter,emb,source)
                if arm=="permuted": d=d[torch.randperm(len(d),generator=torch.Generator().manual_seed(r["seed"]))]
                elif arm=="random": d=torch.randn(d.shape,generator=torch.Generator().manual_seed(r["seed"])); d=d/d.norm()
                state["direction"]=d
            prompt=r["prompt"]
            if arm=="text_oracle": prompt="Injected memory (text oracle): "+r["source"]+"\n\n"+prompt
            torch.manual_seed(r["seed"]); np.random.seed(r["seed"]%2**32); random.seed(r["seed"])
            ids=tok(prompt,return_tensors="pt").input_ids
            with torch.no_grad(): out=model.generate(ids,max_new_tokens=cfg["max_new_tokens"],do_sample=cfg["do_sample"],temperature=cfg["temperature"],top_p=cfg["top_p"],pad_token_id=tok.eos_token_id)
            text=tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)
            f.write(json.dumps({**r,"output":text},sort_keys=True)+"\n"); f.flush()
            print(f"cell={r['cell_id']} arm={arm} gain={r['gain']:+.2f} seed={r['seed']}",flush=True)
if __name__=="__main__": main()
