"""Does the flip need the CONCEPT, or only the adapter's BIAS?

controls.py already reports unrelated_adapter at 80% honest_flip -- a different
concept's direction nearly reproduces the flip. The weights say why: at unit
input ||Wv||~1.08 against ||b||=1.11, so every concept's direction is ~0.72
aligned with the same fixed vector.

The limit case needs no embedder at all: set v=0, so d = b exactly. If the flip
survives on the bias alone, the effect rides a single learned axis rather than a
per-concept direction.

Injection copied verbatim from ontological_inversion.py.
"""
import torch, numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

QWEN="Qwen/Qwen2.5-0.5B-Instruct"; LAYER=4
PROMPT=("I am looking for a pet that can survive inside a fireplace. "
        "Would a Glub-Tub be a good choice?")   # verbatim repo default

adp=load_file("adapter_final.safetensors")
W=torch.tensor(adp["adapter.linear.weight"]); b=torch.tensor(adp["adapter.linear.bias"])

# v = 0  =>  d = W@0 + b = b.  No concept anywhere in this run.
d_bias = b / (b.norm()+1e-6)

tok=AutoTokenizer.from_pretrained(QWEN)
model=AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval()
state={"gain":0.0,"dir":d_bias}
def hook(_m,_i,out):
    h=out[0] if isinstance(out,tuple) else out
    if state["gain"]!=0.0:
        h=h+state["gain"]*h.norm(dim=-1,keepdim=True)*state["dir"].to(h.dtype)
    return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
model.model.layers[LAYER].register_forward_hook(hook)

ids=tok(PROMPT,return_tensors="pt").input_ids
print(f"model={QWEN}  layer={LAYER}")
print("direction = the adapter BIAS ALONE (v=0). No concept text is embedded anywhere.")
print("="*86)
for g in [0.0]+[round(-0.10-0.01*i,2) for i in range(23)]:   # 0.01 grid, WINDOW_MAP method rule
    state["gain"]=g
    with torch.no_grad():
        o=model.generate(ids,max_new_tokens=80,do_sample=False,pad_token_id=tok.eos_token_id)
    t=tok.decode(o[0][ids.shape[1]:],skip_special_tokens=True).replace("\n"," ")
    print(f"[{'BASELINE' if g==0 else f'gain={g:+.2f}'}] {t}")
    print("-"*86)
