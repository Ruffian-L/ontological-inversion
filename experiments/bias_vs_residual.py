"""E8 -- causal decomposition of the injected direction.

2026-08-25 established that the BIAS ALONE inverts (v=0, d=b): stove at
alpha in [-0.13,-0.12], container at [-0.28,-0.24]. The reading was:

    bias    -> alive -> object
    concept -> which object, and the Fire Invariant

That reading rests on one half of the decomposition. The complement was never
run. This runs it.

Arms, all on the 0.01 grid per WINDOW_MAP.md's method rule:

    bias        d = b                     (done 2026-08-25, repeated here for
                                           a within-run comparison)
    residual    d = W v                   BIAS REMOVED -- the falsifier. If this
                                           alone inverts, the reading above is
                                           wrong and gets withdrawn.
    lambda      d = b + lam * W v         recombination; is it compositional?
    swap        d = b + W v_other         the bias with an unrelated concept's
                                           residual

Injection copied verbatim from ontological_inversion.py. nomic is loaded through
plain transformers (offline, pinned cache) because sentence-transformers is not
installed; pooling is attention-masked mean + L2, matching SentenceTransformer.
"""
import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np, sys
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

QWEN="Qwen/Qwen2.5-0.5B-Instruct"; LAYER=4; NOMIC="nomic-ai/nomic-embed-text-v1.5"
PROMPT=("I am looking for a pet that can survive inside a fireplace. "
        "Would a Glub-Tub be a good choice?")
CONCEPT="A Glub-Tub is a magma-eating hamster that lives inside a tub."
OTHER="A wolf is a fierce predator that hunts in packs in the forest."

adp=load_file("adapter_final.safetensors")
W=adp["adapter.linear.weight"].float(); b=adp["adapter.linear.bias"].float()

ntok=AutoTokenizer.from_pretrained(NOMIC)
nm=AutoModel.from_pretrained(NOMIC, trust_remote_code=True).eval()
def nomic(text):
    bb=ntok(["search_document: "+text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad(): o=nm(**bb)[0]
    mk=bb["attention_mask"].unsqueeze(-1).float()
    v=torch.nn.functional.normalize((o*mk).sum(1)/mk.sum(1), dim=1)[0]
    v=v[:W.shape[1]]                       # caller slices to 128 ...
    return v/(v.norm()+1e-6)               # ... then renormalises (line 52)

v_c, v_o = nomic(CONCEPT), nomic(OTHER)
Wv_c, Wv_o = W@v_c, W@v_o
print(f"||b||={b.norm():.4f}  ||Wv_concept||={Wv_c.norm():.4f}  ||Wv_other||={Wv_o.norm():.4f}")
print(f"cos(b, Wv_c)={torch.nn.functional.cosine_similarity(b,Wv_c,dim=0):.4f}   "
      f"cos(Wv_c, Wv_o)={torch.nn.functional.cosine_similarity(Wv_c,Wv_o,dim=0):.4f}", flush=True)

tok=AutoTokenizer.from_pretrained(QWEN)
DEV="cuda" if torch.cuda.is_available() else "cpu"
model=AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval().to(DEV)
print("generator on", DEV, "- fp32, kept identical to the CPU runs for comparability", flush=True)
state={"gain":0.0,"dir":None}
def hook(_m,_i,out):
    h=out[0] if isinstance(out,tuple) else out
    if state["gain"]!=0.0 and state["dir"] is not None:
        h=h+state["gain"]*h.norm(dim=-1,keepdim=True)*state["dir"].to(device=h.device, dtype=h.dtype)
    return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
model.model.layers[LAYER].register_forward_hook(hook)
ids=tok(PROMPT,return_tensors="pt").input_ids.to(DEV)

def unit(x): return x/(x.norm()+1e-6)
ARMS=[("bias",       unit(b)),
      ("residual",   unit(Wv_c)),          # <- the falsifier
      ("full",       unit(Wv_c+b)),
      ("lam0.25",    unit(b+0.25*Wv_c)),
      ("lam0.50",    unit(b+0.50*Wv_c)),
      ("lam2.0",     unit(b+2.0*Wv_c)),
      ("lam4.0",     unit(b+4.0*Wv_c)),
      ("swap_other", unit(b+Wv_o))]
GAINS=[0.0]+[round(-0.10-0.01*i,2) for i in range(23)]

for name,d in ARMS:
    print(f"\n{'='*100}\nARM: {name}\n{'='*100}", flush=True)
    # keep the direction on the generator's device; a CPU tensor here forces a
    # host round-trip on every generated token
    state["dir"]=d.to(DEV)
    for g in GAINS:
        state["gain"]=g
        with torch.no_grad():
            o=model.generate(ids,max_new_tokens=80,do_sample=False,pad_token_id=tok.eos_token_id)
        t=tok.decode(o[0][ids.shape[1]:],skip_special_tokens=True).replace("\n"," ")
        print(f"[{name} gain={g:+.2f}] {t}", flush=True)
