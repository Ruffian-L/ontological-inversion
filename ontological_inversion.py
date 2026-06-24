#!/usr/bin/env python3
"""
Ontological Inversion ("The Anti-Splat") — reproducible baseline.

Claim: negative steering of a concept does NOT just erase it. Under an anchor it
moves to the concept's *structured opposite* (living -> inanimate, etc.), while
meaning stays coherent across a gain band. This is an empirical approximation of a
self-involution / Householder reflection in concept space:  Phi_c(h) = mu + (I - 2 P_c)(h - mu).

Pipeline (faithful to the original SplatRAG system):
  concept text --nomic-embed-v1.5 (Matryoshka[:128])--> 128-d
              --adapter_final.safetensors (trained linear 128->896, the "Synapse")--> direction
  inject that TRAINED direction into the LLM residual stream (one early layer, every
  position), scaled to the local hidden norm * gain, then greedy-generate.
  gain < 0  => subtract the concept => the inversion.  Sweet spot ~ -0.15..-0.30; collapses past -0.4.

Run:
  pip install -r requirements.txt
  python ontological_inversion.py                      # default Glub-Tub demo
  python ontological_inversion.py --concept "wolf predator hunting fierce living animal" \
         --prompt "Describe a wolf in the forest." "--gains=0,-0.2,-0.25"
"""
import argparse, os, numpy as np, torch
from safetensors.numpy import load_file
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    p = argparse.ArgumentParser(description="Ontological Inversion baseline")
    p.add_argument("--qwen", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF model id (original used Qwen2.5-Coder-0.5B-Instruct)")
    p.add_argument("--embedder", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--adapter", default=os.path.join(HERE, "adapter_final.safetensors"))
    p.add_argument("--prompt", default="I am looking for a pet that can survive inside a "
                                        "fireplace. Would a Glub-Tub be a good choice?")
    p.add_argument("--concept", default="A Glub-Tub is a magma-eating hamster that lives inside a tub.")
    p.add_argument("--layer", type=int, default=4, help="residual layer to inject at")
    p.add_argument("--gains", default="0,-0.15,-0.2,-0.25,-0.3,-0.4",
                   help="comma list (use --gains=-0.2,... form for leading minus)")
    p.add_argument("--max-new", type=int, default=50)
    return p.parse_args()

def concept_direction(adapter_path, embedder, concept):
    """concept text -> trained 896-d injection direction (unit)."""
    adp = load_file(adapter_path)
    W = torch.tensor(adp["adapter.linear.weight"])   # [896,128]
    b = torch.tensor(adp["adapter.linear.bias"])      # [896]
    v = embedder.encode(["search_document: " + concept], convert_to_numpy=True)[0]
    v = v[:W.shape[1]]
    v = v / (np.linalg.norm(v) + 1e-6)
    d = (W @ torch.tensor(v, dtype=torch.float32)) + b
    return d / (d.norm() + 1e-6)

def main():
    a = parse_args()
    embedder = SentenceTransformer(a.embedder, trust_remote_code=True)
    direction = concept_direction(a.adapter, embedder, a.concept)
    tok = AutoTokenizer.from_pretrained(a.qwen)
    model = AutoModelForCausalLM.from_pretrained(a.qwen, dtype=torch.float32).eval()

    state = {"gain": 0.0}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["gain"] != 0.0:
            h = h + state["gain"] * h.norm(dim=-1, keepdim=True) * direction.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    model.model.layers[a.layer].register_forward_hook(hook)

    ids = tok(a.prompt, return_tensors="pt").input_ids
    print(f"model={a.qwen}  layer={a.layer}")
    print(f"prompt : {a.prompt}")
    print(f"concept (subtracted at gain<0): {a.concept}")
    print("=" * 78)
    for g in [float(x) for x in a.gains.split(",")]:
        state["gain"] = g
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " ")
        print(f"[{'BASELINE' if g == 0 else f'gain={g:+.2f}'}] {txt}")
        print("-" * 78)

if __name__ == "__main__":
    main()
