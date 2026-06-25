"""
Steering operators for ontological inversion, applied to a residual hidden state
`h` [batch, seq, hidden] given a unit concept direction `d` [hidden].

All operators share one knob `strength` (>= 0); strength == 0 is identity (baseline),
so they're directly comparable on one sweep grid.

  negative_gain       : h + (-strength) * ||h|| * d        (the current baseline; subtract concept)
  householder         : (1-s)h + s*(h - 2 (h.d) d)         (TRUE reflection about the concept
                        hyperplane; s=strength. A genuine involution: applied twice => identity)
  projection_polarity : set the concept's signed component to -strength*|p_before|
                        (mirrors niodoo/src/physics/ontological_inversion.rs::polarity_aware_inversion)

Direction sources (return a unit [hidden] tensor in the model's hidden space):
  adapter_direction     : trained Synapse  (nomic-128 -> adapter_final.safetensors -> 896)
  contrastive_direction : mean(concept word embeddings) - mean(antipode word embeddings)
"""
import numpy as np
import torch
from safetensors.numpy import load_file


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-6)


# --------------------------------------------------------------------------- #
# direction sources
# --------------------------------------------------------------------------- #
def adapter_direction(adapter_path, embedder, concept):
    adp = load_file(adapter_path)
    W = torch.tensor(adp["adapter.linear.weight"])  # [896, 128]
    b = torch.tensor(adp["adapter.linear.bias"])     # [896]
    v = embedder.encode(["search_document: " + concept], convert_to_numpy=True)[0]
    v = v[: W.shape[1]]
    v = v / (np.linalg.norm(v) + 1e-6)
    d = (W @ torch.tensor(v, dtype=torch.float32)) + b
    return _unit(d)


def contrastive_direction(model, tok, concept_words, antipode_words):
    emb = model.get_input_embeddings()

    def mean_emb(words):
        ids = tok(" " + " ".join(words), return_tensors="pt").input_ids
        with torch.no_grad():
            return emb(ids)[0].mean(0).float()

    return _unit(mean_emb(concept_words) - mean_emb(antipode_words))


# --------------------------------------------------------------------------- #
# operators:  (h [b,seq,hidden], d [hidden] unit, strength float) -> h'
# --------------------------------------------------------------------------- #
def op_negative_gain(h, d, strength):
    return h + (-strength) * h.norm(dim=-1, keepdim=True) * d


def op_householder(h, d, strength):
    p = (h * d).sum(-1, keepdim=True)          # h . d   (d is unit)
    reflected = h - 2.0 * p * d                 # (I - 2 d d^T) h
    return (1.0 - strength) * h + strength * reflected


def op_projection_polarity(h, d, strength):
    p = (h * d).sum(-1, keepdim=True)
    target = -strength * p.abs()                # set signed component to -strength*|p|
    return h + (target - p) * d


OPERATORS = {
    "negative_gain": op_negative_gain,
    "householder": op_householder,
    "projection_polarity": op_projection_polarity,
}


def householder_involution_error(dim=896, seed=0):
    """f(f(h)) should return h for a true involution. Returns ||f(f(h)) - h||."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, 4, dim, generator=g)
    d = _unit(torch.randn(dim, generator=g))
    once = op_householder(h, d, 1.0)
    twice = op_householder(once, d, 1.0)
    return (twice - h).norm().item()


if __name__ == "__main__":
    err = householder_involution_error()
    print(f"householder involution error ||f(f(h))-h|| = {err:.2e}  ->",
          "OK" if err < 1e-4 else "FAIL")
