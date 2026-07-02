"""
activations.py — Extraction d'activations avec masquage rigoureux.

Diagnostic "syndrome du premier mot"
------------------------------------
Cause principale : attention sink / massive activations (Xiao et al. 2023 ;
Sun et al. 2024). Le BOS (et souvent le 1er token de contenu) porte des
activations de norme 10–100× supérieure dans le résidu. Comme Pipeline 1
max-poole en espace SAE (doc_vec[f] = max_t enc(x_t)[f]), tout feature
partiellement aligné avec la direction "sink" est saturé par la position 0.
Ce n'est PAS un bug de slicing — mais l'ancien code aggravait l'artefact :
(1) BOS non exclu du pool, (2) tokens de padding inclus quand
attention_mask n'était pas propagé jusqu'au pooling.

Correctif (3 couches de défense, conformes à la pratique GDM/GemmaScope) :
  1. Masque des special tokens (BOS/EOS/PAD) exclus de tout pooling.
  2. Option skip_first_content_token (le 1er token réel absorbe aussi du sink).
  3. Garde-fou norme : z-score des ||x_t||₂ intra-doc, exclusion > sigma_clip.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterator

import torch


@dataclass
class ActBatch:
    acts: torch.Tensor          # [n_valid_tokens, d_model]
    doc_ids: torch.Tensor       # [n_valid_tokens]
    token_pos: torch.Tensor     # [n_valid_tokens]


def valid_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    skip_first_content_token: bool = True,
) -> torch.Tensor:
    """[B, T] bool. Exclut PAD, special tokens, et optionnellement le 1er token de contenu."""
    mask = attention_mask.bool()
    special = torch.zeros_like(mask)
    for tid in tokenizer.all_special_ids:
        special |= input_ids == tid
    mask &= ~special
    if skip_first_content_token:
        first = mask.float().cumsum(dim=1) == 1
        mask &= ~first
    return mask


def norm_outlier_mask(resid: torch.Tensor, mask: torch.Tensor, sigma_clip: float = 4.0) -> torch.Tensor:
    """Exclut les tokens dont ||x_t|| est un outlier intra-batch (massive activations résiduelles)."""
    norms = resid.norm(dim=-1)                       # [B, T]
    vals = norms[mask]
    if vals.numel() < 8:
        return mask
    mu, sd = vals.mean(), vals.std() + 1e-6
    return mask & ((norms - mu) / sd < sigma_clip)


@torch.no_grad()
def extract_residual_acts(
    texts: list[str],
    model,
    tokenizer,
    layer: int,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    skip_first_content_token: bool = True,
    sigma_clip: float = 4.0,
) -> Iterator[ActBatch]:
    """
    Stream d'activations résiduelles masquées. `model` : HF causal LM avec
    output_hidden_states ; layer indexé sur hidden_states (0 = embeddings).
    """
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)          # dict explicite — jamais enc.to(device)
        attn = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        resid = out.hidden_states[layer]                  # [B, T, d]

        mask = valid_token_mask(input_ids, attn, tokenizer, skip_first_content_token)
        mask = norm_outlier_mask(resid, mask, sigma_clip)

        b_idx, t_idx = mask.nonzero(as_tuple=True)
        yield ActBatch(
            acts=resid[b_idx, t_idx].float().cpu(),
            doc_ids=(b_idx + start).cpu(),
            token_pos=t_idx.cpu(),
        )


def maxpool_sae_docs(
    act_stream: Iterator[ActBatch],
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    n_docs: int,
    d_sae: int,
    device: str = "cuda",
    chunk: int = 4096,
) -> torch.Tensor:
    """doc_vec[f] = max_t enc(x_t)[f] — pooling APRÈS encodage SAE, sur tokens valides uniquement."""
    doc_acts = torch.zeros(n_docs, d_sae)
    for ab in act_stream:
        for s in range(0, ab.acts.shape[0], chunk):
            f = encode_fn(ab.acts[s:s + chunk].to(device)).float().cpu()
            ids = ab.doc_ids[s:s + chunk]
            for d in ids.unique():
                sel = ids == d
                doc_acts[d] = torch.maximum(doc_acts[d], f[sel].max(dim=0).values)
    return doc_acts