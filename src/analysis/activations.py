"""
activations.py — Extraction d'activations avec masquage rigoureux.

Diagnostic "syndrome du premier mot"
------------------------------------
Cause principale : attention sink / massive activations (Xiao et al. 2023 ;
Sun et al. 2024). Le BOS (et souvent le 1er token de contenu) porte des
activations de norme 10–100× supérieure dans le résidu. Comme Pipeline 1
max-poole en espace SAE (doc_vec[f] = max_t enc(x_t)[f]), tout feature
partiellement aligné avec la direction "sink" est saturé par la position 0.

Défense (3 couches, conformes à la pratique GDM/GemmaScope) :
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
    """Exclut les tokens dont ||x_t|| est un outlier intra-DOCUMENT (massive activations
    résiduelles) -- mu/sd calculées séparément par document (B.10, docs/AUDIT_2026-08.md) :
    les calculer sur tout le batch aplati mélangeait des documents indépendants et rendait
    le seuil dépendant de `batch_size` (quels AUTRES documents partagent le batch), pas
    seulement du document lui-même."""
    norms = resid.norm(dim=-1)                       # [B, T]
    out = mask.clone()
    for b in range(norms.shape[0]):
        vals = norms[b][mask[b]]
        if vals.numel() < 8:
            continue
        mu, sd = vals.mean(), vals.std() + 1e-6
        out[b] &= (norms[b] - mu) / sd < sigma_clip
    return out


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
    # @torch.no_grad() sur une fonction génératrice ne protège QUE l'appel qui crée
    # l'objet générateur (quasi instantané, le corps ne s'exécute pas encore) — pas les
    # itérations réelles faites via next()/for. Sans le `with` explicite ci-dessous,
    # chaque forward tourne avec l'autograd actif (graphe retenu pour les ~48 couches,
    # output_hidden_states=True) → OOM CUDA avant la fin du corpus sur A100 40GB.
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)       # dict explicite — jamais enc.to(device)
            attn = enc["attention_mask"].to(device)
            # logits_to_keep=1 : seul hidden_states nous intéresse ici, mais le forward
            # de AutoModelForCausalLM calcule par défaut (logits_to_keep=0) les logits sur
            # TOUTE la séquence (vocab Gemma-3 ~262k) -> ex. 2 GiB rien que pour un batch
            # de 8 sur 512 tokens.
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True, logits_to_keep=1)
            # out.hidden_states garde une référence à TOUTES les couches (48 pour le 12b)
            # simultanément tant que `out` vit ; on n'en veut qu'une -> clone + del pour
            # libérer les 47 autres immédiatement.
            resid = out.hidden_states[layer].clone()      # [B, T, d]
            del out

            mask = valid_token_mask(input_ids, attn, tokenizer, skip_first_content_token)
            mask = norm_outlier_mask(resid, mask, sigma_clip)

            b_idx, t_idx = mask.nonzero(as_tuple=True)
            yield ActBatch(
                acts=resid[b_idx, t_idx].float().cpu(),
                doc_ids=(b_idx + start).cpu(),
                token_pos=t_idx.cpu(),
            )


def scatter_maxpool(
    values: torch.Tensor,        # [n_units, d]  (tokens ou phrases)
    unit_to_doc: torch.Tensor,   # [n_units] int64
    n_docs: int,
    d: int = None,
) -> torch.Tensor:
    """
    doc_vec[j, f] = max_{i : unit_to_doc[i]=j} values[i, f].
    Implémentation UNIQUE du max-pooling en espace SAE (scatter_reduce amax).
    Remplace : sae_shared.pool_embeddings_by_document,
               phrase_sae.encode_documents_with_phrase_sae (boucle interne),
               la double boucle de maxpool_sae_docs.
    Docs sans unité → vecteur nul.
    """
    d = d or values.shape[1]
    out = torch.full((n_docs, d), float("-inf"), dtype=values.dtype, device=values.device)
    idx = unit_to_doc.long().unsqueeze(-1).expand(-1, d)
    out.scatter_reduce_(0, idx, values, reduce="amax", include_self=False)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def maxpool_sae_docs(
    act_stream: Iterator[ActBatch],
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    n_docs: int,
    d_sae: int,
    device: str = "cuda",
    chunk: int = 4096,
) -> torch.Tensor:
    """doc_vec[f] = max_t enc(x_t)[f] — pooling APRÈS encodage SAE, sur tokens valides uniquement.

    Scatter direct dans l'accumulateur persistant (in-place), au lieu d'appeler
    scatter_maxpool (qui réalloue et parcourt un tenseur [n_docs, d_sae] COMPLET)
    à chaque chunk/batch : sur un corpus de dizaines de milliers de documents, ce
    coût O(n_docs·d) payé à chaque batch (au lieu de O(batch·d)) devient
    quadratique en pratique et fait exploser le temps total.
    """
    doc_acts = torch.full((n_docs, d_sae), float("-inf"))
    for ab in act_stream:
        for s in range(0, ab.acts.shape[0], chunk):
            f = encode_fn(ab.acts[s:s + chunk].to(device)).float().cpu()
            ids = ab.doc_ids[s:s + chunk].long()
            idx = ids.unsqueeze(-1).expand(-1, d_sae)
            doc_acts.scatter_reduce_(0, idx, f, reduce="amax", include_self=True)
    return torch.where(torch.isinf(doc_acts), torch.zeros_like(doc_acts), doc_acts)