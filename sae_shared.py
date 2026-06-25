"""
sae_shared.py — Utilitaires partagés pour les pipelines SAE (v8).
Alignement strict sur interp_embed (Nick Jiang et al.) et SAELens.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "external/interp_embed"))
sys.path.insert(0, os.path.join(ROOT_DIR, "external/sae-lens"))

try:
    from interp_embed.sae.utils import get_reconstruction_error
    from interp_embed import Dataset as InterpDataset
except ImportError:
    InterpDataset = None

from src.data.preparation import (
    keyword_match,
    prepare_domain_dataset,
    split_into_phrases,
    load_and_clean_emails,
    url_match,
)
from src.analysis.metrics import (
    compute_metrics,
    compute_rho_sae,
    downstream_classification,
)
from src.sae.frozen_core import ExtendedSAE, FrozenCoreResidualSAE
from src.sae.phrase_sae import (
    PhraseLevelSAE,
    extract_f2llm_embeddings,
    encode_documents_with_phrase_sae,
    load_or_train_sae,
    compute_sae_metrics,
)

from src.data.keywords import (
    ENERGY_KEYWORDS,
    ENERGY_URL_PATTERNS,
    SPORTS_KEYWORDS,
    SPORTS_URL_PATTERNS,
    SUPPORT_KEYWORDS,
    SUPPORT_URL_PATTERNS,
)


# ─── DIFFING DE RÉFÉRENCE (ALIGNÉ SUR INTERP_EMBED) ───

def diff_features(
    acts1: torch.Tensor,
    acts2: torch.Tensor,
    feature_labels: Dict[int, str] = None,
    min_coverage: float = 0.0,
    max_coverage: float = 1.0,
) -> pd.DataFrame:
    """
    Compare deux distributions d'activations SAE par écart absolu de fréquences.
    Garde les features dont la fréquence est dans [min_coverage, max_coverage]
    dans les DEUX corpus (filtrage symétrique).
    """
    fa1 = (acts1.float().detach().cpu() > 1e-6).numpy()
    fa2 = (acts2.float().detach().cpu() > 1e-6).numpy()

    freq1 = fa1.mean(axis=0)
    freq2 = fa2.mean(axis=0)

    diff = freq1 - freq2
    n_features = freq1.shape[0]
    labels = (
        [feature_labels.get(i, f"Feature #{i}") for i in range(n_features)]
        if feature_labels
        else [f"Feature #{i}" for i in range(n_features)]
    )

    df = pd.DataFrame({
        "feature_id": np.arange(n_features),
        "feature_label": labels,
        "freq_A": freq1,
        "freq_B": freq2,
        "frequency_difference": diff,
    })

    # FIX: filtrage symétrique — les deux fréquences doivent être dans [min_coverage, max_coverage].
    # Version originale : (freq_A >= min_coverage) & (freq_B <= max_coverage) était asymétrique
    # et filtrait incorrectement les features trop fréquentes dans A mais pas dans B.
    df = df[
        df["freq_A"].between(min_coverage, max_coverage) &
        df["freq_B"].between(min_coverage, max_coverage)
    ]
    return (
        df.reindex(df["frequency_difference"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )


# ─── NPMI EN PYTORCH ───

def compute_npmi(acts: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """
    NPMI = PMI / -log(p(i,j)) en PyTorch.
    NPMI(i,i) = 1 par convention (diagonale forcée).
    """
    device = acts.device
    X = (acts.float() > threshold).float()
    n_samples = X.shape[0]

    cooc = X.T @ X
    eps = 1e-10

    p_i = X.mean(dim=0).clamp(min=eps)
    p_ij = cooc / n_samples

    p_i_outer = p_i.unsqueeze(1) @ p_i.unsqueeze(0)

    pmi = torch.log(p_ij.clamp(min=eps) / p_i_outer)
    log_p_ij = torch.log(p_ij.clamp(min=eps))

    npmi = pmi / (-log_p_ij)
    npmi = torch.nan_to_num(npmi, nan=0.0, posinf=1.0, neginf=-1.0)
    npmi.fill_diagonal_(1.0)
    return npmi.cpu()


# ─── VISUALISATION ───

def highlight_activations_as_string(
    tokens: List[str],
    activations: np.ndarray,
    left_marker: str = "<<",
    right_marker: str = ">>",
) -> str:
    result = []
    in_highlight = False
    for tok, act in zip(tokens, activations):
        if act > 1e-6 and not in_highlight:
            result.append(left_marker)
            in_highlight = True
        if act <= 1e-6 and in_highlight:
            result.append(right_marker)
            in_highlight = False
        result.append(tok)
    if in_highlight:
        result.append(right_marker)
    return "".join(result).replace(" ", " ").replace("Ġ", " ").replace("▁", " ")


def highlight_doc_for_feature(
    token_data: Dict[str, Any],
    feature_idx: int,
    left: str = "<<",
    right: str = ">>",
) -> str:
    tokens = token_data["token_strings"]
    if isinstance(token_data["token_sae_acts"], torch.Tensor) and token_data["token_sae_acts"].is_sparse:
        dense_col = token_data["token_sae_acts"].to_dense()[:, feature_idx].numpy()
    else:
        sparse_col = token_data["token_sae_acts"][:, feature_idx]
        dense_col = np.asarray(sparse_col.todense()).flatten()
    return highlight_activations_as_string(tokens, dense_col, left, right)


# ─── STEERING ───

def steer_activations(
    doc_acts: torch.Tensor,
    amplifications: Dict[int, float],
) -> torch.Tensor:
    steered = doc_acts.clone()
    for f_idx, mult in amplifications.items():
        steered[:, f_idx] = steered[:, f_idx] * mult
    return steered.to(torch.bfloat16)


def steer_and_decode(
    doc_acts: torch.Tensor,
    amplifications: Dict[int, float],
    sae: nn.Module,
) -> torch.Tensor:
    steered = steer_activations(doc_acts, amplifications)
    device = next(sae.parameters()).device
    with torch.no_grad():
        return sae.decode(steered.to(device).to(torch.bfloat16))


def pool_embeddings_by_document(
    phrase_embeddings: torch.Tensor,
    phrase_to_doc: np.ndarray,
    n_docs: int = None,
) -> torch.Tensor:
    if phrase_embeddings.shape[0] != len(phrase_to_doc):
        raise ValueError("Incohérence de taille entre phrases et mappage de documents.")

    if n_docs is None:
        n_docs = int(phrase_to_doc.max()) + 1

    d = phrase_embeddings.shape[1]
    doc_emb = torch.full((n_docs, d), float('-inf'), dtype=phrase_embeddings.dtype)

    for phrase_idx, doc_idx in enumerate(phrase_to_doc):
        doc_idx = int(doc_idx)
        doc_emb[doc_idx] = torch.max(doc_emb[doc_idx], phrase_embeddings[phrase_idx])

    doc_emb[doc_emb == float('-inf')] = 0.0
    return doc_emb