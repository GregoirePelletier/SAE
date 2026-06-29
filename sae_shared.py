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

import json
from typing import Tuple, List, Dict

def _train_extended_sae(
    model: nn.Module,
    acts_train: torch.Tensor,
    epochs: int,
    lr: float,
    model_name: str,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict]:
    N = acts_train.shape[0]
    batch_size = int(os.environ.get("BATCH_TRAIN", "256"))
    steps_per_ep = max(1, N // batch_size)
    total_steps = steps_per_ep * epochs
    warmup_steps = min(500, total_steps // 10)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, betas=(0.0, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: s / max(warmup_steps, 1) if s < warmup_steps
        else 0.5 * (1.0 + math.cos(math.pi * (s - warmup_steps) / max(total_steps - warmup_steps, 1)))
    )
    history = {"nmse": [], "l0": [], "dead_frac": [], "step": []}
    step = 0
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(N)
        for start in range(0, N - batch_size + 1, batch_size):
            b = acts_train[perm[start: start + batch_size]].to(device).to(torch.bfloat16)
            out = model(b)
            optimizer.zero_grad(set_to_none=True)
            out["normalized_mse"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            if hasattr(model, "normalize_decoder"):
                model.normalize_decoder()
            if step % 50 == 0:
                history["nmse"].append(out["normalized_mse"].item())
                history["l0"].append(out["l0_extra"].item())
                history["dead_frac"].append(out["dead_frac"].item())
                history["step"].append(step)
            step += 1
        print(f"  [{model_name}] Epoch {epoch+1}/{epochs} | "
              f"NMSE={out['normalized_mse'].item():.4f} | "
              f"L0={out['l0_extra'].item():.1f} | dead={out['dead_frac'].item():.3f}")
    return model, history


def load_or_train(
    model: nn.Module,
    model_name: str,
    acts_train: torch.Tensor,
    epochs: int,
    lr: float,
    save_dir: str = "./results/",
    device: str = "cuda",
) -> Tuple[nn.Module, Dict]:
    pt_path = os.path.join(save_dir, f"{model_name.lower()}_state.pt")
    hist_path = os.path.join(save_dir, f"{model_name.lower()}_hist.json")
    if os.path.exists(pt_path) and os.path.exists(hist_path):
        print(f"  [sae_shared] Restauration checkpoint : {pt_path}")
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        with open(hist_path) as f:
            history = json.load(f)
    else:
        model, history = _train_extended_sae(
            model, acts_train, epochs=epochs, lr=lr,
            model_name=model_name, device=device,
        )
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, pt_path)
        with open(hist_path, "w") as f:
            json.dump(history, f)
        print(f"  [sae_shared] Checkpoint sauvegardé : {pt_path}")
    return model, history

def get_top_activating_examples(
    feature_idx: int,
    acts: torch.Tensor,
    texts: List[str],
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    feat_acts = acts[:, feature_idx].float()
    k = min(top_k, len(texts))
    top_vals, top_idx = feat_acts.topk(k)
    return [
        (texts[i], top_vals[j].item())
        for j, i in enumerate(top_idx.tolist())
        if top_vals[j] > 1e-6
    ]

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