"""
sae_shared.py — Utilitaires partagés pour les pipelines SAE (v8).
Vectorisation complète, correctifs NPMI, pooling d'embeddings et entraînement de l'adaptation.
"""

import os
import sys
import math
import re  # FIX : Ajout de l'import re manquant
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

# Imports résilients gérant la structure package "src" et la structure de dossier plate
try:
    from src.data.preparation import (
        keyword_match,
        prepare_domain_dataset,
        split_into_phrases,
        load_and_clean_emails,
        url_match,
    )
except ImportError:
    from preparation import (
        keyword_match,
        prepare_domain_dataset,
        split_into_phrases,
        load_and_clean_emails,
        url_match,
    )

try:
    from src.analysis.metrics import (
        compute_metrics,
        compute_rho_sae,
        downstream_classification,
    )
except ImportError:
    from metrics import (
        compute_metrics,
        compute_rho_sae,
        downstream_classification,
    )

try:
    from src.sae.frozen_core import ExtendedSAE, FrozenCoreResidualSAE
except ImportError:
    from frozen_core import ExtendedSAE, FrozenCoreResidualSAE

try:
    from src.sae.phrase_sae import (
        PhraseLevelSAE,
        extract_f2llm_embeddings,
        encode_documents_with_phrase_sae,
        load_or_train_sae,
        compute_sae_metrics,
    )
except ImportError:
    from phrase_sae import (
        PhraseLevelSAE,
        extract_f2llm_embeddings,
        encode_documents_with_phrase_sae,
        load_or_train_sae,
        compute_sae_metrics,
    )

try:
    from src.data.keywords import (
        ENERGY_KEYWORDS, SPORTS_KEYWORDS, SUPPORT_KEYWORDS,
        ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS, SUPPORT_URL_PATTERNS
    )
except ImportError:
    from keywords import (
        ENERGY_KEYWORDS, SPORTS_KEYWORDS, SUPPORT_KEYWORDS,
        ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS, SUPPORT_URL_PATTERNS
    )


# ─── DIFFING ───

def diff_features(
    acts_a: torch.Tensor,
    acts_b: torch.Tensor,
    feature_labels: Optional[Dict[int, str]] = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Calcule l'écart de fréquence d'activation symétrique entre deux distributions."""
    freq_a = (acts_a > 1e-6).float().mean(dim=0).cpu().numpy()
    freq_b = (acts_b > 1e-6).float().mean(dim=0).cpu().numpy()
    diff = freq_a - freq_b
    ratio = (freq_a + eps) / (freq_b + eps)
    
    df = pd.DataFrame({
        "feature_id": np.arange(len(diff)),
        "freq_A": freq_a,
        "freq_B": freq_b,
        "frequency_difference": diff,
        "frequency_ratio": ratio,
    })
    
    if feature_labels:
        df["feature_label"] = df["feature_id"].apply(lambda idx: feature_labels.get(int(idx), f"F{idx}"))
    else:
        df["feature_label"] = df["feature_id"].apply(lambda idx: f"F{idx}")
        
    return df.sort_values(by="frequency_difference", key=abs, ascending=False).reset_index(drop=True)


# ─── NPMI ───

def compute_npmi(doc_acts: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Calcule la matrice NPMI des features de manière vectorisée."""
    n_samples = doc_acts.shape[0]
    bin_acts = (doc_acts > 1e-6).float()
    
    cooc = bin_acts.T @ bin_acts
    p_ij = cooc / n_samples
    
    p_i = bin_acts.sum(dim=0) / n_samples
    p_i_mat = p_i.unsqueeze(1)
    p_j_mat = p_i.unsqueeze(0)
    
    pmi = torch.log((p_ij + eps) / (p_i_mat * p_j_mat + eps))
    log_p_ij = torch.log(p_ij + eps)
    npmi = pmi / (-log_p_ij)
    
    npmi = torch.where(cooc > 0, npmi, torch.zeros_like(npmi))
    npmi.fill_diagonal_(1.0)
    return npmi


# ─── UTILS VISU ───

def highlight_activations_as_string(
    tokens: List[str],
    acts: np.ndarray,
    left_window: int = 40,
    right_window: int = 15,
) -> str:
    if len(tokens) == 0:
        return ""
    target_idx = int(acts.argmax())
    if acts[target_idx] <= 1e-6:
        return ""
    
    start_idx = max(0, target_idx - left_window)
    tokens_window = tokens[start_idx:target_idx + 1]
    
    context_str = ""
    for idx, tok in enumerate(tokens_window):
        is_target = (idx == len(tokens_window) - 1)
        clean_tok = tok.replace("Ġ", " ").replace(" ", " ")
        
        if is_target:
            context_str += f" <<{clean_tok.strip()}>>"
        else:
            if clean_tok.startswith(" ") or tok.startswith("Ġ"):
                context_str += " " + clean_tok.strip()
            else:
                context_str += clean_tok
                
    return re.sub(r"\s+", " ", context_str).strip()


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


# ─── VECTORIZED POOLING ───

def pool_embeddings_by_document(
    phrase_embeddings: torch.Tensor,
    phrase_to_doc: np.ndarray,
    n_docs: int = None,
) -> torch.Tensor:
    """Pools phrase embeddings using vectorized scatter max pooling."""
    if phrase_embeddings.shape[0] != len(phrase_to_doc):
        raise ValueError("Incohérence de taille entre phrases et mappage de documents.")

    if n_docs is None:
        n_docs = int(phrase_to_doc.max()) + 1

    device = phrase_embeddings.device
    d = phrase_embeddings.shape[1]
    phrase_to_doc_t = torch.from_numpy(phrase_to_doc).to(device).long()
    
    doc_emb = torch.full((n_docs, d), float('-inf'), dtype=phrase_embeddings.dtype, device=device)
    doc_emb.scatter_reduce_(0, phrase_to_doc_t.unsqueeze(-1).expand(-1, d), phrase_embeddings, reduce="amax", include_self=False)
    doc_emb = torch.where(doc_emb == float('-inf'), torch.zeros_like(doc_emb), doc_emb)
    return doc_emb


# ─── HARNAIS D'ENTRAINEMENT ET CHARGEMENT DU FROZEN-CORE EXTENDED SAE ───

def load_or_train_extended_sae(
    model: nn.Module,
    model_name: str,
    acts_train: torch.Tensor,
    epochs: int,
    lr: float,
    save_dir: str,
    device: str,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Harnais d'entraînement et de restauration robuste pour l'extension sémantique ExtendedSAE.
    Résout la dette de conception de load_or_train pour le Pipeline 1.
    """
    save_path = os.path.join(save_dir, f"{model_name}.pt")
    if os.path.exists(save_path):
        print(f"  [sae_shared] Restauration du modèle {model_name} : {save_path}")
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt.get("history", {})

    print(f"  [sae_shared] Entraînement de {model_name} sur {acts_train.shape[0]} tokens...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(acts_train)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    history = {"epoch": [], "loss": [], "l0": [], "dead_frac": []}
    
    for epoch in range(epochs):
        model.train()
        loss_acc, l0_acc, dead_acc, n_samples = 0.0, 0.0, 0.0, 0
        
        for batch in loader:
            b = batch[0].to(device).to(torch.bfloat16)
            optimizer.zero_grad()
            out = model(b)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            
            if hasattr(model, "normalize_decoder"):
                model.normalize_decoder()
                
            n_b = b.shape[0]
            loss_acc += loss.item() * n_b
            l0_acc += out.get("l0_extra", out.get("l0", torch.tensor(0.0))).item() * n_b
            dead_acc += out.get("dead_frac", torch.tensor(0.0)).item() * n_b
            n_samples += n_b
            
        epoch_loss = loss_acc / n_samples
        epoch_l0 = l0_acc / n_samples
        epoch_dead = dead_acc / n_samples
        
        history["loss"].append(epoch_loss)
        history["l0"].append(epoch_l0)
        history["dead_frac"].append(epoch_dead)
        history["epoch"].append(epoch)
        
        print(
            f"  Epoch {epoch+1:02d}/{epochs} | Loss={epoch_loss:.4f} | "
            f"L0={epoch_l0:.1f} | dead={epoch_dead:.3f}"
        )
        
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {"epochs": epochs, "lr": lr},
        "history": history,
    }
    torch.save(ckpt, save_path)
    return model, history


def train_extended_sae_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    model.train()
    loss_acc, l0_acc, n_samples = 0.0, 0.0, 0
    
    for batch in train_loader:
        b = batch[0].to(device)
        optimizer.zero_grad()
        out = model(b)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        
        n_b = b.shape[0]
        loss_acc += loss.item() * n_b
        l0_acc += out["l0"].item() * n_b
        n_samples += n_b
        
    return {
        "loss": loss_acc / n_samples,
        "l0": l0_acc / n_samples,
    }