"""
sae_shared.py — v10. Harnais d'entraînement ExtendedSAE + steering + ré-exports.
Purgé (audit) : diff_features → cooccurrence.corpus_diff_stats ; compute_npmi →
cooccurrence.compute_npmi ; highlight_activations_as_string → SAEDashboard ;
pool_embeddings_by_document → alias activations.scatter_maxpool ;
train_extended_sae_one_epoch supprimée (code mort sans AuxK ni projection).
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
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "..", ".."))  # src/sae/ -> racine du repo
# external/ vit à la racine du repo, pas sous src/sae/ — chemin corrigé (bug historique
# qui pointait vers src/sae/external/..., toujours inexistant, masqué par le try/except
# ci-dessous). interp_embed reste volontairement non peuplé (Context.md : inspiration
# seulement) ; ce chemin ne prend effet que si le submodule est un jour initialisé.
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "interp_embed"))
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "sae-lens"))
sys.path.insert(0, os.path.join(ROOT_DIR, "..", "data"))
sys.path.insert(0, os.path.join(ROOT_DIR, "..", "analysis"))

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
        build_email_train_test_corpus,
        url_match,
    )
except ImportError:
    from preparation import (
        keyword_match,
        prepare_domain_dataset,
        split_into_phrases,
        load_and_clean_emails,
        build_email_train_test_corpus,
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
    from src.sae.frozen_core import ExtendedSAE, FrozenCoreResidualSAE, FrozenDecoderExtendedSAE
except ImportError:
    from frozen_core import ExtendedSAE, FrozenCoreResidualSAE, FrozenDecoderExtendedSAE

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


# ─── POOLING : implémentation unique dans src/analysis/activations ───

try:
    from src.analysis.activations import scatter_maxpool
except ImportError:
    from activations import scatter_maxpool


def pool_embeddings_by_document(phrase_embeddings, phrase_to_doc, n_docs=None):
    """Alias de compat — délègue à activations.scatter_maxpool (implémentation unique)."""
    if n_docs is None:
        n_docs = int(phrase_to_doc.max()) + 1
    idx = torch.from_numpy(phrase_to_doc).to(phrase_embeddings.device)
    return scatter_maxpool(phrase_embeddings, idx, n_docs)


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
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  [sae_shared] Checkpoint v8 (sans θ/AuxK) — fallback TopK per-sample en eval. "
                  f"Manquants: {missing}")
        return model, ckpt.get("history", {})

    print(f"  [sae_shared] Entraînement de {model_name} sur {acts_train.shape[0]} tokens...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(acts_train)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    history = {"epoch": [], "loss": [], "l0": [], "dead_frac": [], "aux_loss": []}
    
    for epoch in range(epochs):
        model.train()
        loss_acc, l0_acc, dead_acc, n_samples = 0.0, 0.0, 0.0, 0
        
        aux_acc = 0.0
        for batch in loader:
            b = batch[0].to(device).to(torch.bfloat16)
            optimizer.zero_grad()
            out = model(b)
            loss = out["loss"]
            loss.backward()

            if hasattr(model, "normalize_decoder"):
                model.normalize_decoder()   # projette le gradient parallèle AVANT le step
            optimizer.step()
            if hasattr(model, "normalize_decoder"):
                model.normalize_decoder()   # renormalise après le step

            n_b = b.shape[0]
            loss_acc += loss.item() * n_b
            l0_acc += out.get("l0_extra", out.get("l0", torch.tensor(0.0))).item() * n_b
            dead_acc += out.get("dead_frac", torch.tensor(0.0)).item() * n_b
            aux_acc += float(out.get("aux_loss", 0.0)) * n_b
            n_samples += n_b
            
        epoch_loss = loss_acc / n_samples
        epoch_l0 = l0_acc / n_samples
        epoch_dead = dead_acc / n_samples
        
        history["loss"].append(epoch_loss)
        history["l0"].append(epoch_l0)
        history["dead_frac"].append(epoch_dead)
        history["epoch"].append(epoch)
        history["aux_loss"].append(aux_acc / n_samples)

        print(
            f"  Epoch {epoch+1:02d}/{epochs} | Loss={epoch_loss:.4f} | "
            f"L0={epoch_l0:.1f} | dead={epoch_dead:.3f} | aux={aux_acc/n_samples:.4f}"
        )
        
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {"epochs": epochs, "lr": lr},
        "history": history,
    }
    torch.save(ckpt, save_path)
    return model, history