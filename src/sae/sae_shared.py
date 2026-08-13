"""
sae_shared.py — Harnais d'entraînement ExtendedSAE + steering + ré-exports
partagés entre les deux pipelines.
"""

import os
import sys
import math
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "..", ".."))  # src/sae/ -> racine du repo
# external/ vit à la racine du repo, pas sous src/sae/. interp_embed reste
# volontairement non peuplé (inspiration méthodologique seulement) ; ce chemin
# ne prend effet que si le submodule est un jour initialisé.
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
        sample_fineweb2_chunks,
        split_into_phrases,
        load_and_clean_emails,
        build_email_train_test_corpus,
        url_match,
    )
except ImportError:
    from preparation import (
        keyword_match,
        prepare_domain_dataset,
        sample_fineweb2_chunks,
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
        ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS, SUPPORT_URL_PATTERNS,
    )
except ImportError:
    from keywords import (
        ENERGY_KEYWORDS, SPORTS_KEYWORDS, SUPPORT_KEYWORDS,
        ENERGY_URL_PATTERNS, SPORTS_URL_PATTERNS, SUPPORT_URL_PATTERNS,
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
    Harnais d'entraînement et de restauration pour l'extension sémantique
    ExtendedSAE (Pipeline 1).
    """
    save_path = os.path.join(save_dir, f"{model_name}.pt")
    history_path = save_path.replace(".pt", "_history.json")
    if os.path.exists(save_path):
        print(f"  [sae_shared] Restauration du modèle {model_name} : {save_path}")
        ckpt = torch.load(save_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  [sae_shared] Checkpoint sans certains buffers (θ/AuxK) — "
                  f"fallback TopK per-sample en eval. Manquants: {missing}")
        return model, ckpt.get("history", {})

    # Split de validation tenu à l'écart du gradient -- compute_sae_metrics
    # reste une métrique post-hoc calculée après coup sur tout le corpus, ce
    # split ajoute un signal de sur-apprentissage pendant l'entraînement
    # lui-même. acts_train peut être un tenseur memmap disque (cf.
    # open_mmap_reservoir, saev5.py) : indexer par un sous-ensemble d'indices
    # (Subset) reste paginé à la demande, jamais matérialisé en RAM.
    n_total = acts_train.shape[0]
    n_val = min(8192, max(1, n_total // 20))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(0))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    acts_val = acts_train[val_idx]

    print(f"  [sae_shared] Entraînement de {model_name} sur {len(train_idx)} tokens résidus "
          f"({n_val} tenus à l'écart pour validation)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from torch.utils.data import TensorDataset, DataLoader, Subset
    train_dataset = Subset(TensorDataset(acts_train), train_idx)
    loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # Historique PAR STEP (pas par époque), aligné avec la convention du
    # Pipeline 2 (phrase_sae.py::load_or_train_sae) -- permet de tracer des
    # courbes de perte, pas seulement des moyennes d'époque.
    history = {"epoch": [], "step": [], "loss": [], "l0": [], "dead_frac": [], "aux_loss": [],
               "val_epoch": [], "val_loss": []}
    step = 0

    for epoch in range(epochs):
        model.train()
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

            history["loss"].append(loss.item())
            history["l0"].append(out.get("l0_extra", out.get("l0", torch.tensor(0.0))).item())
            history["dead_frac"].append(out.get("dead_frac", torch.tensor(0.0)).item())
            history["aux_loss"].append(float(out.get("aux_loss", 0.0)))
            history["epoch"].append(epoch)
            history["step"].append(step)
            step += 1

        model.eval()
        with torch.no_grad():
            vb = acts_val.to(device).to(torch.bfloat16)
            val_loss = model(vb)["loss"].item()
        history["val_epoch"].append(epoch)
        history["val_loss"].append(val_loss)

        print(
            f"  Epoch {epoch+1:02d}/{epochs} | Loss={history['loss'][-1]:.4f} | "
            f"ValLoss={val_loss:.4f} | L0={history['l0'][-1]:.1f} | "
            f"dead={history['dead_frac'][-1]:.3f} | aux={history['aux_loss'][-1]:.4f}"
        )

    ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {"epochs": epochs, "lr": lr},
        "history": history,
    }
    torch.save(ckpt, save_path)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    return model, history