"""
sae_shared.py — Harnais d'entraînement SAEBoostResidualSAE + steering + ré-exports
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
        group_indices_by_doc,
        build_reencode_targets,
        is_filler_document,
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
        group_indices_by_doc,
        build_reencode_targets,
        is_filler_document,
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
    from src.sae.frozen_core import SAEBoostResidualSAE, FrozenCoreResidualSAE, FrozenDecoderExtendedSAE
except ImportError:
    from frozen_core import SAEBoostResidualSAE, FrozenCoreResidualSAE, FrozenDecoderExtendedSAE

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

def block_shuffle_indices(idx: torch.Tensor, block_size: int = 65536,
                           generator: torch.Generator = None) -> torch.Tensor:
    """Approxime `idx[torch.randperm(len(idx))]` avec une empreinte mémoire
    O(block_size) au lieu de O(len(idx)) -- shuffle l'ordre des blocs de
    `block_size` indices, puis shuffle intra-bloc, plutôt qu'une permutation
    globale. Sur `len(idx)=100_000_000`, `torch.randperm` alloue et régénère
    deux tenseurs int64 de 800 Mo à chaque appel (audit perf §2.4) ; ici
    l'allocation dominante (le tenseur de sortie) est de même taille que
    l'entrée, mais aucun tenseur intermédiaire de la taille de `idx` n'est
    créé pour le calculer, et l'accès reste par blocs contigus (meilleure
    localité mémoire qu'un index global aléatoire). Chaque élément de `idx`
    apparaît exactement une fois dans la sortie (propriété nécessaire et
    suffisante pour un epoch de SGD) -- ce n'est PAS une permutation uniforme
    sur toutes les `len(idx)!` possibles (l'ordre relatif intra-bloc est
    aléatoire, mais les blocs ne se mélangent jamais entre eux au-delà de
    leur propre réordonnancement), un compromis assumé pour ce gain mémoire."""
    n = idx.shape[0]
    if n <= block_size:
        return idx[torch.randperm(n, generator=generator)]
    n_blocks = (n + block_size - 1) // block_size
    block_order = torch.randperm(n_blocks, generator=generator).tolist()
    out = torch.empty_like(idx)
    pos = 0
    for b in block_order:
        start, end = b * block_size, min((b + 1) * block_size, n)
        block = idx[start:end]
        block_len = end - start
        out[pos:pos + block_len] = block[torch.randperm(block_len, generator=generator)]
        pos += block_len
    return out


def load_or_train_extended_sae(
    model: nn.Module,
    model_name: str,
    acts_train: torch.Tensor,
    epochs: int,
    lr: float,
    save_dir: str,
    device: str,
    batch_size: int = 1024,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Harnais d'entraînement et de restauration pour l'extension sémantique
    SAEBoostResidualSAE (Pipeline 1).
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

    # Indexation vectorisée (acts_train[idx_batch], un seul gather par batch) --
    # PAS DataLoader(Subset(TensorDataset(...))), qui appelle __getitem__ 1024 fois
    # (une fois par échantillon) avant collate : coûteux en soi, et catastrophique
    # si acts_train est un tenseur memmap disque (open_mmap_reservoir, saev5.py),
    # où chaque accès individuel déclenche sa propre lecture au lieu d'un seul
    # gather. Même pattern que Pipeline 2 (phrase_sae.py::load_or_train_sae).
    # batch_size affecte aussi le régime de sparsité BatchTopK (budget partagé
    # sur le batch, src/sae/batch.py) -- paramétré (au lieu d'une constante en
    # dur) pour permettre l'ablation avant tout changement de défaut
    # (AUDIT_SAE_2026-08.md §2.9, item 7).
    BATCH_SIZE = batch_size

    # Historique PAR STEP (pas par époque), aligné avec la convention du
    # Pipeline 2 (phrase_sae.py::load_or_train_sae) -- permet de tracer des
    # courbes de perte, pas seulement des moyennes d'époque.
    history = {"epoch": [], "step": [], "loss": [], "l0": [], "dead_frac": [], "aux_loss": [],
               "val_epoch": [], "val_loss": []}
    step = 0

    def _as_device_tensor(v):
        return v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=device)

    for epoch in range(epochs):
        model.train()
        # block_shuffle_indices plutôt que train_idx[torch.randperm(len(train_idx))] :
        # évite de réallouer un tenseur int64 de la taille de train_idx à chaque
        # époque (jusqu'à ~800 Mo sur un run à 100M tokens, cf. AUDIT_SAE_2026-08.md).
        epoch_perm = block_shuffle_indices(train_idx)
        # Métriques accumulées comme tenseurs GPU pendant l'époque, converties en
        # Python UNE SEULE FOIS à la fin (un seul sync CPU<->GPU par époque) plutôt
        # qu'à chaque step (audit perf §2.4 : 4x .item()/float() par step = 4x
        # cudaStreamSynchronize par step, alors qu'un step dure <1ms -- le step est
        # dominé par la synchro, pas le calcul). Valeurs identiques à l'ancien code,
        # seul le moment du sync change.
        step_losses, step_l0, step_dead, step_aux = [], [], [], []
        for i in range(0, len(epoch_perm), BATCH_SIZE):
            batch_idx = epoch_perm[i:i + BATCH_SIZE]
            b = acts_train[batch_idx].to(device).to(torch.bfloat16)
            optimizer.zero_grad()
            # return_feature_acts=False : ce harnais est scopé à SAEBoostResidualSAE/
            # FrozenCoreResidualSAE (docstring ci-dessus), dont forward() n'alloue
            # feature_acts ([B, d_core+d_extra] fp32) que si demandé -- jamais lu
            # dans cette boucle, coûteux à chaque step (audit perf §2.4).
            out = model(b, return_feature_acts=False) if hasattr(model, "core_sae") else model(b)
            loss = out["loss"]
            loss.backward()

            if hasattr(model, "normalize_decoder"):
                model.normalize_decoder()   # projette le gradient parallèle AVANT le step
            optimizer.step()
            if hasattr(model, "normalize_decoder"):
                model.normalize_decoder()   # renormalise après le step

            step_losses.append(loss.detach())
            step_l0.append(_as_device_tensor(out.get("l0_extra", out.get("l0", 0.0))))
            step_dead.append(_as_device_tensor(out.get("dead_frac", 0.0)))
            step_aux.append(_as_device_tensor(out.get("aux_loss", 0.0)))
            history["epoch"].append(epoch)
            history["step"].append(step)
            step += 1

        history["loss"].extend(torch.stack(step_losses).tolist())
        history["l0"].extend(torch.stack(step_l0).tolist())
        history["dead_frac"].extend(torch.stack(step_dead).tolist())
        history["aux_loss"].extend(torch.stack(step_aux).tolist())

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