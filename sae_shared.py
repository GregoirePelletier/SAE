"""
sae_shared.py — Utilitaires partagés pour les pipelines SAE (v7).
"""

import gc
import glob
import hashlib
import json
import math
import os
import re
from collections import Counter
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix as sp_csr
from tqdm import tqdm

from src.data.preparation import (
    keyword_match,
    prepare_domain_dataset,
    split_into_phrases,
    load_and_clean_emails,
    url_match,
    is_expressive_or_support,
)
from src.analysis.metrics import (
    compute_metrics,
    compute_rho_sae,
    diff_features,
    compute_npmi,
    pool_embeddings_by_document,
    downstream_classification,
)
from src.storage.shards import save_activations_sharded, load_activations_mmap
from src.sae.frozen_core import ExtendedSAE, FrozenCoreResidualSAE
from src.sae.phrase_sae import (
    PhraseLevelSAE,
    extract_f2llm_embeddings,
    encode_documents_with_phrase_sae,
    load_or_train_sae,
    compute_sae_metrics,
)

# ══════════════════════════════════════════════════════════════════════════════
# MOTS-CLÉS & PATTERNS DE LIENS
# ══════════════════════════════════════════════════════════════════════════════

from src.data.keywords import (
    DOMAIN_KEYWORDS_MAP,
    DOMAIN_URL_MAP,
    ENERGY_KEYWORDS,
    ENERGY_URL_PATTERNS,
    SPORTS_KEYWORDS,
    SPORTS_URL_PATTERNS,
    SUPPORT_KEYWORDS,
    SUPPORT_URL_PATTERNS,
)

SHARD_SIZE_GB = float(os.environ.get("SHARD_SIZE_GB", "4.0"))


# ══════════════════════════════════════════════════════════════════════════════
# FILTRAGE SÉMANTIQUE
# ══════════════════════════════════════════════════════════════════════════════

# Re-exported semantic filtering helpers imported from src.data.preparation


# ══════════════════════════════════════════════════════════════════════════════
# SURBRILLANCE DES TOKENS
# ══════════════════════════════════════════════════════════════════════════════

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
    sparse_col = token_data["token_sae_acts"][:, feature_idx]
    acts = np.asarray(sparse_col.todense()).flatten()
    return highlight_activations_as_string(tokens, acts, left, right)


def build_highlighted_examples(
    feature_idx: int,
    per_doc_token_data: List[Dict[str, Any]],
    doc_acts: torch.Tensor,
    top_k: int = 8,
) -> List[str]:
    feat_acts = doc_acts[:, feature_idx].float()
    k = min(top_k, feat_acts.shape[0])
    top_vals, top_idx = feat_acts.topk(k)
    results = []
    for j, i in enumerate(top_idx.tolist()):
        if top_vals[j].item() < 1e-6:
            break
        results.append(highlight_doc_for_feature(per_doc_token_data[i], feature_idx))
    return results


def build_negative_examples(
    feature_idx: int,
    per_doc_token_data: List[Dict[str, Any]],
    doc_acts: torch.Tensor,
    top_k: int = 8,
) -> List[str]:
    feat_acts = doc_acts[:, feature_idx].float()
    zero_mask = feat_acts == 0
    zero_indices = torch.where(zero_mask)[0]
    if len(zero_indices) == 0:
        _, bottom_idx = feat_acts.topk(top_k, largest=False)
        zero_indices = bottom_idx
    k = min(top_k, len(zero_indices))
    selected = zero_indices[torch.randperm(len(zero_indices))[:k]]
    results = []
    for i in selected.tolist():
        td = per_doc_token_data[i]
        plain_text = "".join(td["token_strings"]).replace(" ", " ").replace("Ġ", " ").replace("▁", " ")
        results.append(plain_text)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE COMPARATIVE (DIFFING)
# ══════════════════════════════════════════════════════════════════════════════

def diff_features(
    acts1: torch.Tensor,
    acts2: torch.Tensor,
    feature_labels: Dict[int, str] = None,
    metric: str = "absolute",
    min_coverage: float = 0.0,
    max_coverage: float = 1.0,
) -> "pd.DataFrame":
    def _binarize(acts):
        return (acts.float().detach().cpu() > 1e-6).numpy()

    fa1 = _binarize(acts1)
    fa2 = _binarize(acts2)

    freq1 = fa1.mean(axis=0)
    freq2 = fa2.mean(axis=0)

    min_freq = np.minimum(freq1, freq2)
    max_freq = np.maximum(freq1, freq2)
    mask = (min_freq < min_coverage) | (max_freq > max_coverage)
    f1, f2 = freq1.copy(), freq2.copy()
    f1[mask] = -1.0
    f2[mask] = -1.0

    if metric == "absolute":
        diff = f1 - f2
    elif metric == "relative":
        denom = np.maximum(f1, f2)
        denom[denom == 0] = 1.0
        diff = f1 / denom - f2 / denom
    else:
        raise ValueError(f"Métrique non supportée : {metric}")

    n_features = freq1.shape[0]
    labels = [feature_labels.get(i, f"Feature #{i}") for i in range(n_features)] if feature_labels else [""] * n_features

    df = pd.DataFrame({
        "feature_id": np.arange(n_features),
        "feature_label": labels,
        "freq_A": f1,
        "freq_B": f2,
        "frequency_difference": diff,
    })
    df = df[df["frequency_difference"] != -1.0]
    df = df.reindex(df["frequency_difference"].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# NPMI
# ══════════════════════════════════════════════════════════════════════════════

def compute_npmi(acts: torch.Tensor, threshold: float = 1e-6) -> sp_csr:
    X = (acts.float().detach().cpu() > threshold).numpy().astype(np.int32)
    n_samples = X.shape[0]
    X_sp = sp_csr(X.T)
    cooc = X_sp @ X_sp.T
    cooc.eliminate_zeros()
    rows, cols = cooc.nonzero()
    cooc_data = cooc.data
    feat_counts = np.asarray(X_sp.sum(axis=1)).flatten()
    eps = 1e-10
    P_x = (feat_counts + eps) / n_samples
    P_xy = (cooc_data + eps) / n_samples
    P_x_vals = P_x[rows]
    P_y_vals = P_x[cols]
    pmi = np.log(P_xy / (P_x_vals * P_y_vals))
    log_P_xy = np.log(P_xy)
    npmi_vals = np.where(np.abs(log_P_xy) < eps, 1.0, pmi / (-log_P_xy))
    npmi_vals = np.nan_to_num(npmi_vals, nan=0.0, posinf=1.0, neginf=-1.0)
    F = X_sp.shape[0]
    npmi_sparse = sp_csr((npmi_vals, (rows, cols)), shape=(F, F))
    npmi_sparse.setdiag(1.0)
    return npmi_sparse


# ══════════════════════════════════════════════════════════════════════════════
# STEERING
# ══════════════════════════════════════════════════════════════════════════════

def steer_activations(
    doc_acts: torch.Tensor,
    amplifications: Dict[int, float],
) -> torch.Tensor:
    steered = doc_acts.clone()
    for f_idx, mult in amplifications.items():
        steered[:, f_idx] = steered[:, f_idx] * mult
    return steered


def steer_and_decode(
    doc_acts: torch.Tensor,
    amplifications: Dict[int, float],
    sae: nn.Module,
) -> torch.Tensor:
    steered = steer_activations(doc_acts, amplifications)
    device = next(sae.parameters()).device
    with torch.no_grad():
        return sae.decode(steered.to(device).to(torch.bfloat16))


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION DE LABELS (HORS-LIGNE)
# ══════════════════════════════════════════════════════════════════════════════

def score_feature_label(
    feature_idx: int,
    proposed_label: str,
    pos_examples: List[str],
    neg_examples: List[str],
    offline_mode: bool = True,
) -> float:
    if not proposed_label:
        return 0.0
    label_words = [w.lower() for w in re.findall(r'\w+', proposed_label) if len(w) > 3]
    if not label_words:
        return 0.5
    pos_score, neg_score = 0, 0
    for word in label_words:
        for ex in pos_examples:
            active_spans = re.findall(r'<<([^>]+)>>', ex.lower())
            for span in active_spans:
                if word in span:
                    pos_score += 2.0
                elif word in ex.lower():
                    pos_score += 0.5
        for ex in neg_examples:
            if word in ex.lower():
                neg_score += 1.0
    raw_score = (pos_score / (len(pos_examples) + 1e-8)) - (neg_score / (len(neg_examples) + 1e-8))
    return float(1.0 / (1.0 + math.exp(-raw_score)))


# ══════════════════════════════════════════════════════════════════════════════
# PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

# Re-exported data preparation helpers imported from src.data.preparation


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT MAILS TSV
# ══════════════════════════════════════════════════════════════════════════════

# Re-exported email loading helper imported from src.data.preparation


# ══════════════════════════════════════════════════════════════════════════════
# SHARDED ACTIVATION I/O
# ══════════════════════════════════════════════════════════════════════════════

# Re-exported sharded activation I/O helpers imported from src.storage.shards


# ══════════════════════════════════════════════════════════════════════════════
# BATCH TOPK
# ══════════════════════════════════════════════════════════════════════════════

def batch_topk_encode(pre_acts: torch.Tensor, k: int, training: bool) -> torch.Tensor:
    if training:
        flat = pre_acts.reshape(-1)
        total_k = k * pre_acts.shape[0]
        if total_k >= flat.numel():
            return pre_acts
        threshold = flat.kthvalue(flat.numel() - total_k + 1).values
        return pre_acts * (pre_acts >= threshold).to(pre_acts.dtype)
    else:
        k_clamp = min(k, pre_acts.shape[-1])
        topk_vals, topk_idx = pre_acts.topk(k_clamp, dim=-1)
        return torch.zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULING & TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _make_lr_schedule(total_steps: int, warmup_steps: int):
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_fn


def train_model(
    model: nn.Module,
    acts_train: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int = 2048,
    log_every: int = 100,
    model_name: str = "model",
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print(f"  [sae_shared] Aucun paramètre entraînable pour {model_name}.")
        return model, {"nmse": [], "l0": [], "dead_frac": []}

    N = acts_train.shape[0]
    steps_per_epoch = N // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(1000, total_steps // 10)

    optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.0, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_schedule(total_steps, warmup_steps)
    )
    model.train()
    history = {"nmse": [], "l0": [], "dead_frac": [], "step": []}
    step = 0

    for epoch in range(epochs):
        perm = torch.randperm(N)
        for start in range(0, N - batch_size, batch_size):
            idx = perm[start: start + batch_size]
            idx_sorted, sort_order = idx.sort()
            b = acts_train[idx_sorted].to(device).to(torch.bfloat16)
            b = b[sort_order.argsort()]
            out = model(b)
            loss = out["normalized_mse"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            if hasattr(model, "normalize_decoder"):
                model.normalize_decoder()
            if step % log_every == 0:
                l0_key = next((k for k in ["l0_extra", "l0_res", "l0"] if k in out), None)
                l0_v = out[l0_key].item() if l0_key else 0.0
                dead_v = out.get("dead_frac", torch.tensor(0.0)).item()
                nmse_v = out["normalized_mse"].item()
                history["nmse"].append(nmse_v)
                history["l0"].append(l0_v)
                history["dead_frac"].append(dead_v)
                history["step"].append(step)
                if step % (log_every * 10) == 0:
                    print(
                        f"  [{model_name}] ep={epoch:2d} step={step:5d} | "
                        f"NMSE={nmse_v:.5f} | L0={l0_v:.1f} | "
                        f"dead={dead_v:.3f} | lr={scheduler.get_last_lr()[0]:.2e}"
                    )
            step += 1

    return model, history


def load_or_train(
    model: nn.Module,
    model_name: str,
    acts_train: torch.Tensor,
    epochs: int,
    lr: float,
    save_dir: str = "./results/",
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    pt_path = os.path.join(save_dir, f"{model_name.lower()}_state.pt")
    hist_path = os.path.join(save_dir, f"{model_name.lower()}_hist.json")
    if os.path.exists(pt_path) and os.path.exists(hist_path):
        print(f"  [sae_shared] Restauration checkpoint : {pt_path}")
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        with open(hist_path) as f:
            history = json.load(f)
    else:
        model, history = train_model(model, acts_train, epochs=epochs, lr=lr,
                                     model_name=model_name, device=device)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, pt_path)
        with open(hist_path, "w") as f:
            json.dump(history, f)
        print(f"  [sae_shared] Checkpoint sauvegardé : {pt_path}")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    model: nn.Module,
    acts: torch.Tensor,
    batch_size: int = 4096,
    is_saelens: bool = False,
    var_sample: int = 65536,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    FVE = 1 - MSE(x, x_hat) / Var(x)
    NMSE = MSE(x, x_hat) / Var(x)   [= 1 - FVE, exposé séparément]
    """
    if isinstance(model, nn.Module):
        model.eval()

    model_dtype = None
    if isinstance(model, nn.Module):
        for p in model.parameters():
            model_dtype = p.dtype
            break
    if model_dtype is None:
        model_dtype = torch.float32

    mse_acc, l0_acc, n_tok = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, acts.shape[0], batch_size):
            b = acts[i: i + batch_size].to(device).to(model_dtype)
            if is_saelens:
                feat = model.encode(b)
                recon = model.decode(feat)
            else:
                model_d_sae = getattr(model, "d_sae", None)
                if model_d_sae is None and hasattr(model, "cfg"):
                    model_d_sae = getattr(model.cfg, "d_sae", None)
                if model_d_sae is None and hasattr(model, "core_sae"):
                    model_d_sae = model.core_sae.cfg.d_sae + getattr(model, "d_extra", 0)

                if (model_d_sae is not None and b.shape[-1] == model_d_sae and hasattr(model, "decode")
                        and hasattr(model, "encode")):
                    feat = b
                    x_hat = model.decode(b)
                    recon = model.encode(x_hat)
                else:
                    out = model(b)
                    recon = out["sae_out"]
                    feat = out["feature_acts"]
            mse_acc += (b - recon).pow(2).sum(dim=-1).sum().item()
            l0_acc += (feat.abs() > 1e-6).float().sum(dim=-1).sum().item()
            n_tok += b.shape[0]

    mse = mse_acc / n_tok
    l0 = l0_acc / n_tok
    n_var = min(var_sample, acts.shape[0])
    idx_v = torch.linspace(0, acts.shape[0] - 1, n_var).long()
    var_total = acts[idx_v].float().var(dim=0).sum().item()
    fve = 1.0 - (mse / (var_total + 1e-8))
    nmse = mse / (var_total + 1e-8)

    if isinstance(model, nn.Module):
        model.train()
    return {"MSE": mse, "NMSE": nmse, "FVE": fve, "L0": l0}


def compute_extra_feature_stats(
    model: nn.Module,
    acts: torch.Tensor,
    batch_size: int = 4096,
    device: str = "cuda",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    model.eval()
    freq_chunks = []
    with torch.no_grad():
        for i in range(0, acts.shape[0], batch_size):
            b = acts[i: i + batch_size].to(device).to(torch.bfloat16)
            out = model(b)
            fa = next(out[k] for k in ["extra_acts", "res_acts", "feature_acts"] if k in out)
            freq_chunks.append((fa.abs() > 1e-6).float().mean(dim=0).cpu())
    freq = torch.stack(freq_chunks).mean(dim=0).detach().cpu().numpy()
    return freq, {
        "total": len(freq),
        "dead": int((freq < 1e-6).sum()),
        "dead_pct": float((freq < 1e-6).mean() * 100),
        "active_gt1pct": int((freq > 0.01).sum()),
        "freq_max": float(freq.max()),
        "freq_mean": float(freq.mean()),
    }


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


# ══════════════════════════════════════════════════════════════════════════════
# ρ_SAE — Préservation du ranking cosinus
# ══════════════════════════════════════════════════════════════════════════════

def compute_rho_sae(
    model: nn.Module,
    acts: torch.Tensor,
    n_sample: int = 500,
    batch_size: int = 512,
    is_saelens: bool = False,
    device: str = "cuda",
) -> float:
    """
    ρ_SAE : corrélation de rang de Spearman entre similarités cosinus
    pairwise sur l'échantillon original vs reconstruit.

    ρ_SAE = Spearman({ cos(x_i, x_j) }_{i<j},  { cos(x̂_i, x̂_j) }_{i<j})

    Mesure la préservation de la topologie sémantique par le SAE.
    O(n²) en mémoire — limité à n_sample=500 par défaut.
    """
    from scipy.stats import spearmanr

    n = min(n_sample, acts.shape[0])
    idx = torch.randperm(acts.shape[0])[:n]
    sub = acts[idx].to(device).to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        if is_saelens:
            recon = model.decode(model.encode(sub)).float()
        else:
            sub_for_model = sub
            model_d_sae = getattr(model, "d_sae", None)
            if model_d_sae is None and hasattr(model, "core_sae"):
                model_d_sae = model.core_sae.cfg.d_sae + getattr(model, "d_extra", 0)

            model_dtype = next(model.parameters()).dtype
            if sub_for_model.dtype != model_dtype:
                sub_for_model = sub_for_model.to(model_dtype)

            if sub_for_model.shape[-1] == model_d_sae and hasattr(model, "decode") and hasattr(model, "encode"):
                x_hat = model.decode(sub_for_model)
                recon = model.encode(x_hat).float()
            else:
                out = model(sub_for_model)
                recon = out["sae_out"].float()

    sub = sub.float()

    orig_norm = F.normalize(sub, dim=1)
    recon_norm = F.normalize(recon, dim=1)
    cos_orig = (orig_norm @ orig_norm.T).cpu().numpy()
    cos_recon = (recon_norm @ recon_norm.T).cpu().numpy()

    triu = np.triu_indices(n, k=1)
    rho, _ = spearmanr(cos_orig[triu], cos_recon[triu])
    return float(rho)


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING POOLING — agrégation par document
# ══════════════════════════════════════════════════════════════════════════════

def pool_embeddings_by_document(
    phrase_embeddings: torch.Tensor,
    phrase_to_doc: np.ndarray,
    n_docs: int = None,
) -> torch.Tensor:
    """
    Agrège des embeddings au niveau phrase vers le niveau document via max-pooling.
    
    Args:
        phrase_embeddings: [n_phrases, d] tensor
        phrase_to_doc: [n_phrases] int array, chaque entrée est l'indice du document
        n_docs: nombre de documents (optionnel, inféré de phrase_to_doc si omis)
    
    Returns:
        Document-level embeddings [n_docs, d] via max-pooling
    
    Raises:
        ValueError si phrase_to_doc.shape[0] != phrase_embeddings.shape[0]
    """
    if phrase_embeddings.shape[0] != len(phrase_to_doc):
        raise ValueError(
            f"Mismatch: phrase_embeddings has {phrase_embeddings.shape[0]} rows "
            f"but phrase_to_doc has {len(phrase_to_doc)} entries"
        )
    
    if n_docs is None:
        n_docs = int(phrase_to_doc.max()) + 1
    
    d = phrase_embeddings.shape[1]
    doc_emb = torch.full((n_docs, d), float('-inf'), dtype=phrase_embeddings.dtype)
    
    for phrase_idx, doc_idx in enumerate(phrase_to_doc):
        doc_idx = int(doc_idx)
        doc_emb[doc_idx] = torch.max(doc_emb[doc_idx], phrase_embeddings[phrase_idx])
    
    return doc_emb


# ══════════════════════════════════════════════════════════════════════════════
# DOWNSTREAM CLASSIFICATION — sonde linéaire (perspective slide 19)
# ══════════════════════════════════════════════════════════════════════════════

def downstream_classification(
    acts_by_label: Dict[str, torch.Tensor],
    raw_emb_by_label: Dict[str, torch.Tensor] = None,
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Sonde logistique 5-fold sur activations SAE vs embeddings bruts.
    Retourne acc_sae et (si fourni) acc_raw pour comparaison.

    Cela permet de mesurer si la représentation SAE est au least aussi
    discriminante que l'embedding original (critère d'auditabilité).
    
    IMPORTANT: acts_by_label et raw_emb_by_label doivent avoir le même nombre
    d'échantillons par label pour éviter les erreurs de validation sklearn.
    Utiliser pool_embeddings_by_document() si nécessaire pour aligner les dimensions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    labels_list = sorted(acts_by_label.keys())
    
    # Validation des tailles
    sae_sizes = {l: len(acts_by_label[l]) for l in labels_list}
    if raw_emb_by_label is not None:
        raw_sizes = {l: len(raw_emb_by_label[l]) for l in labels_list}
        for label in labels_list:
            if sae_sizes[label] != raw_sizes[label]:
                raise ValueError(
                    f"[downstream_classification] Mismatch for label '{label}': "
                    f"acts_by_label['{label}'] has {sae_sizes[label]} samples, "
                    f"raw_emb_by_label['{label}'] has {raw_sizes[label]} samples. "
                    f"Use pool_embeddings_by_document() to align phrase-level embeddings to document-level."
                )
    
    X_sae = torch.cat([acts_by_label[l] for l in labels_list]).float().detach().cpu().numpy()
    y = np.concatenate([
        np.full(len(acts_by_label[l]), i, dtype=int)
        for i, l in enumerate(labels_list)
    ])

    clf = make_pipeline(
        StandardScaler(with_mean=False, copy=True),
        LogisticRegression(max_iter=1500, C=1.0, random_state=seed, solver="saga")
    )
    acc_sae = float(cross_val_score(clf, X_sae, y, cv=cv, scoring="accuracy").mean())
    results = {"labels": labels_list, "acc_sae": acc_sae}

    if raw_emb_by_label is not None:
        X_raw = torch.cat([raw_emb_by_label[l] for l in labels_list]).float().detach().cpu().numpy()
        clf_raw = make_pipeline(
            StandardScaler(copy=True),
            LogisticRegression(max_iter=1500, C=1.0, random_state=seed, solver="saga")
        )
        acc_raw = float(cross_val_score(clf_raw, X_raw, y, cv=cv, scoring="accuracy").mean())
        results["acc_raw"] = acc_raw
        results["delta_acc"] = acc_sae - acc_raw

    print(f"  [Downstream] acc_SAE={acc_sae:.4f}" +
          (f" | acc_raw={results.get('acc_raw', float('nan')):.4f}" if raw_emb_by_label else ""))
    return results
