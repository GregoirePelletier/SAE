"""
metrics.py — Métriques d'évaluation scientifique des auto-encodeurs (FVE, NMSE, L0, Spearman rank).
Alignement strict sur les formules mathématiques de SAELens et interp_embed.
"""

import torch
import numpy as np
from typing import Dict, Any


def compute_metrics(
    model: torch.nn.Module,
    acts: torch.Tensor,
    is_saelens: bool = False,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    FVE = 1 - NMSE, où NMSE = MSE(x, x̂) / Var(x).
    Note : variance normalisée sur l'ensemble du batch (mean sur tokens ET dimensions),
    ce qui est cohérent avec la définition scalaire de FVE utilisée dans SAEBench.
    """
    model.eval()
    acts_bf16 = acts.to(device).to(torch.bfloat16)

    with torch.no_grad():
        if is_saelens:
            recon = model.decode(model.encode(acts_bf16))
        else:
            out = model(acts_bf16)
            recon = out["sae_out"]

    acts_f = acts_bf16.float()
    recon_f = recon.float()

    mse = torch.mean((acts_f - recon_f).pow(2))
    variance = torch.mean((acts_f - acts_f.mean(dim=0, keepdim=True)).pow(2)) + 1e-8

    nmse = mse / variance
    fve = 1.0 - nmse

    if is_saelens:
        with torch.no_grad():
            codes = model.encode(acts_bf16)
        l0 = (codes.abs() > 1e-6).float().sum(dim=-1).mean().item()
    else:
        l0 = out.get("l0_extra", torch.tensor(0.0)).item()

    return {
        "FVE": float(fve.item()),
        "NMSE": float(nmse.item()),
        "L0": float(l0),
    }


def compute_rho_sae(
    model: torch.nn.Module,
    acts: torch.Tensor,
    n_sample: int = 500,
    is_saelens: bool = False,
    device: str = "cuda",
) -> float:
    """
    ρ_SAE = Spearman(cos_sim_originaux, cos_sim_reconstruits) sur n_sample² / 2 paires.
    Mesure la conservation de la topologie locale dans l'espace de représentation.
    Complexité : O(n²) en mémoire — n_sample = 500 → 125K paires, tractable.
    """
    from scipy.stats import spearmanr
    import torch.nn.functional as F

    n = min(n_sample, acts.shape[0])
    idx = torch.randperm(acts.shape[0])[:n]
    sub = acts[idx].to(device).to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        if is_saelens:
            recon = model.decode(model.encode(sub)).float()
        else:
            out = model(sub)
            recon = out["sae_out"].float()

    sub = sub.float()
    orig_norm = F.normalize(sub, dim=1)
    recon_norm = F.normalize(recon, dim=1)

    cos_orig = (orig_norm @ orig_norm.T).cpu().numpy()
    cos_recon = (recon_norm @ recon_norm.T).cpu().numpy()

    triu = np.triu_indices(n, k=1)
    rho, _ = spearmanr(cos_orig[triu], cos_recon[triu])
    return float(rho)


def downstream_classification(
    acts_by_label: Dict[str, torch.Tensor],
    raw_emb_by_label: Dict[str, torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Sonde logistique à 5 plis (StratifiedKFold) pour évaluer la séparabilité linéaire
    des activations latentes vs embeddings bruts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    X_sae_list, X_raw_list, y_list = [], [], []
    for label_id, (label_name, sae_acts) in enumerate(acts_by_label.items()):
        sae_np = sae_acts.float().detach().cpu().numpy()
        X_sae_list.append(sae_np)
        y_list.append(np.full(sae_np.shape[0], label_id))
        if raw_emb_by_label and label_name in raw_emb_by_label:
            X_raw_list.append(raw_emb_by_label[label_name].float().detach().cpu().numpy())

    X_sae = np.concatenate(X_sae_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs_sae = []

    for train_idx, test_idx in skf.split(X_sae, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
        clf.fit(X_sae[train_idx], y[train_idx])
        preds = clf.predict(X_sae[test_idx])
        accs_sae.append(accuracy_score(y[test_idx], preds))

    results = {"acc_sae": float(np.mean(accs_sae))}

    if X_raw_list:
        X_raw = np.concatenate(X_raw_list, axis=0)
        accs_raw = []
        for train_idx, test_idx in skf.split(X_raw, y):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
            clf.fit(X_raw[train_idx], y[train_idx])
            preds = clf.predict(X_raw[test_idx])
            accs_raw.append(accuracy_score(y[test_idx], preds))
        results["acc_raw"] = float(np.mean(accs_raw))
        results["delta_acc"] = results["acc_sae"] - results["acc_raw"]

    return results