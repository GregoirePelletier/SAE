import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix as sp_csr
from typing import Dict, Any, List


def compute_metrics(
    model: nn.Module,
    acts: torch.Tensor,
    batch_size: int = 4096,
    is_saelens: bool = False,
    var_sample: int = 65536,
    device: str = "cuda",
) -> Dict[str, float]:
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


def compute_rho_sae(
    model: nn.Module,
    acts: torch.Tensor,
    n_sample: int = 500,
    batch_size: int = 512,
    is_saelens: bool = False,
    device: str = "cuda",
) -> float:
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
    orig_norm = torch.nn.functional.normalize(sub, dim=1)
    recon_norm = torch.nn.functional.normalize(recon, dim=1)
    cos_orig = (orig_norm @ orig_norm.T).cpu().numpy()
    cos_recon = (recon_norm @ recon_norm.T).cpu().numpy()

    triu = np.triu_indices(n, k=1)
    rho, _ = spearmanr(cos_orig[triu], cos_recon[triu])
    return float(rho)


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

    import pandas as pd
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


def pool_embeddings_by_document(
    phrase_embeddings: torch.Tensor,
    phrase_to_doc: np.ndarray,
    n_docs: int = None,
) -> torch.Tensor:
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


def downstream_classification(
    acts_by_label: Dict[str, torch.Tensor],
    raw_emb_by_label: Dict[str, torch.Tensor] = None,
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    labels_list = sorted(acts_by_label.keys())
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
