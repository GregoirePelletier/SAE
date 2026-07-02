"""
crosslingual.py — Alignement FR/EN entre SAE + évaluation downstream.

Cross-lingue : mêmes primitives que model_compare (le corpus parallèle ou
comparable sert de base commune), mais l'appariement se fait entre SAE_fr et
SAE_en du MÊME reader — mesure du transfert de concepts, pas de la pollution.
Métrique : mean matched corr + fraction de features appariées (>0.5).

Downstream (Mission 5) : probes logistiques 5-fold sur trois représentations
(raw, SAE acts, reconstruction x̂) + CE-loss increase du LM sous patch x → x̂.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .model_compare import match_features


def crosslingual_alignment(Z_fr: torch.Tensor, Z_en: torch.Tensor) -> dict:
    m = match_features(Z_fr, Z_en)
    return {
        "matches": m,
        "mean_corr": float(m["corr"].mean()),
        "frac_aligned_05": float((m["corr"] > 0.5).mean()),
    }


def probe_eval(
    X: np.ndarray, y: np.ndarray, cv: int = 5, C: float = 1.0
) -> tuple[float, float]:
    """Accuracy 5-fold stratifiée, probe logistique L2."""
    clf = LogisticRegression(max_iter=2000, C=C, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def downstream_report(
    raw: torch.Tensor, sae_acts: torch.Tensor, recon: torch.Tensor, y: np.ndarray
) -> dict:
    """R², L0, et probes sur (raw | acts | recon)."""
    mse = F.mse_loss(recon.float(), raw.float())
    var = raw.float().var()
    r2 = float(1 - mse / (var + 1e-8))
    l0 = float((sae_acts.abs() > 1e-6).float().sum(-1).mean())
    out = {"R2": r2, "L0": l0}
    for name, X in [("raw", raw), ("sae_acts", sae_acts), ("recon", recon)]:
        acc, sd = probe_eval(X.float().cpu().numpy(), y)
        out[f"probe_{name}"] = {"acc": acc, "std": sd}
    out["probe_delta_recon"] = out["probe_recon"]["acc"] - out["probe_raw"]["acc"]
    return out


@torch.no_grad()
def ce_loss_increase(
    texts: list[str], model, tokenizer, sae, layer: int,
    device: str = "cuda", max_length: int = 256, batch_size: int = 4,
) -> dict:
    """
    ΔCE = CE(patched) - CE(clean), patch x_t → x̂_t = SAE(x_t) à la couche `layer`
    via forward hook. Métrique standard SAEBench de fidélité fonctionnelle.
    """
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        rec = sae.decode(sae.encode(h.to(torch.bfloat16))).to(h.dtype)
        return (rec, *output[1:]) if isinstance(output, tuple) else rec

    ce_clean, ce_patch, n = 0.0, 0.0, 0
    layer_module = model.model.layers[layer]
    for s in range(0, len(texts), batch_size):
        enc = tokenizer(texts[s:s + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        labels = ids.masked_fill(attn == 0, -100)

        ce_clean += float(model(input_ids=ids, attention_mask=attn, labels=labels).loss) * len(ids)
        h = layer_module.register_forward_hook(hook)
        try:
            ce_patch += float(model(input_ids=ids, attention_mask=attn, labels=labels).loss) * len(ids)
        finally:
            h.remove()
        n += len(ids)
    return {"ce_clean": ce_clean / n, "ce_patched": ce_patch / n,
            "delta_ce": (ce_patch - ce_clean) / n}