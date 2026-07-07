"""
src/storage/fragment_store.py — Stockage sparse (CSR) des fragments token-level.

Motivation : token_sae_acts dense float32 [T, d_core+D_EXTRA] à width 262k
= ~1 Mo/token, ~400 Mo/doc → I/O disque dominait le runtime (5.6 s/doc).
Avec L0 ≈ 70 (core JumpReLU) + 32 (extra), densité ≈ 4e-4 → CSR ≈ 250 Ko/doc.

Format (torch.save, .pt) :
  { "token_strings": list[str],
    "rowptr": int64 [T+1], "cols": int32 [nnz], "vals": float16 [nnz],
    "shape": (T, d_total),
    "raw_acts": bf16 [T, d_in] (optionnel, supprimé après entraînement ExtendedSAE) }

Rétro-compat : load_fragment lit aussi les anciens doc_*.pkl denses et les
convertit en CSR en mémoire (mêmes helpers utilisables partout).
"""
from __future__ import annotations

import os
import glob
import pickle

import numpy as np
import torch


# ─── chemins ───

def _pt_path(d: str, i: int) -> str:  return os.path.join(d, f"doc_{i:05d}.pt")
def _pkl_path(d: str, i: int) -> str: return os.path.join(d, f"doc_{i:05d}.pkl")


def fragment_exists(fragments_dir: str, doc_id: int) -> bool:
    return os.path.exists(_pt_path(fragments_dir, doc_id)) or \
           os.path.exists(_pkl_path(fragments_dir, doc_id))


def list_fragment_ids(fragments_dir: str) -> list[int]:
    ids = set()
    for p in glob.glob(os.path.join(fragments_dir, "doc_*.pt")):
        ids.add(int(os.path.basename(p)[4:9]))
    for p in glob.glob(os.path.join(fragments_dir, "doc_*.pkl")):
        ids.add(int(os.path.basename(p)[4:9]))
    return sorted(ids)


# ─── conversion dense -> CSR ───

def _dense_to_csr(acts: torch.Tensor, eps: float = 1e-6):
    """acts [T, d] -> (rowptr, cols, vals, shape). CPU only."""
    acts = acts.detach().float().cpu()
    mask = acts > eps
    counts = mask.sum(dim=1)
    rowptr = torch.zeros(acts.shape[0] + 1, dtype=torch.int64)
    rowptr[1:] = counts.cumsum(0)
    rows, cols = mask.nonzero(as_tuple=True)
    vals = acts[rows, cols].to(torch.float32)
    return rowptr, cols.to(torch.int32), vals, tuple(acts.shape)


# ─── écriture ───

def save_fragment(
    fragments_dir: str,
    doc_id: int,
    token_strings: list[str],
    acts_dense: torch.Tensor = None,       # [T, d_stored] (peut être < d_total : cols au-delà = zéros)
    csr: tuple = None,                     # (rowptr, cols, vals, shape) déjà construit
    d_total: int = None,                   # largeur logique (core + extra)
    raw_acts: torch.Tensor = None,
) -> str:
    if csr is None:
        assert acts_dense is not None
        rowptr, cols, vals, shape = _dense_to_csr(acts_dense)
    else:
        rowptr, cols, vals, shape = csr
    T = shape[0]
    payload = {
        "token_strings": token_strings,
        "rowptr": rowptr, "cols": cols, "vals": vals,
        "shape": (T, int(d_total or shape[1])),
    }
    if raw_acts is not None:
        payload["raw_acts"] = raw_acts.detach().to(torch.bfloat16).cpu()
    path = _pt_path(fragments_dir, doc_id)
    torch.save(payload, path)
    # supprime l'ancien pkl dense s'il existe (migration in-place)
    legacy = _pkl_path(fragments_dir, doc_id)
    if os.path.exists(legacy):
        os.remove(legacy)
    return path


# ─── lecture ───

def load_fragment(fragments_dir: str, doc_id: int) -> dict:
    path = _pt_path(fragments_dir, doc_id)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    legacy = _pkl_path(fragments_dir, doc_id)
    with open(legacy, "rb") as f:
        old = pickle.load(f)
    rowptr, cols, vals, shape = _dense_to_csr(old["token_sae_acts"])
    frag = {"token_strings": old["token_strings"],
            "rowptr": rowptr, "cols": cols, "vals": vals, "shape": shape}
    if "raw_acts" in old:
        frag["raw_acts"] = old["raw_acts"]
    return frag


# ─── accès O(nnz) ───

def feature_column(frag: dict, f_idx: int) -> np.ndarray:
    """Colonne f_idx -> np.float32 [T]. O(nnz)."""
    T = frag["shape"][0]
    out = np.zeros(T, dtype=np.float32)
    cols = frag["cols"].numpy()
    sel = np.where(cols == f_idx)[0]
    if len(sel):
        rowptr = frag["rowptr"].numpy()
        rows = np.searchsorted(rowptr, sel, side="right") - 1
        out[rows] = frag["vals"].numpy()[sel].astype(np.float32)
    return out


def doc_maxpool(frag: dict) -> torch.Tensor:
    """max_t acts[t, f] -> [d_total]. O(nnz)."""
    d = frag["shape"][1]
    out = torch.zeros(d)
    out.scatter_reduce_(0, frag["cols"].long(),
                        frag["vals"].float(), reduce="amax", include_self=True)
    return out


def sum_columns(frag: dict) -> np.ndarray:
    """Σ_t acts[t, :] -> np.float64 [d_total] (pour feature_selection_by_magnitude)."""
    d = frag["shape"][1]
    acc = np.zeros(d, dtype=np.float64)
    np.add.at(acc, frag["cols"].numpy().astype(np.int64),
              frag["vals"].numpy().astype(np.float64))
    return acc


def _row_index(frag: dict) -> torch.Tensor:
    """Index de ligne par nnz, reconstruit depuis rowptr."""
    rowptr = frag["rowptr"]
    counts = rowptr[1:] - rowptr[:-1]
    return torch.repeat_interleave(torch.arange(len(counts)), counts)


def decode_core_sparse(frag: dict, sae, d_core: int, device: str = "cuda") -> torch.Tensor:
    """
    x̂_core[t] = Σ_{j: col<d_core} v · W_dec[j] + b_dec  — O(nnz · d_in),
    sans jamais densifier [T, d_core]. Retourne bf16 [T, d_in] sur device.
    """
    T = frag["shape"][0]
    keep = frag["cols"] < d_core
    cols = frag["cols"][keep].long().to(device)
    vals = frag["vals"][keep].to(device).to(torch.bfloat16)
    rows = _row_index(frag)[keep].to(device)

    W_dec = sae.W_dec.to(device).to(torch.bfloat16)      # [d_sae, d_in]
    b_dec = sae.b_dec.to(device).to(torch.bfloat16)
    out = torch.zeros(T, W_dec.shape[1], device=device, dtype=torch.bfloat16)
    out.index_add_(0, rows, vals.unsqueeze(1) * W_dec[cols])
    return out + b_dec


def merge_extra(frag: dict, extra_dense: torch.Tensor, d_core: int) -> tuple:
    """
    Fusionne les nnz core existants (cols < d_core) avec les activations extra
    denses [T, D_EXTRA] (sparsifiées, offset +d_core). Retourne un CSR trié.
    """
    T, _ = frag["shape"]
    keep = frag["cols"] < d_core
    rows_c = _row_index(frag)[keep]
    cols_c, vals_c = frag["cols"][keep].long(), frag["vals"][keep].float()

    rp_e, cols_e, vals_e, _ = _dense_to_csr(extra_dense)
    rows_e = torch.repeat_interleave(torch.arange(T), rp_e[1:] - rp_e[:-1])
    cols_e = cols_e.long() + d_core

    rows = torch.cat([rows_c, rows_e])
    cols = torch.cat([cols_c, cols_e])
    vals = torch.cat([vals_c, vals_e.float()])
    order = torch.argsort(rows * (d_core + extra_dense.shape[1]) + cols)
    rows, cols, vals = rows[order], cols[order], vals[order]

    rowptr = torch.zeros(T + 1, dtype=torch.int64)
    rowptr.scatter_add_(0, rows + 1, torch.ones_like(rows))
    rowptr = rowptr.cumsum(0)
    return rowptr, cols.to(torch.int32), vals, \
        (T, d_core + extra_dense.shape[1])