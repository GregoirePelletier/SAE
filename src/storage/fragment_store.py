"""
src/storage/fragment_store.py — Stockage sparse (CSR) des fragments token-level.

Motivation : token_sae_acts dense float32 [T, d_core+D_EXTRA] à width 262k
= ~1 Mo/token, ~400 Mo/doc → I/O disque dominait le runtime (5.6 s/doc).
Avec L0 ≈ 70 (core JumpReLU) + 32 (extra), densité ≈ 4e-4 → CSR ≈ 250 Ko/doc.

Format (torch.save, .pt) :
  { "token_strings": list[str],
    "rowptr": int64 [T+1], "cols": int32 [nnz], "vals": float32 [nnz],
    # float32, pas float16 : les activations JumpReLU du core (non bornées,
    # outliers ~1e5) débordent la plage fp16 (max ~65504).
    "shape": (T, d_total),
    "raw_acts": bf16 [T, d_in] (optionnel, supprimé après entraînement SAEBoostResidualSAE) }

Rétro-compat : load_fragment lit aussi les anciens doc_*.pkl denses et les
convertit en CSR en mémoire (mêmes helpers utilisables partout).
"""
from __future__ import annotations

import os
import glob
import pickle
import queue
import threading

import numpy as np
import torch


class AsyncFragmentWriter:
    """Écriture de fragments en arrière-plan (audit perf G2,
    AUDIT_SAE_2026-08.md §2.2) : `torch.save` est libérateur du GIL (I/O), un
    seul thread consommateur suffit -- le débit est dominé par la latence de
    métadonnées d'un volume réseau partagé (~1-5 ms/fichier), pas par le CPU
    d'écriture. Le GPU ne doit jamais attendre le disque pour continuer.

    IMPORTANT pour la reprise (R1) : `flush()` DOIT être appelé avant tout
    checkpoint de progression qui prétend "ces documents sont traités" --
    sinon un crash entre le checkpoint et l'écriture réelle du fragment
    laisserait un état incohérent (checkpoint avancé, fragment absent),
    invisible tant qu'on ne tente pas de relire ce fragment à la reprise."""

    def __init__(self, maxsize: int = 64):
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            path, payload = item
            try:
                torch.save(payload, path)
            except Exception as e:  # noqa: BLE001 -- remonté à submit()/flush(), pas avalé
                self._error = e
            finally:
                self._queue.task_done()

    def submit(self, path: str, payload: dict) -> None:
        if self._error is not None:
            raise RuntimeError(f"Écriture de fragment en arrière-plan échouée : {self._error}") from self._error
        self._queue.put((path, payload))

    def flush(self) -> None:
        """Bloque jusqu'à ce que toutes les écritures déjà soumises soient
        terminées sur disque -- point de synchronisation nécessaire (et
        suffisant) avant d'avancer un checkpoint de reprise."""
        self._queue.join()
        if self._error is not None:
            raise RuntimeError(f"Écriture de fragment en arrière-plan échouée : {self._error}") from self._error

    def close(self) -> None:
        self.flush()
        self._queue.put(None)
        self._thread.join()


# ─── chemins ───

def _pt_path(d: str, i: int) -> str:  return os.path.join(d, f"doc_{i:05d}.pt")
def _pkl_path(d: str, i: int) -> str: return os.path.join(d, f"doc_{i:05d}.pkl")


# ─── shards (audit perf item 3, AUDIT_SAE_2026-08.md §2.2) ───
#
# Un fichier par document (doc_*.pt) coûte ~1-5 ms de latence de métadonnées
# par création sur le volume réseau partagé -- 432k créations + fsync
# implicites ≈ 22 min de latence pure sur le run de référence. SHARD_SIZE
# documents par fichier (shards/shard_{k:05d}.pt = {doc_id: payload, ...})
# réduit ce nombre par SHARD_SIZE. Utilisé UNIQUEMENT côté extraction
# (ShardedFragmentWriter) : le ré-encodage continue d'écrire un fichier par
# document via save_fragment() (déjà testé cette session, aucune raison de
# le retoucher) -- ces écritures individuelles PRIMENT sur le contenu du
# shard à la lecture (cf. load_fragment), donc un document ré-encodé sort
# proprement de son shard sans jamais le réécrire.
SHARD_SIZE = 1000


def _shard_dir(fragments_dir: str) -> str:
    return os.path.join(fragments_dir, "shards")


def _shard_idx(doc_id: int) -> int:
    return doc_id // SHARD_SIZE


def _shard_path(fragments_dir: str, shard_idx: int) -> str:
    return os.path.join(_shard_dir(fragments_dir), f"shard_{shard_idx:05d}.pt")


class _ShardReadCache:
    """Cache le dernier shard chargé (fragments_dir, shard_idx) -> contenu.
    Extraction et ré-encodage parcourent les documents dans l'ordre : des
    lectures consécutives portent presque toujours sur la MÊME shard -- sans
    ce cache, lire SHARD_SIZE documents d'un même shard le rechargerait
    entièrement SHARD_SIZE fois (régression pire que le problème d'origine).

    Péremption intra-process : si un shard est lu (mis en cache) PUIS
    modifié par un ShardedFragmentWriter du MÊME process (ex. reprise dans
    une shard déjà partiellement lue par la reconstruction de repli, cf.
    saev5.py), une lecture ultérieure de cette shard doit voir le contenu
    frais, pas la version mise en cache avant l'écriture --
    ShardedFragmentWriter.flush() appelle invalidate() après confirmation
    que l'écriture a atteint le disque (jamais avant, cf. sa docstring)."""

    def __init__(self):
        self._key = None
        self._content: dict = {}

    def get(self, fragments_dir: str, shard_idx: int) -> dict:
        key = (fragments_dir, shard_idx)
        if key != self._key:
            path = _shard_path(fragments_dir, shard_idx)
            self._content = torch.load(path, map_location="cpu", weights_only=False) \
                if os.path.exists(path) else {}
            self._key = key
        return self._content

    def invalidate(self, fragments_dir: str, shard_idx: int) -> None:
        if self._key == (fragments_dir, shard_idx):
            self._key = None
            self._content = {}


_shard_read_cache = _ShardReadCache()


def fragment_exists(fragments_dir: str, doc_id: int) -> bool:
    if os.path.exists(_pt_path(fragments_dir, doc_id)) or \
       os.path.exists(_pkl_path(fragments_dir, doc_id)):
        return True
    return doc_id in _shard_read_cache.get(fragments_dir, _shard_idx(doc_id))


def list_fragment_ids(fragments_dir: str) -> list[int]:
    ids = set()
    for p in glob.glob(os.path.join(fragments_dir, "doc_*.pt")):
        ids.add(int(os.path.basename(p)[4:9]))
    for p in glob.glob(os.path.join(fragments_dir, "doc_*.pkl")):
        ids.add(int(os.path.basename(p)[4:9]))
    for p in glob.glob(os.path.join(_shard_dir(fragments_dir), "shard_*.pt")):
        shard = torch.load(p, map_location="cpu", weights_only=False)
        ids.update(shard.keys())
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


class ShardedFragmentWriter:
    """Écrit les fragments par lots de SHARD_SIZE documents dans un seul
    fichier (audit perf item 3, AUDIT_SAE_2026-08.md §2.2/§2.9) au lieu d'un
    fichier par document -- réduit 432k créations de fichiers/fsync à ~432
    sur le run de référence. Utilisée UNIQUEMENT côté extraction (saev5.py) ;
    le ré-encodage continue d'écrire un fichier par document via
    save_fragment() (cf. docstring load_fragment pour la résolution de
    priorité fichier-individuel > shard).

    Écriture réellement effectuée par un AsyncFragmentWriter interne (même
    discipline "flush() avant tout checkpoint" que celui-ci, cf. sa
    docstring) -- flush() reste le seul point de synchronisation nécessaire
    et suffisant avant d'avancer un checkpoint de reprise (R1).

    Reprise : un shard peut avoir été flushé PARTIEL (coupure gracieuse en
    cours de shard, cf. saev5.py). add() précharge alors le contenu déjà sur
    disque pour ce shard au premier contact -- sans ça, un flush() ultérieur
    écraserait le fichier avec seulement les documents ajoutés APRÈS la
    reprise, perdant ceux d'avant la coupure."""

    def __init__(self, fragments_dir: str):
        self.fragments_dir = fragments_dir
        self._buffers: dict[int, dict] = {}
        self._async_writer = AsyncFragmentWriter()
        os.makedirs(_shard_dir(fragments_dir), exist_ok=True)

    def add(
        self,
        doc_id: int,
        token_strings: list[str],
        acts_dense: torch.Tensor = None,
        csr: tuple = None,
        d_total: int = None,
        raw_acts: torch.Tensor = None,
    ) -> None:
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

        shard_idx = _shard_idx(doc_id)
        if shard_idx not in self._buffers:
            # Précharge un éventuel contenu déjà sur disque (reprise après
            # coupure en cours de shard) -- sinon le flush() suivant
            # écraserait ces documents déjà persistés.
            self._buffers[shard_idx] = dict(_shard_read_cache.get(self.fragments_dir, shard_idx))
        self._buffers[shard_idx][doc_id] = payload

    def flush(self) -> None:
        """Écrit l'état courant de tous les shards en mémoire sur disque --
        idempotent (écrase le fichier existant avec le contenu COMPLET connu
        à cet instant), sûr à appeler à tout moment. Les shards COMPLETS
        (SHARD_SIZE documents atteints) sont libérés de la RAM après
        écriture ; les shards partiels restent pour continuer d'être
        complétés et réécrits en entier au prochain flush().

        Invalide le cache de lecture (_shard_read_cache) pour chaque shard
        touché, APRÈS confirmation que l'écriture a atteint le disque
        (self._async_writer.flush() bloque jusque-là) -- jamais avant, sinon
        une lecture concurrente entre l'invalidation et l'écriture réelle
        rechargerait encore l'ancien contenu depuis le disque."""
        touched = list(self._buffers.keys())
        for shard_idx, docs in list(self._buffers.items()):
            path = _shard_path(self.fragments_dir, shard_idx)
            self._async_writer.submit(path, dict(docs))  # copie : docs continue d'être mutable après ce point
            if len(docs) >= SHARD_SIZE:
                del self._buffers[shard_idx]
        self._async_writer.flush()
        for shard_idx in touched:
            _shard_read_cache.invalidate(self.fragments_dir, shard_idx)

    def close(self) -> None:
        self.flush()
        self._async_writer.close()


# ─── écriture ───

def save_fragment(
    fragments_dir: str,
    doc_id: int,
    token_strings: list[str],
    acts_dense: torch.Tensor = None,       # [T, d_stored] (peut être < d_total : cols au-delà = zéros)
    csr: tuple = None,                     # (rowptr, cols, vals, shape) déjà construit
    d_total: int = None,                   # largeur logique (core + extra)
    raw_acts: torch.Tensor = None,
    writer: "AsyncFragmentWriter" = None,  # si fourni, écriture en arrière-plan (audit perf G2)
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
        # .cpu() ici, synchrone : le payload remis à writer.submit() ne doit
        # plus contenir aucun tenseur GPU (le thread d'écriture tourne
        # indépendamment de tout contexte CUDA appelant).
        payload["raw_acts"] = raw_acts.detach().to(torch.bfloat16).cpu()
    path = _pt_path(fragments_dir, doc_id)
    if writer is not None:
        writer.submit(path, payload)
    else:
        torch.save(payload, path)
    # supprime l'ancien pkl dense s'il existe (migration in-place)
    legacy = _pkl_path(fragments_dir, doc_id)
    if os.path.exists(legacy):
        os.remove(legacy)
    return path


# ─── lecture ───

def load_fragment(fragments_dir: str, doc_id: int) -> dict:
    # Fichier individuel (écrit par save_fragment, ex. ré-encodage) : PRIME sur
    # le shard -- c'est la version la plus récente d'un document déjà réencodé
    # (cf. ShardedFragmentWriter, un document ré-encodé "sort" de son shard).
    path = _pt_path(fragments_dir, doc_id)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    legacy = _pkl_path(fragments_dir, doc_id)
    if os.path.exists(legacy):
        with open(legacy, "rb") as f:
            old = pickle.load(f)
        rowptr, cols, vals, shape = _dense_to_csr(old["token_sae_acts"])
        frag = {"token_strings": old["token_strings"],
                "rowptr": rowptr, "cols": cols, "vals": vals, "shape": shape}
        if "raw_acts" in old:
            frag["raw_acts"] = old["raw_acts"]
        return frag
    shard = _shard_read_cache.get(fragments_dir, _shard_idx(doc_id))
    if doc_id in shard:
        return shard[doc_id]
    raise FileNotFoundError(
        f"Fragment {doc_id} introuvable dans {fragments_dir} (ni fichier individuel, ni shard)."
    )


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
    vals = frag["vals"][keep].to(device).to(torch.float32)
    rows = _row_index(frag)[keep].to(device)

    if not hasattr(sae, "_W_dec_fp32"):
        sae._W_dec_fp32 = sae.W_dec.to(device).to(torch.float32)
        sae._b_dec_fp32 = sae.b_dec.to(device).to(torch.float32)
    W_dec, b_dec = sae._W_dec_fp32, sae._b_dec_fp32
    
    out = torch.zeros(T, W_dec.shape[1], device=device, dtype=torch.float32)
    out.index_add_(0, rows, vals.unsqueeze(1) * W_dec[cols])
    return out + b_dec   # sortie bf16 inchangée côté appelants

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