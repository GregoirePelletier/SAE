"""
sae_shared.py — Harnais d'entraînement SAEBoostResidualSAE + steering + ré-exports
partagés entre les deux pipelines.
"""

import os
import sys
import math
import json
import re
import hashlib
import time
import socket
import threading
import contextlib
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
        normalize_by_p90_and_score,
        average_precision,
        precision_at_k,
        mean_average_precision,
        mean_precision_at_k,
        reciprocal_rank_fusion,
        rank_biased_overlap,
    )
except ImportError:
    from metrics import (
        compute_metrics,
        compute_rho_sae,
        downstream_classification,
        normalize_by_p90_and_score,
        average_precision,
        precision_at_k,
        mean_average_precision,
        mean_precision_at_k,
        reciprocal_rank_fusion,
        rank_biased_overlap,
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

try:
    from src.storage.checkpoint import (
        atomic_create_exclusive as _lock_atomic_create_exclusive,
        write_checkpoint as _lock_write_checkpoint,
        read_checkpoint as _lock_read_checkpoint,
    )
except ImportError:
    from checkpoint import (
        atomic_create_exclusive as _lock_atomic_create_exclusive,
        write_checkpoint as _lock_write_checkpoint,
        read_checkpoint as _lock_read_checkpoint,
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


# ─── CACHE D'ACTIVATIONS PARTAGÉ ENTRE RUNS (R5) ───

def compute_activation_cache_key(
    train_texts: List[str], volume_filler_texts: List[str],
    test_texts: List[str], diff_texts: List[str],
    model_id: str, layer: int, hook_type: str, dtype: str,
    sae_id: str, n_tokens_extra_train: int,
    max_length: int, sigma_clip: float, skip_first_content_token: bool,
) -> str:
    """Clé de cache mécanique (R5) pour les artefacts d'extraction PURE
    (résidus bruts du réservoir, activations core max-poolées, fragments
    token-level) -- partageables entre deux runs qui ne diffèrent QUE par des
    paramètres downstream de SAEBoostResidualSAE (K_EXTRA/D_EXTRA/EPOCHS_EXTRA),
    puisque ces artefacts ne dépendent que du modèle, de la couche, du hook,
    du SAE core, du budget de tokens, du corpus VU à l'extraction (après
    troncature -- `max_length` fait partie du payload pour cette raison
    précise, N8, AUDIT_SAE_2026-08.md §8 : le hash du corpus ci-dessous porte
    sur le texte AVANT troncature, `max_length` est le seul signal qui
    distingue deux runs dont le texte tronqué diffère à texte source
    identique) et du masquage des tokens (`sigma_clip`,
    `skip_first_content_token` -- mêmes arguments passés tels quels à
    `valid_token_mask`/`norm_outlier_mask`, saev5.py). Hash du CONTENU du
    corpus (pas seulement de sa config de génération : chemins de fichiers,
    seed) -- robuste à tout changement de logique de génération qui
    produirait un corpus différent à longueurs égales, sans quoi une
    collision de clé réutiliserait silencieusement les activations d'un
    AUTRE corpus (piège cache/checkpoint, `CLAUDE.md`)."""
    corpus_hash = hashlib.sha1(
        "\n".join(train_texts + volume_filler_texts + test_texts + diff_texts)
        .encode("utf-8", errors="ignore")
    ).hexdigest()
    payload = {
        "corpus_hash": corpus_hash, "model_id": model_id, "layer": layer,
        "hook_type": hook_type, "dtype": dtype, "sae_id": sae_id,
        "n_tokens_extra_train": n_tokens_extra_train,
        "max_length": max_length, "sigma_clip": sigma_clip,
        "skip_first_content_token": skip_first_content_token,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def shared_activation_cache_dir(key: str) -> str:
    """Répertoire physique du cache partagé pour une clé donnée
    (`compute_activation_cache_key`) -- `local_data/activation_cache/<clé>/`,
    hors de tout `SAVE_DIR` individuel, créé si absent."""
    path = os.path.join(REPO_ROOT, "local_data", "activation_cache", key)
    os.makedirs(path, exist_ok=True)
    return path


def save_doc_acts_sparse_filler(tensor: "torch.Tensor", n_train: int, n_filler: int, path: str) -> None:
    """Sauvegarde compacte de `all_doc_sae_acts` (N10, AUDIT_SAE_2026-08.md
    §8) : la plage filler `[n_train, n_train+n_filler)` est connue a priori
    et n'est JAMAIS lue en aval (aucun consommateur n'indexe cette plage, cf.
    saev5.py -- placeholders "jamais lus en aval") -- son contenu ne mérite
    aucune place sur le cache d'extraction PARTAGÉ et persistant (avant ce
    correctif, ~42 Go de lignes filler, généralement des zéros, vivaient dans
    `local_data/activation_cache/<clé>/`, dupliqués par clé de cache -- alors
    qu'un `SAVE_DIR` individuel jetable les aurait au moins vus supprimés au
    nettoyage disque). Stocke uniquement les lignes hors filler + assez de
    métadonnées pour reconstruire un tenseur de la bonne forme, zero-paddé sur
    la plage filler, via `load_doc_acts_sparse_filler`."""
    n_total = tensor.shape[0]
    keep = torch.ones(n_total, dtype=torch.bool)
    keep[n_train:n_train + n_filler] = False
    torch.save({
        "n_total": n_total, "d": tensor.shape[1],
        "n_train": n_train, "n_filler": n_filler,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "kept_rows": tensor[keep].clone(),
    }, path)


def save_doc_acts_compact(
    compact_tensor: "torch.Tensor", n_train: int, n_filler: int, n_total: int, path: str
) -> None:
    """Comme `save_doc_acts_sparse_filler`, pour un appelant qui n'a JAMAIS
    matérialisé les lignes filler en mémoire (correctif E00,
    `memory_diagnosis.md` §2 -- la construction en liste Python + append d'une
    ligne zéro par document filler, suivie de `torch.stack` sur l'ensemble,
    faisait transitoirement coexister liste et tenseur à l'échelle `n_total`
    plutôt qu'à l'échelle réellement utile). `compact_tensor` a déjà
    `n_total - n_filler` lignes (train ++ test ++ diff, même ordre que
    `build_reencode_targets`) -- rien à masquer, contrairement à
    `save_doc_acts_sparse_filler`. Même schéma de fichier en sortie : tout
    consommateur existant (`load_doc_acts_sparse_filler`/`load_all_doc_acts`)
    continue de fonctionner sans modification."""
    torch.save({
        "n_total": n_total, "d": compact_tensor.shape[1],
        "n_train": n_train, "n_filler": n_filler,
        "dtype": str(compact_tensor.dtype).removeprefix("torch."),
        "kept_rows": compact_tensor,
    }, path)


def _reconstruct_sparse_filler_payload(payload: dict) -> "torch.Tensor":
    dtype = getattr(torch, payload["dtype"])
    out = torch.zeros(payload["n_total"], payload["d"], dtype=dtype)
    n_train, n_filler = payload["n_train"], payload["n_filler"]
    keep = torch.ones(payload["n_total"], dtype=torch.bool)
    keep[n_train:n_train + n_filler] = False
    out[keep] = payload["kept_rows"]
    return out


def load_doc_acts_sparse_filler(path: str) -> "torch.Tensor":
    """Inverse de `save_doc_acts_sparse_filler` : reconstruit un tenseur
    `[n_total, d]` avec la plage filler `[n_train, n_train+n_filler)`
    zero-paddée -- équivalent, pour tout consommateur en aval, au tenseur
    dense original (les lignes filler n'y étaient de toute façon jamais lues
    qu'à zéro ou en valeurs jamais consommées, cf. docstring ci-dessus)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _reconstruct_sparse_filler_payload(payload)


def load_all_doc_acts(path: str) -> "torch.Tensor":
    """Charge un tenseur `all_doc_sae_acts`, qu'il soit au format dense
    classique (`p1_all_doc_acts_ext_d*.pt`, ré-encodage privé par run, jamais
    compacté -- les lignes filler n'y sont jamais matérialisées, cf.
    `build_reencode_targets`) ou au format compact filler-creux
    (`p1_all_doc_acts.pt`, cache d'extraction PARTAGÉ, N10,
    AUDIT_SAE_2026-08.md §8). Dispatché sur le CONTENU du fichier (dict vs
    tenseur), pas sur son nom -- un appelant qui suit la convention de
    repli `p1_all_doc_acts_ext_d*.pt` puis `p1_all_doc_acts.pt` (une
    dizaine de scripts d'analyse) n'a pas besoin de savoir laquelle des deux
    conventions de stockage s'applique."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "kept_rows" in obj:
        return _reconstruct_sparse_filler_payload(obj)
    return obj


class SharedCacheLockTimeout(RuntimeError):
    """Levée par `acquire_shared_cache_lock` quand `max_wait_s` est dépassé
    sans obtenir le verrou -- un tiers le tient depuis plus longtemps que
    `stale_after_s` sans jamais rafraîchir son heartbeat serait un bug (ou un
    job mort dont le heartbeat a cessé net, cf. docstring) : abandon explicite
    plutôt qu'un blocage indéfini d'un job dont le budget SLURM est de toute
    façon borné."""


def _shared_cache_lock_owner_id() -> str:
    """Identité mécanique du détenteur du verrou -- `SLURM_JOB_ID` si présent
    (cas normal, run soumis via `sbatch`), sinon hostname:pid (run
    interactif/débogage)."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"slurm:{job_id}"
    return f"{socket.gethostname()}:{os.getpid()}"


@contextlib.contextmanager
def acquire_shared_cache_lock(
    cache_dir: str,
    heartbeat_interval_s: float = 30.0,
    stale_after_s: float = 300.0,
    poll_interval_s: float = 10.0,
    max_wait_s: float = 6 * 3600.0,
):
    """Verrou de création exclusive (`atomic_create_exclusive`, équivalent
    `O_EXCL` sans fenêtre de contenu partiel, cf. `src/storage/checkpoint.py`)
    sur `cache_dir/.extraction.lock` (N2, AUDIT_SAE_2026-08.md §8) -- la
    méthode de travail de ce dépôt lance
    délibérément des jobs de MÊME clé de cache en parallèle sur plusieurs
    partitions (course, l'utilisateur annulant le perdant une fois qu'un des
    deux démarre réellement), mais sans ce verrou, deux jobs qui
    démarreraient tous les deux leur extraction avant que l'annulation
    manuelle n'intervienne écriraient concurremment le même memmap/shards/
    checkpoint de progression -- corruption silencieuse possible.

    Un second processus qui rencontre un verrou actif ATTEND (poll) que le
    premier le libère, plutôt que d'échouer ou de se replier sur un cache
    privé -- cohérent avec le workflow de course : le perdant doit de toute
    façon attendre le résultat du gagnant, dupliquer le travail dans un
    cache privé serait pire (double coût GPU pour le même résultat). Un
    verrou dont le heartbeat n'a plus été rafraîchi depuis `stale_after_s`
    (job mort/tué, ou sortie anticipée par `_GracefulShutdown`/`sys.exit(0)`
    sans repasser par la libération normale du verrou -- cf. appelant) est
    considéré abandonné et repris par le prochain prétendant.

    N'essaie PAS de garantir une libération propre sur toute sortie anticipée
    (`sys.exit(0)` de reprise checkpointée, notamment) : le thread de
    heartbeat, démon, meurt avec le process sans repasser par `finally` --
    le verrou s'auto-guérit via l'expiration `stale_after_s` au prochain
    prétendant plutôt que par une libération explicite. Acceptable ici (un
    job repris n'est typiquement pas resoumis dans la même minute) mais à
    garder en tête si `stale_after_s` est un jour resserré."""
    lock_path = os.path.join(cache_dir, ".extraction.lock")
    owner = _shared_cache_lock_owner_id()
    deadline = time.monotonic() + max_wait_s

    while True:
        # atomic_create_exclusive (src/storage/checkpoint.py) : écrit le
        # contenu complet dans un fichier temporaire PUIS l'expose sous
        # lock_path via os.link (échoue atomiquement si lock_path existe déjà)
        # -- contrairement à O_CREAT|O_EXCL suivi d'une écriture séparée dans
        # le même fd, aucune fenêtre où un lecteur concurrent verrait
        # lock_path vide/tronqué et le jugerait à tort corrompu ou périmé.
        if _lock_atomic_create_exclusive(lock_path, owner=owner, heartbeat=time.time()):
            break

        stale = True
        try:
            info = _lock_read_checkpoint(lock_path)
            stale = info is None or (time.time() - info.get("heartbeat", 0)) > stale_after_s
        except (OSError, ValueError):
            stale = True   # fichier illisible/corrompu : traité comme abandonné

        if stale:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass
            continue   # retente l'acquisition immédiatement, sans attendre poll_interval_s

        if time.monotonic() >= deadline:
            raise SharedCacheLockTimeout(
                f"{lock_path} : verrou tenu par un autre run depuis plus de "
                f"{max_wait_s:.0f}s sans expirer -- abandon plutôt que blocage indéfini."
            )
        time.sleep(poll_interval_s)

    stop_heartbeat = threading.Event()

    def _heartbeat_loop():
        while not stop_heartbeat.wait(heartbeat_interval_s):
            try:
                # write_checkpoint (tmp + os.replace) : remplacement atomique,
                # jamais de troncature en place -- même raison que
                # atomic_create_exclusive ci-dessus (R1, un lecteur concurrent
                # ne doit jamais voir un contenu partiel).
                _lock_write_checkpoint(lock_path, owner=owner, heartbeat=time.time())
            except OSError:
                pass   # verrou déjà supprimé/volé -- rien à faire ici, cf. libération ci-dessous

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()
    try:
        yield
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=heartbeat_interval_s + 5)
        # Ne supprime le fichier que s'il nous appartient TOUJOURS -- jamais
        # volé entretemps par un tiers qui l'aurait cru abandonné (fenêtre
        # improbable mais réelle si ce process a été suspendu > stale_after_s
        # sans que son thread de heartbeat n'ait pu tourner).
        try:
            info = _lock_read_checkpoint(lock_path)
            if info is not None and info.get("owner") == owner:
                os.remove(lock_path)
        except (OSError, ValueError):
            pass


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