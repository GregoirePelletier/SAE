"""
src/storage/checkpoint.py — Reprise après coupure (R1, AUDIT_SAE_2026-08.md
§2.3/§4.3), partagé entre Pipeline 1 (saev5.py, extraction Gemma-3) et
Pipeline 2 (phrase_sae.py, extraction F2LLM). Deux briques génériques :

  - `read_checkpoint`/`write_checkpoint` : sidecar JSON, écriture atomique
    (tmp + os.replace) -- un crash pendant l'écriture ne doit jamais laisser
    un checkpoint à moitié écrit, illisible ou incohérent, ce serait pire que
    l'absence de checkpoint (reprise sur un état corrompu plutôt que sur
    "repartir de zéro").
  - `GracefulShutdown` : drapeau positionné par SIGTERM/SIGUSR1 (SLURM :
    `--signal=B:USR1@600` envoie SIGUSR1 ~10 min avant le SIGKILL d'un
    timeout), vérifié entre deux unités de travail dans la boucle appelante --
    jamais de travail fait DANS le handler lui-même (Python + CUDA ne
    garantit rien sur ce qui est sûr à l'intérieur d'un signal handler).

Principe commun aux deux pipelines : le critère de reprise est TOUJOURS "quel
est le prochain élément non traité", jamais "le run est-il complet".
"""
from __future__ import annotations

import json
import os
import signal
import threading


def checkpoint_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, f"{name}.progress.json")


def read_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_checkpoint(path: str, **fields) -> None:
    """Écriture atomique : le fichier temporaire est sur le même volume que la
    cible (même répertoire) pour que os.replace reste une opération atomique
    au niveau du système de fichiers (pas garanti entre volumes différents)."""
    tmp_path = path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(fields, f)
    os.replace(tmp_path, path)


def atomic_create_exclusive(path: str, **fields) -> bool:
    """Crée `path` de façon atomique avec un contenu JSON déjà complet, en
    échouant proprement (sans effet) si `path` existe déjà -- primitive pour
    tout verrou de type `O_CREAT|O_EXCL` qui a aussi besoin d'écrire un
    contenu non trivial (ex. `acquire_shared_cache_lock`, sae_shared.py).

    `O_CREAT|O_EXCL` suivi d'une écriture séparée dans le même fd laisse une
    fenêtre où le fichier est visible avec un contenu vide/tronqué : un
    lecteur concurrent qui tombe dans cette fenêtre peut le juger corrompu ou
    périmé. Ici, le contenu est entièrement écrit dans un fichier temporaire
    AVANT que `os.link` ne l'expose sous `path` -- `os.link` échoue
    atomiquement si `path` existe déjà (même garantie que O_EXCL), mais ne
    rend jamais visible un contenu partiel : soit `path` n'existe pas encore,
    soit il pointe déjà vers un inode entièrement écrit."""
    tmp_path = path + f".tmp{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w") as f:
        json.dump(fields, f)
    try:
        os.link(tmp_path, path)
        return True
    except FileExistsError:
        return False
    finally:
        os.remove(tmp_path)


def clear_checkpoint(path: str) -> None:
    """À appeler une fois le travail complet : un checkpoint périmé qui reste
    sur disque après un run terminé avec succès serait relu par erreur au
    prochain lancement et ferait croire à une reprise partielle."""
    if os.path.exists(path):
        os.remove(path)


class GracefulShutdown:
    """Un seul jeu de handlers process-wide (signal.signal est global, pas par
    instance) -- utiliser la classe directement, ne pas instancier."""
    requested = False

    @classmethod
    def _handler(cls, signum, frame):
        cls.requested = True

    @classmethod
    def install(cls) -> None:
        cls.requested = False
        signal.signal(signal.SIGTERM, cls._handler)
        signal.signal(signal.SIGUSR1, cls._handler)
