"""Teste src/sae/sae_shared.py::acquire_shared_cache_lock -- verrou O_EXCL
sur le cache d'extraction partagé (N2, AUDIT_SAE_2026-08.md §8), nécessaire
car la méthode de travail de ce dépôt lance délibérément des jobs de même
clé de cache en parallèle sur plusieurs partitions (course de SLURM)."""
import json
import os
import threading
import time

import pytest

from src.sae.sae_shared import (
    SharedCacheLockTimeout,
    acquire_shared_cache_lock,
    _shared_cache_lock_owner_id,
)


def _lock_path(cache_dir):
    return os.path.join(cache_dir, ".extraction.lock")


def test_lock_created_and_removed(tmp_path):
    cache_dir = str(tmp_path)
    with acquire_shared_cache_lock(cache_dir):
        assert os.path.exists(_lock_path(cache_dir))
    assert not os.path.exists(_lock_path(cache_dir))


def test_lock_content_has_owner_and_heartbeat(tmp_path):
    cache_dir = str(tmp_path)
    with acquire_shared_cache_lock(cache_dir):
        with open(_lock_path(cache_dir)) as f:
            info = json.load(f)
        assert info["owner"] == _shared_cache_lock_owner_id()
        assert isinstance(info["heartbeat"], (int, float))


def test_sequential_acquisitions_do_not_deadlock(tmp_path):
    cache_dir = str(tmp_path)
    with acquire_shared_cache_lock(cache_dir):
        pass
    # Le verrou a bien été libéré -- une seconde acquisition ne doit pas
    # attendre (poll_interval_s serait sinon visible dans le temps du test).
    t0 = time.monotonic()
    with acquire_shared_cache_lock(cache_dir):
        pass
    assert time.monotonic() - t0 < 1.0


def test_second_acquirer_waits_for_release(tmp_path):
    cache_dir = str(tmp_path)
    holder_acquired = threading.Event()   # synchronisation par événement, pas par sleep deviné
    acquired_second = threading.Event()
    release_first = threading.Event()

    def _holder():
        with acquire_shared_cache_lock(cache_dir, heartbeat_interval_s=0.05):
            holder_acquired.set()
            release_first.wait(timeout=15)

    def _waiter():
        with acquire_shared_cache_lock(cache_dir, poll_interval_s=0.05, max_wait_s=15):
            acquired_second.set()

    t_holder = threading.Thread(target=_holder)
    t_waiter = threading.Thread(target=_waiter)
    t_holder.start()
    # Pas de sleep deviné : attend le signal explicite que le premier tient
    # RÉELLEMENT le verrou avant de lancer le second (élimine la course sur
    # "le premier a-t-il eu le temps de l'acquérir ?").
    assert holder_acquired.wait(timeout=15)
    t_waiter.start()

    # Le second ne doit PAS avoir progressé pendant que le premier tient
    # encore le verrou -- marge large (2s pour un poll_interval_s=0.05, soit
    # ~40 cycles de poll) : un scheduler chargé (autres jobs/threads de la
    # session) peut retarder le thread sans que ça invalide le test, seul un
    # FAUX POSITIF (le second progresse AVANT la libération) le ferait échouer.
    time.sleep(2.0)
    assert not acquired_second.is_set()

    release_first.set()
    t_holder.join(timeout=15)
    t_waiter.join(timeout=15)
    assert acquired_second.is_set()

    release_first.set()
    t_holder.join(timeout=5)
    t_waiter.join(timeout=5)
    assert acquired_second.is_set()


def test_stale_lock_is_stolen_without_waiting_full_timeout(tmp_path):
    cache_dir = str(tmp_path)
    # Verrou "abandonné" : heartbeat très ancien, jamais rafraîchi (simule un
    # job tué avant sa libération normale du verrou).
    with open(_lock_path(cache_dir), "w") as f:
        json.dump({"owner": "slurm:12345", "heartbeat": time.time() - 10_000}, f)

    t0 = time.monotonic()
    with acquire_shared_cache_lock(cache_dir, stale_after_s=1.0, poll_interval_s=5, max_wait_s=5):
        pass
    # Le verrou périmé doit être repris immédiatement (retry sans dormir
    # poll_interval_s), pas après une attente complète.
    assert time.monotonic() - t0 < 2.0


def test_timeout_raised_when_lock_held_and_fresh(tmp_path):
    cache_dir = str(tmp_path)
    with open(_lock_path(cache_dir), "w") as f:
        json.dump({"owner": "slurm:99999", "heartbeat": time.time()}, f)

    with pytest.raises(SharedCacheLockTimeout):
        with acquire_shared_cache_lock(
            cache_dir, stale_after_s=300.0, poll_interval_s=0.05, max_wait_s=0.2,
        ):
            pass


def test_release_does_not_remove_lock_stolen_by_another_owner(tmp_path):
    # Si notre verrou a expiré et a été volé par un tiers pendant qu'on le
    # tenait encore (fenêtre improbable mais couverte par la docstring), la
    # libération ne doit PAS supprimer le fichier du nouveau propriétaire.
    cache_dir = str(tmp_path)
    lock_path = _lock_path(cache_dir)
    cm = acquire_shared_cache_lock(cache_dir, heartbeat_interval_s=1000)
    cm.__enter__()
    with open(lock_path, "w") as f:
        json.dump({"owner": "slurm:other-job", "heartbeat": time.time()}, f)
    cm.__exit__(None, None, None)
    assert os.path.exists(lock_path)
    with open(lock_path) as f:
        assert json.load(f)["owner"] == "slurm:other-job"


def test_owner_id_uses_slurm_job_id_when_set(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    assert _shared_cache_lock_owner_id() == "slurm:424242"


def test_owner_id_falls_back_to_host_pid(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    owner = _shared_cache_lock_owner_id()
    assert ":" in owner
    assert not owner.startswith("slurm:")
