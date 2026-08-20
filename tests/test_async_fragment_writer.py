"""Teste src/storage/fragment_store.py::AsyncFragmentWriter (audit perf G2,
AUDIT_SAE_2026-08.md §2.2) -- écriture de fragments en arrière-plan. La
propriété critique n'est pas "ça écrit" (torch.save est déjà testé ailleurs)
mais "flush() garantit que tout ce qui a été soumis avant est bien sur
disque" : c'est l'invariant dont dépend la reprise (R1) pour ne jamais avancer
un checkpoint au-delà de ce qui est réellement persisté."""
import os
import time

import torch

from src.storage.fragment_store import AsyncFragmentWriter, save_fragment, load_fragment


def test_flush_guarantees_all_submitted_writes_are_on_disk(tmp_path):
    writer = AsyncFragmentWriter()
    n = 30
    for i in range(n):
        writer.submit(str(tmp_path / f"f{i}.pt"), {"value": i})
    writer.flush()

    for i in range(n):
        assert os.path.exists(tmp_path / f"f{i}.pt")
        assert torch.load(tmp_path / f"f{i}.pt", weights_only=False)["value"] == i


def test_close_flushes_and_stops_the_worker_thread(tmp_path):
    writer = AsyncFragmentWriter()
    writer.submit(str(tmp_path / "a.pt"), {"value": 1})
    writer.close()
    assert os.path.exists(tmp_path / "a.pt")
    assert not writer._thread.is_alive()


def test_save_fragment_with_writer_lands_on_disk_after_flush(tmp_path):
    """Intégration avec save_fragment() : mêmes garanties que le chemin
    synchrone, juste différées jusqu'au flush()."""
    writer = AsyncFragmentWriter()
    acts = torch.zeros(4, 6)
    acts[0, 1] = 0.5
    acts[2, 3] = 0.7
    save_fragment(str(tmp_path), 0, token_strings=["a", "b", "c", "d"],
                  acts_dense=acts, d_total=6, writer=writer)

    # Rien ne garantit que le fichier existe déjà à cet instant précis --
    # c'est exactement le point de flush().
    writer.flush()

    frag = load_fragment(str(tmp_path), 0)
    assert frag["shape"] == (4, 6)
    assert frag["token_strings"] == ["a", "b", "c", "d"]


def test_error_in_background_write_surfaces_on_next_submit_or_flush(tmp_path):
    """Une écriture ratée (répertoire cible inexistant) ne doit jamais
    disparaître silencieusement -- elle doit remonter à l'appelant."""
    writer = AsyncFragmentWriter()
    bad_path = str(tmp_path / "does_not_exist_dir" / "f.pt")
    writer.submit(bad_path, {"value": 1})
    try:
        writer.flush()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_writes_actually_happen_concurrently_with_caller(tmp_path):
    """Sanity check faible mais utile : submit() ne doit pas bloquer le temps
    d'une écriture complète -- sinon "asynchrone" ne veut rien dire."""
    writer = AsyncFragmentWriter(maxsize=8)
    t0 = time.perf_counter()
    for i in range(4):
        writer.submit(str(tmp_path / f"g{i}.pt"), {"value": i})
    submit_elapsed = time.perf_counter() - t0
    writer.flush()
    # Les 4 submit() doivent revenir quasi instantanément (pas le temps réel
    # d'écrire 4 fichiers un par un de façon synchrone).
    assert submit_elapsed < 1.0
