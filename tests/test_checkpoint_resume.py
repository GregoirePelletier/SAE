"""Teste src/storage/checkpoint.py -- brique de reprise après coupure (R1,
AUDIT_SAE_2026-08.md §2.3/§4.3) partagée entre saev5.py (Pipeline 1) et
phrase_sae.py (Pipeline 2)."""
import os
import signal

import pytest

from src.storage.checkpoint import (
    checkpoint_path, read_checkpoint, write_checkpoint, clear_checkpoint, GracefulShutdown,
)


def test_read_checkpoint_missing_file_returns_none(tmp_path):
    assert read_checkpoint(str(tmp_path / "does_not_exist.json")) is None


def test_write_then_read_roundtrip(tmp_path):
    path = str(tmp_path / "p1_extraction.progress.json")
    write_checkpoint(path, next_doc_idx=1234, n_residuals_seen=5000, n_residuals_collected=4800)
    got = read_checkpoint(path)
    assert got == {"next_doc_idx": 1234, "n_residuals_seen": 5000, "n_residuals_collected": 4800}


def test_write_is_atomic_no_partial_tmp_file_left_behind(tmp_path):
    path = str(tmp_path / "ckpt.json")
    write_checkpoint(path, x=1)
    remaining = os.listdir(tmp_path)
    assert remaining == ["ckpt.json"]  # aucun .tmp<pid> résiduel


def test_overwrite_replaces_previous_checkpoint(tmp_path):
    path = str(tmp_path / "ckpt.json")
    write_checkpoint(path, next_doc_idx=100)
    write_checkpoint(path, next_doc_idx=200)
    assert read_checkpoint(path) == {"next_doc_idx": 200}


def test_checkpoint_path_is_deterministic_and_namespaced(tmp_path):
    p1 = checkpoint_path(str(tmp_path), "p1_extraction")
    p2 = checkpoint_path(str(tmp_path), "p2_embeddings")
    assert p1 != p2
    assert checkpoint_path(str(tmp_path), "p1_extraction") == p1  # déterministe


def test_clear_checkpoint_removes_file(tmp_path):
    path = str(tmp_path / "ckpt.json")
    write_checkpoint(path, next_doc_idx=1)
    assert os.path.exists(path)
    clear_checkpoint(path)
    assert not os.path.exists(path)


def test_clear_checkpoint_missing_file_does_not_raise(tmp_path):
    clear_checkpoint(str(tmp_path / "never_existed.json"))  # ne doit pas lever


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """GracefulShutdown.install() pose des handlers process-wide -- ne pas les
    laisser fuiter vers le reste de la suite de tests (ou vers pytest
    lui-même, qui peut avoir sa propre gestion de SIGTERM)."""
    prev_term = signal.getsignal(signal.SIGTERM)
    prev_usr1 = signal.getsignal(signal.SIGUSR1)
    yield
    signal.signal(signal.SIGTERM, prev_term)
    signal.signal(signal.SIGUSR1, prev_usr1)
    GracefulShutdown.requested = False


def test_graceful_shutdown_flag_set_by_sigterm():
    GracefulShutdown.install()
    assert GracefulShutdown.requested is False
    os.kill(os.getpid(), signal.SIGTERM)
    assert GracefulShutdown.requested is True


def test_graceful_shutdown_flag_set_by_sigusr1():
    GracefulShutdown.install()
    assert GracefulShutdown.requested is False
    os.kill(os.getpid(), signal.SIGUSR1)
    assert GracefulShutdown.requested is True


def test_graceful_shutdown_install_resets_flag():
    GracefulShutdown.requested = True
    GracefulShutdown.install()
    assert GracefulShutdown.requested is False
