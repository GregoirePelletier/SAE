import importlib
import os


def test_layer_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_SIZE", "12b")
    monkeypatch.setenv("LAYER", "31")
    import src.config as config
    importlib.reload(config)
    assert config.LAYER == 31
    monkeypatch.delenv("LAYER", raising=False)
    monkeypatch.delenv("MODEL_SIZE", raising=False)
    importlib.reload(config)
    assert config.LAYER == 31  # défaut preset 12b (RESULTS_TESTS.md §51)
