"""Teste src/sae/sae_shared.py::compute_activation_cache_key -- cache
d'activations partagé entre runs qui ne diffèrent que par K_EXTRA/D_EXTRA/
EPOCHS_EXTRA (downstream de l'extraction, docs/archive/audits/AUDIT_SAE_2026-08.md)."""
from src.sae.sae_shared import compute_activation_cache_key


BASE_KWARGS = dict(
    train_texts=["mail un", "mail deux"],
    volume_filler_texts=["filler un"],
    test_texts=["test un"],
    diff_texts=[],
    model_id="/models/gemma-3-12b-it",
    layer=31,
    hook_type="resid_post",
    dtype="bf16",
    sae_id="layer_31_width_16k_l0_medium",
    n_tokens_extra_train=25_000_000,
    max_length=512,
    sigma_clip=4.0,
    skip_first_content_token=True,
)


def test_same_inputs_same_key():
    k1 = compute_activation_cache_key(**BASE_KWARGS)
    k2 = compute_activation_cache_key(**BASE_KWARGS)
    assert k1 == k2


def test_downstream_only_params_not_in_signature():
    # K_EXTRA/D_EXTRA/EPOCHS_EXTRA n'apparaissent jamais dans les kwargs :
    # deux runs qui ne diffèrent que par ces paramètres partagent la clé --
    # documenté par absence, la clé ne les prend pas en argument du tout.
    import inspect
    sig = inspect.signature(compute_activation_cache_key)
    names = set(sig.parameters.keys())
    assert "k_extra" not in names
    assert "d_extra" not in names
    assert "epochs_extra" not in names


def test_different_corpus_different_key():
    kwargs = dict(BASE_KWARGS)
    kwargs["train_texts"] = ["mail trois", "mail deux"]
    k1 = compute_activation_cache_key(**BASE_KWARGS)
    k2 = compute_activation_cache_key(**kwargs)
    assert k1 != k2


def test_different_layer_different_key():
    kwargs = dict(BASE_KWARGS)
    kwargs["layer"] = 24
    k1 = compute_activation_cache_key(**BASE_KWARGS)
    k2 = compute_activation_cache_key(**kwargs)
    assert k1 != k2


def test_different_n_tokens_different_key():
    kwargs = dict(BASE_KWARGS)
    kwargs["n_tokens_extra_train"] = 5_000_000
    k1 = compute_activation_cache_key(**BASE_KWARGS)
    k2 = compute_activation_cache_key(**kwargs)
    assert k1 != k2


def test_different_model_id_different_key():
    kwargs = dict(BASE_KWARGS)
    kwargs["model_id"] = "/models/gemma-3-27b-it"
    k1 = compute_activation_cache_key(**BASE_KWARGS)
    k2 = compute_activation_cache_key(**kwargs)
    assert k1 != k2


def test_key_is_deterministic_short_hex_string():
    k = compute_activation_cache_key(**BASE_KWARGS)
    assert isinstance(k, str)
    assert len(k) == 20
    int(k, 16)  # lève ValueError si pas hexadécimal


def test_different_max_length_different_key():
    # N7/N8 (docs/archive/audits/AUDIT_SAE_2026-08.md §8) : max_length affecte le corpus
    # RÉELLEMENT vu à l'extraction (troncature), doit invalider la clé.
    kwargs = dict(BASE_KWARGS)
    kwargs["max_length"] = 2048
    k1 = compute_activation_cache_key(**BASE_KWARGS)
    k2 = compute_activation_cache_key(**kwargs)
    assert k1 != k2


def test_different_sigma_clip_different_key():
    kwargs = dict(BASE_KWARGS)
    kwargs["sigma_clip"] = 3.0
    k1 = compute_activation_cache_key(**BASE_KWARGS)
    k2 = compute_activation_cache_key(**kwargs)
    assert k1 != k2


def test_different_skip_first_content_token_different_key():
    kwargs = dict(BASE_KWARGS)
    kwargs["skip_first_content_token"] = False
    k1 = compute_activation_cache_key(**BASE_KWARGS)
    k2 = compute_activation_cache_key(**kwargs)
    assert k1 != k2
