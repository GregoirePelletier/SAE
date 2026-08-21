"""Teste src/data/preparation.py::prepare_domain_dataset/sample_fineweb2_chunks
en streaming=True (AUDIT_SAE_2026-08.md, item A1 : ces deux appels chargeaient
le parquet FineWeb2 entier en RAM avant filtrage -- `sample_fineweb2_chunks`
est appelée en boucle sur 18 shards de 4,6 Go pour construire le filler, c'est
le chemin de production le plus lourd). Le correctif change uniquement le
paramètre `streaming` du `load_dataset` interne ; ce test vérifie (a) que le
résultat filtré/chunké est inchangé par rapport au comportement non-streaming
qu'il remplace, (b) que l'arrêt anticipé (`break` dès `n_target` atteint)
fonctionne bien sur un `IterableDataset` (jamais exercé avant ce correctif,
seul le repli Wikipedia l'utilisait), et (c) que l'appel passe bien
`streaming=True` -- garde-fou contre une régression silencieuse vers
`streaming=False`.
"""
import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.preparation import prepare_domain_dataset, sample_fineweb2_chunks
from datasets import load_dataset as _real_load_dataset

# Garde-fou : si n_target n'est pas atteint par le fixture local,
# prepare_domain_dataset retombe sur un fetch réseau de wikimedia/wikipedia
# (cf. AUDIT_SAE_2026-08.md, item D1) -- offline forcé pour que toute erreur
# de calibration du fixture échoue vite et proprement plutôt que de tenter un
# accès réseau depuis ce test CPU.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _write_parquet(tmp_path, rows):
    path = tmp_path / "fineweb2_fixture.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def _spy_load_dataset(calls):
    def _wrapped(*args, **kwargs):
        calls.append(kwargs.get("streaming"))
        return _real_load_dataset(*args, **kwargs)
    return _wrapped


def test_prepare_domain_dataset_streaming_true_and_filters_correctly(tmp_path):
    rows = [{"text": f"Rien à voir ici, du contenu générique sans rapport numéro {i}.", "url": f"https://x/{i}"}
            for i in range(5)]
    rows += [{"text": f"Facture électricité et compteur Linky de la maison numéro {i} " * 5,
              "url": f"https://x/energy/{i}"} for i in range(3)]
    path = _write_parquet(tmp_path, rows)

    calls = []
    with patch("src.data.preparation.load_dataset", side_effect=_spy_load_dataset(calls)):
        texts = prepare_domain_dataset(
            keywords=["électricité", "compteur"], domain_name="energy", n_target=3,
            local_dataset_path=path, use_fineweb2=True,
        )

    assert calls and calls[0] is True, "doit appeler load_dataset avec streaming=True"
    assert len(texts) == 3
    assert all("Facture" in t for t in texts)


def test_prepare_domain_dataset_streaming_stops_early(tmp_path):
    # 200 lignes correspondantes (contenu UNIQUE par ligne : la dédup par hash de
    # chunk collapse sinon tout à 1 résultat), n_target=4 -> ne doit en retenir
    # que 4 malgré un IterableDataset qui ne permet plus l'indexation a priori.
    rows = [{"text": f"Facture électricité Linky numéro {i} " * 5, "url": f"https://x/{i}"}
            for i in range(200)]
    path = _write_parquet(tmp_path, rows)

    texts = prepare_domain_dataset(
        keywords=["électricité"], domain_name="energy", n_target=4,
        local_dataset_path=path, use_fineweb2=True,
    )
    assert len(texts) == 4


def test_sample_fineweb2_chunks_streaming_true_and_unfiltered(tmp_path):
    rows = [{"text": f"Un paragraphe de contenu générique quelconque numéro {i}, " * 3}
            for i in range(20)]
    path = _write_parquet(tmp_path, rows)

    calls = []
    with patch("src.data.preparation.load_dataset", side_effect=_spy_load_dataset(calls)):
        texts = sample_fineweb2_chunks(n_target=5, local_dataset_path=path)

    assert calls and calls[0] is True, "doit appeler load_dataset avec streaming=True"
    assert len(texts) == 5  # aucun filtre thématique : les 5 premiers chunks valides suffisent


def test_sample_fineweb2_chunks_missing_path_returns_empty():
    assert sample_fineweb2_chunks(n_target=5, local_dataset_path=None) == []


def test_sample_fineweb2_chunks_streaming_stops_early(tmp_path):
    rows = [{"text": f"Contenu générique long numéro {i} " * 10} for i in range(500)]
    path = _write_parquet(tmp_path, rows)
    texts = sample_fineweb2_chunks(n_target=7, local_dataset_path=path)
    assert len(texts) == 7
