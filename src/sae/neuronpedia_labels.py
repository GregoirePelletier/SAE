"""
src/sae/neuronpedia_labels.py — Labels des features GemmaScope 2 via Neuronpedia.

Identifiants Neuronpedia :
  model_id = "gemma-3-12b-it"
  source   = "24-gemmascope-2-res-262k"    # {layer}-gemmascope-2-res-{width}
  feature  = index entier

Deux voies, dans l'ordre :
  1. Export bulk (endpoint /api/explanation/export) — un appel, toutes les features.
  2. Fallback par feature (GET /api/feature/{model}/{source}/{idx}) — pour trous.

⚠ Cluster hors-ligne : exécuter fetch_neuronpedia_labels() sur une machine avec
internet, copier le JSON dans CACHE_DIR ; le reste de la pipeline ne lit que le cache.
Le format du cache est identique au p1_saebench_judge_labels.json existant
(dict {feature_id: label}) → consommé tel quel par analyze_with_umap(feature_labels=...)
et cooccurrence.corpus_diff_stats(feature_labels=...).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests

NP_BASE = "https://www.neuronpedia.org/api"


def np_source(layer: int = 24, width: str = "262k") -> str:
    return f"{layer}-gemmascope-2-res-{width}"


def fetch_neuronpedia_labels(
    model_id: str = "gemma-3-12b-it",
    layer: int = 24,
    width: str = "262k",
    cache_path: str = "neuronpedia_labels_l24_262k.json",
    api_key: Optional[str] = None,
) -> dict[int, str]:
    """Export bulk des explications auto-interp (dict feature_id → label)."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    source = np_source(layer, width)
    headers = {"x-api-key": api_key} if api_key else {}
    r = requests.get(
        f"{NP_BASE}/explanation/export",
        params={"modelId": model_id, "saeId": source},
        headers=headers,
        timeout=300,
    )
    r.raise_for_status()
    labels: dict[int, str] = {}
    for item in r.json():
        # champs export : index (str), description ; garde la 1re explication par feature
        idx = int(item["index"])
        if idx not in labels:
            labels[idx] = item.get("description", "").strip()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, ensure_ascii=False)
    print(f"[neuronpedia] {len(labels)} labels → {cache_path}")
    return labels


def fetch_single_feature(
    idx: int,
    model_id: str = "gemma-3-12b-it",
    layer: int = 24,
    width: str = "262k",
    api_key: Optional[str] = None,
    sleep: float = 0.2,
) -> dict:
    """Détail d'une feature : explications + top activating examples (pour la visu)."""
    headers = {"x-api-key": api_key} if api_key else {}
    r = requests.get(f"{NP_BASE}/feature/{model_id}/{np_source(layer, width)}/{idx}",
                     headers=headers, timeout=60)
    r.raise_for_status()
    time.sleep(sleep)  # politesse rate-limit
    return r.json()


def neuronpedia_deep_link(idx: int, model_id: str = "gemma-3-12b-it",
                          layer: int = 24, width: str = "262k") -> str:
    """URL de la page feature — injectée dans les hovers Plotly (visu refondue)."""
    return f"https://www.neuronpedia.org/{model_id}/{np_source(layer, width)}/{idx}"


def merge_with_judge_labels(np_labels: dict[int, str],
                            judge_labels_path: str) -> dict[int, str]:
    """Priorité : labels juge internes (contexte français EDF) > Neuronpedia > F{idx}.
    Les features d'extension (idx >= 262144) n'existent pas sur Neuronpedia → juge only."""
    merged = dict(np_labels)
    if os.path.exists(judge_labels_path):
        with open(judge_labels_path, encoding="utf-8") as f:
            merged.update({int(k): v for k, v in json.load(f).items()})
    return merged