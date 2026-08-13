"""
src/sae/neuronpedia_labels.py — Labels des features GemmaScope 2 via Neuronpedia.

Identifiants Neuronpedia :
  model_id = "gemma-3-12b-it"
  source   = "24-gemmascope-2-res-16k"    # {layer}-gemmascope-2-res-{width}
  feature  = index entier

Source des labels : lots JSONL gzip publics sur le bucket S3 "neuronpedia-datasets"
  https://neuronpedia-datasets.s3.amazonaws.com/v1/{model_id}/{source}/explanations/batch-{N}.jsonl.gz
(chaque ligne = un objet JSON avec au moins "index" et "description") -- le
chemin documenté par Neuronpedia pour le téléchargement en masse des jeux
d'explications, plus fiable que la route REST `/api/explanation/export`.
Couverture mesurée empiriquement : ~10k features labellisées sur 262144 pour
gemma-3-12b-it/24-gemmascope-2-res-262k (faible), bien plus dense en
proportion pour 16k (src/config.py), et ~98% pour
gemma-3-270m-it/12-gemmascope-2-res-65k.

⚠ Cluster hors-ligne : exécuter fetch_neuronpedia_labels() sur une machine avec
internet, copier le JSON dans CACHE_DIR ; le reste de la pipeline ne lit que le cache.
Le format du cache est identique au p1_saebench_judge_labels.json existant
(dict {feature_id: label}) → consommé tel quel par analyze_with_umap(feature_labels=...)
et cooccurrence.corpus_diff_stats(feature_labels=...).

⚠ Neuronpedia n'héberge pas forcément des explications pour tous les (modèle, source) —
notamment les modèles très récents (ex. gemma-3-270m-it). Si aucun lot n'est trouvé
(0 batch téléchargé), fetch_neuronpedia_labels() retourne un dict vide et log un warning ;
le reste de la pipeline dégrade proprement vers les labels du juge LLM local.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
from typing import Optional

import requests

log = logging.getLogger(__name__)

NP_BASE = "https://www.neuronpedia.org/api"
NP_DATASETS_BASE = "https://neuronpedia-datasets.s3.amazonaws.com/v1"


def np_source(layer: int = 24, width: str = "16k") -> str:
    return f"{layer}-gemmascope-2-res-{width}"


def fetch_neuronpedia_labels(
    model_id: str = "gemma-3-12b-it",
    layer: int = 24,
    width: str = "16k",
    cache_path: str = "neuronpedia_labels_l24_16k.json",
    api_key: Optional[str] = None,
    max_batches: int = 10_000,
) -> dict[int, str]:
    """Télécharge tous les lots batch-{N}.jsonl.gz du bucket public Neuronpedia
    Datasets pour (model_id, layer, width) et fusionne en dict {feature_id: label}.

    api_key n'est plus utilisé (bucket public) — conservé uniquement pour compatibilité
    de signature avec les appelants existants.
    """
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    source = np_source(layer, width)
    base_url = f"{NP_DATASETS_BASE}/{model_id}/{source}/explanations"
    labels: dict[int, str] = {}
    batch = 0
    while batch < max_batches:
        url = f"{base_url}/batch-{batch}.jsonl.gz"
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException as e:
            log.warning(f"[neuronpedia] Erreur réseau sur {url} : {e}")
            break
        if r.status_code == 404:
            break
        r.raise_for_status()
        n = 0
        with gzip.open(io.BytesIO(r.content), "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                idx = int(item["index"])
                if idx not in labels:
                    labels[idx] = item.get("description", "").strip()
                n += 1
        log.info(f"[neuronpedia] batch {batch} : {n} labels")
        batch += 1

    if not labels:
        log.warning(
            f"[neuronpedia] Aucun label trouvé pour model_id={model_id} source={source} "
            f"({base_url}/batch-0.jsonl.gz introuvable). Neuronpedia n'héberge "
            f"probablement pas d'explications pour ce (modèle, SAE) — "
            f"repli sur les labels du juge LLM local uniquement."
        )
        return {}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, ensure_ascii=False, indent=2)
    log.info(f"[neuronpedia] {len(labels)} labels ({batch} lots) → {cache_path}")
    return labels


