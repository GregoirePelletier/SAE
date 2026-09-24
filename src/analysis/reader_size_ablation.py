"""
reader_size_ablation.py — App I (arXiv:2512.10092v2, "Ablations on reader
model size", docs/archive/references/PDF_APPENDICES_EXTRACT.md lignes 698-712) : compare la
capacité de généralisation d'un SAE entraîné sur les activations d'un modèle
extracteur PETIT vs GRAND (ici : gemma-3-12b-it vs gemma-3-27b-it, RESULTS_
TESTS.md §82 -- le papier compare Llama-3.1-8B vs Llama-3.3-70B) via un score
F1 entre :
  - "predictions" : les documents sur lesquels le latent s'ACTIVE réellement
    (binarisé, max-pool documentaire > 0, même convention que
    `feature_selection_stratified_by_frequency`) ;
  - "ground truth" : la classification d'un JUGE LLM (le latent relabellisé,
    App. C, puis un juge classifie CHAQUE document du domaine comme ayant ou
    non la propriété décrite) -- réutilise `hypothesis_verifier.
    verify_hypotheses` avec une seule hypothèse (le label du latent) contre
    tous les documents du domaine, pas une fonction dédiée (même prompt/même
    mécanique que App K.1).

Ce module fournit UNIQUEMENT le calcul F1 + l'agrégation par palier de
taille -- l'orchestration (activation réelle par document, nécessite les
fragments token-level du corpus étudié) reste du ressort de l'appelant :
aucune extraction/lecture de fragments faite ici, pour rester utilisable
indépendamment de l'état du cache d'extraction (N1, docs/archive/audits/AUDIT_SAE_2026-08.md §8)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


def f1_activation_vs_judge(activation_binary: np.ndarray, judge_binary: np.ndarray) -> float:
    """F1 (App. I) entre l'activation réelle du latent ("predictions") et la
    classification du juge LLM ("ground truth") sur les mêmes documents.
    `zero_division=0` : un latent qui ne s'active jamais ET que le juge ne
    classe jamais positif donne F1=0 (pas d'exception), cohérent avec
    "aucune généralisation démontrée" plutôt qu'un cas dégénéré à traiter à
    part."""
    return float(f1_score(np.asarray(judge_binary, dtype=int),
                           np.asarray(activation_binary, dtype=int),
                           zero_division=0))


@dataclass
class ReaderSizeAblationResult:
    per_feature_f1: dict          # feature_id -> F1
    median_f1: float
    n_features: int


def compute_reader_size_ablation(
    activation_by_feature: dict,   # feature_id -> np.ndarray[n_docs] binaire (predictions)
    judge_by_feature: dict,        # feature_id -> np.ndarray[n_docs] binaire (ground truth, même ordre de docs)
) -> ReaderSizeAblationResult:
    """Agrège `f1_activation_vs_judge` sur un ensemble de latents échantillonnés
    (App. I : "100 latents actifs dans >10% du dataset étudié" -- l'échantillonnage
    lui-même est laissé à l'appelant, cf. `feature_selection_stratified_by_
    frequency`/un filtre direct sur la fréquence). `median_f1` (pas la moyenne) :
    le papier reporte des DISTRIBUTIONS de F1 (Figures 28-29, "F1 médian
    augmente..."), pas un scalaire agrégé par la moyenne -- rester comparable
    à leur lecture des résultats."""
    common = sorted(set(activation_by_feature) & set(judge_by_feature))
    per_feature_f1 = {
        f: f1_activation_vs_judge(activation_by_feature[f], judge_by_feature[f])
        for f in common
    }
    values = list(per_feature_f1.values())
    median_f1 = float(np.median(values)) if values else float("nan")
    return ReaderSizeAblationResult(per_feature_f1=per_feature_f1, median_f1=median_f1, n_features=len(common))
