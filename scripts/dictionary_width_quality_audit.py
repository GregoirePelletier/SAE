"""
scripts/dictionary_width_quality_audit.py — Phase 1.2 de l'audit méthodologique
(cf. plan approuvé 2026-08-07).

Contexte : `RESULTS_TESTS.md:1017-1032` (`config.py:74-91`) ne compare 16k vs
65k que sur la COUVERTURE de labels Neuronpedia (82,6% vs 87,8%) — jamais sur
leur qualité monosémantique. Hypothèse explicite de l'utilisateur : le
dictionnaire 65k a plus de labels au total, mais une partie n'est pas
monosémantique (ex. mélange de concepts sans rapport dans un même label,
contrairement à 16k). Vérifié qualitativement avant d'écrire ce script
(échantillon de labels contenant une virgule) : 65k contient des labels
clairement incohérents ("FCC ID, agricultural professionals, topological
equivalence" ; "than the, n be, y the, it in" — fragment de titre, pas un
concept) plus fréquemment qu'à l'œil sur 16k.

Deux mesures complémentaires, sur la POPULATION COMPLÈTE des deux caches
(`local_data/neuronpedia_labels/neuronpedia_labels_24-gemmascope-2-res-{16k,65k}.json`,
aucun GPU, aucune extraction) :

  1. `frac_multi_part` — fraction de labels contenant ≥2 "parties" (split sur
     virgule/point-virgule/" or "/" and ") — proxy lexical brut, calculé sur
     la POPULATION ENTIÈRE (n=13535 / n=57551, pas un échantillon).
  2. Cohérence sémantique des labels multi-parties — échantillon de
     N_SAMPLE_COHERENCE labels multi-parties par largeur, embeddings bge-m3
     (`src/sae/saev5.py::_embed_bge_m3`, déjà utilisé ailleurs dans le projet
     pour ce rôle -- dissimilarité de labels, cf. `find_interesting_pairs`),
     similarité cosinus moyenne entre parties d'un même label. Un label dont
     les parties sont peu similaires entre elles (< SIM_THRESHOLD) est
     compté "incohérent" — proxy quantitatif de polysémanticité.

Tests utilisés (`src.analysis.stats`, Phase 1.3) : test à deux proportions
indépendantes + h de Cohen pour `frac_multi_part` (population entière → IC
très étroits, à interpréter avec prudence — cf. commentaire dans le rapport)
et pour le taux d'incohérence échantillonné ; Mann-Whitney U (scipy, pas de
duplication -- `src.analysis.stats` ne réimplémente que ce qui est
spécifique au projet) sur la distribution continue des scores de cohérence.

Usage (CPU uniquement, bge-m3 tourne sur CPU si pas de CUDA disponible) :
    PYTHONPATH=. .venv/bin/python scripts/dictionary_width_quality_audit.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))

from src.analysis.stats import proportion_with_ci, two_proportion_test

LABELS_DIR = "./local_data/neuronpedia_labels"
LAYER = 24
WIDTHS = ["16k", "65k"]
N_SAMPLE_COHERENCE = 200
SIM_THRESHOLD = 0.3   # seuil arbitraire mais explicite -- cf. docstring
SEED = 42
OUT_PATH = os.path.join(LABELS_DIR, "dictionary_width_quality_results.json")

_SPLIT_RE = re.compile(r",|;|\bor\b|\band\b", flags=re.IGNORECASE)
_TRIVIAL = {"a", "an", "the", "of", "to", "in", "on", "or", "and", ""}


def load_labels(width: str) -> dict[str, str]:
    path = os.path.join(LABELS_DIR, f"neuronpedia_labels_{LAYER}-gemmascope-2-res-{width}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_parts(label: str) -> list[str]:
    """Découpe un label en parties candidates. Filtre les fragments triviaux
    (articles/conjonctions isolés, chaînes vides après split) qui ne sont pas
    de vrais "concepts" listés."""
    parts = [p.strip().lower() for p in _SPLIT_RE.split(label)]
    return [p for p in parts if p and p not in _TRIVIAL and len(p) > 1]


def main() -> None:
    random.seed(SEED)
    labels = {w: load_labels(w) for w in WIDTHS}

    print("=" * 78)
    print(" 1. frac_multi_part — POPULATION ENTIÈRE (pas un échantillon)")
    print("=" * 78)
    multi_part_stats = {}
    for w in WIDTHS:
        vals = list(labels[w].values())
        n_multi = sum(1 for v in vals if len(split_parts(v)) >= 2)
        res = proportion_with_ci(n_multi, len(vals))
        multi_part_stats[w] = {"n_total": len(vals), "n_multi_part": n_multi,
                                "rate": res.rate, "ci_low": res.ci_low, "ci_high": res.ci_high}
        print(f"  {w}: {n_multi}/{len(vals)} = {res.rate:.1%}  IC95%=[{res.ci_low:.1%}, {res.ci_high:.1%}]")

    two_prop = two_proportion_test(
        multi_part_stats["65k"]["n_multi_part"], multi_part_stats["65k"]["n_total"],
        multi_part_stats["16k"]["n_multi_part"], multi_part_stats["16k"]["n_total"],
    )
    print(f"  65k vs 16k : diff={two_prop.diff:+.1%}  z={two_prop.z:.2f}  p={two_prop.p:.2e}  "
          f"h de Cohen={two_prop.cohens_h:.3f}")
    print("  NOTE : population entière -> p sera quasi-toujours significatif même pour un\n"
          "  écart minime (puissance ~infinie). Le h de Cohen (taille d'effet, indépendant\n"
          "  de n) est la lecture pertinente ici, pas le p seul.")

    print("\n" + "=" * 78)
    print(f" 2. Cohérence sémantique (bge-m3) — échantillon n={N_SAMPLE_COHERENCE}/largeur")
    print("=" * 78)
    from src.sae.saev5 import _embed_bge_m3  # import tardif : charge bge-m3 (CPU si pas de CUDA)

    coherence_stats = {}
    coherence_scores = {}
    for w in WIDTHS:
        multi_part_labels = [v for v in labels[w].values() if len(split_parts(v)) >= 2]
        sample = random.sample(multi_part_labels, min(N_SAMPLE_COHERENCE, len(multi_part_labels)))
        parts_per_label = [split_parts(lbl) for lbl in sample]
        flat_parts = [p for parts in parts_per_label for p in parts]
        print(f"  {w}: embedding de {len(flat_parts)} parties ({len(sample)} labels)...")
        embs = _embed_bge_m3(flat_parts).numpy()

        scores = []
        offset = 0
        for parts in parts_per_label:
            k = len(parts)
            sub = embs[offset:offset + k]
            offset += k
            sim = sub @ sub.T
            iu = np.triu_indices(k, k=1)
            scores.append(float(sim[iu].mean()) if len(iu[0]) else 1.0)
        coherence_scores[w] = scores
        n_incoherent = sum(1 for s in scores if s < SIM_THRESHOLD)
        res = proportion_with_ci(n_incoherent, len(scores))
        coherence_stats[w] = {
            "n_sampled": len(scores), "mean_coherence": float(np.mean(scores)),
            "median_coherence": float(np.median(scores)),
            "n_incoherent_below_threshold": n_incoherent,
            "incoherent_rate": res.rate, "ci_low": res.ci_low, "ci_high": res.ci_high,
        }
        print(f"    cohérence moyenne={np.mean(scores):.3f}  médiane={np.median(scores):.3f}  "
              f"incohérents (sim<{SIM_THRESHOLD})={n_incoherent}/{len(scores)} "
              f"({res.rate:.1%}, IC95%=[{res.ci_low:.1%},{res.ci_high:.1%}])")

    u_stat, mw_p = mannwhitneyu(coherence_scores["65k"], coherence_scores["16k"], alternative="less")
    print(f"\n  Mann-Whitney U (H1: cohérence 65k < 16k) : U={u_stat:.0f}  p={mw_p:.4f}")

    incoh_two_prop = two_proportion_test(
        coherence_stats["65k"]["n_incoherent_below_threshold"], coherence_stats["65k"]["n_sampled"],
        coherence_stats["16k"]["n_incoherent_below_threshold"], coherence_stats["16k"]["n_sampled"],
    )
    print(f"  Taux d'incohérence 65k vs 16k : diff={incoh_two_prop.diff:+.1%}  "
          f"z={incoh_two_prop.z:.2f}  p={incoh_two_prop.p:.3f}  h de Cohen={incoh_two_prop.cohens_h:.3f}")

    results = {
        "multi_part_full_population": multi_part_stats,
        "multi_part_two_proportion_test": vars(two_prop),
        "coherence_sample": coherence_stats,
        "coherence_mannwhitney_u": {"statistic": float(u_stat), "p": float(mw_p),
                                     "alternative": "65k < 16k"},
        "coherence_incoherent_rate_two_proportion_test": vars(incoh_two_prop),
        "sim_threshold": SIM_THRESHOLD, "n_sample_coherence": N_SAMPLE_COHERENCE, "seed": SEED,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
