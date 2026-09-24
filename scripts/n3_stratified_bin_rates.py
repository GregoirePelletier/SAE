"""
scripts/n3_stratified_bin_rates.py -- N3 (docs/archive/audits/AUDIT_SAE_2026-08.md §8) : 89,3%
(sélection stratifiée, arme mixte, §79/RESULTS_TESTS.md) n'est pas plus
comparable au papier que 45,3% (magnitude), dans l'autre sens -- le
stratifié échantillonne un nombre à peu près fixe de features par bin de
fréquence quelle que soit la taille du bin, sur-pondérant les bins rares
(plus nombreux mais plus spécifiques) par rapport à toute distribution
réellement utilisée.

Ne relance AUCUN juge : recalcule uniquement les métadonnées de bin (fréquence,
taille de strate) des 150 features déjà sélectionnées ET jugées dans
`results_v10_emails_main/cache/b2_stratified_selection_rejudge.json` (§79),
via `feature_selection_stratified_by_frequency(..., return_bin_info=True)`
appelée avec exactement les mêmes arguments (même SEED, mêmes fragments) que
lors de la sélection originale -- reproductible bit-exact (aucun autre appel
au module `random` global entre le seed et cet appel, cf. docstring de la
fonction). Publie le taux d'interprétabilité PAR BIN (au lieu du seul
scalaire agrégé) et un estimateur repondéré (Horvitz-Thompson,
`src/analysis/stats.py`).

CPU-only, aucun modèle chargé -- lecture de fragments déjà en cache
uniquement (R du CLAUDE.md sur les calculs frontend : ce script lit des
tenseurs réels sur ~500 documents, donc PAS un "config check borné" au sens
de CLAUDE.md, doit passer par sbatch même si le coût réel est faible).

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/n3_stratified_bin_rates.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAVE_DIR, CORPUS_SPLIT_SEED, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH
from src.sae.judge import feature_selection_stratified_by_frequency
from src.data.preparation import build_email_train_test_corpus
from src.storage.fragment_store import resolve_extension_fragments_dir
from src.analysis.stats import horvitz_thompson_mean

SEED = int(os.environ.get("SEED", "42"))
# 150 : nombre réel de features sélectionnées par b2_stratified_selection_rejudge.py
# lors du run qui a produit le chiffre 89,3%/134/150 cité en §79/§81 -- PAS le
# défaut de N_FEATURES_TO_LABEL (10, src/config.py), vérifié empiriquement
# (len(b2["results"]) == 150 dans le cache existant).
N_FEATURES = 150

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
B2_CACHE = os.path.join(CACHE_DIR, "b2_stratified_selection_rejudge.json")
TOKEN_FRAGMENTS_DIR = resolve_extension_fragments_dir(CACHE_DIR)
OUT_PATH = os.path.join(CACHE_DIR, "n3_stratified_bin_rates.json")


def main() -> None:
    with open(B2_CACHE, encoding="utf-8") as f:
        b2 = json.load(f)
    judged = b2["results"]
    ref_indices = [int(k) for k in judged.keys()]

    # d_core/D_EXTRA lus directement du checkpoint frozen-core de CE SAVE_DIR
    # (state_dict, forme exacte) -- pas de src.config.D_EXTRA (pourrait avoir
    # dérivé depuis §79) ni d'estimation depuis min/max(ref_indices) (les
    # indices SÉLECTIONNÉS ne couvrent pas nécessairement toute la plage
    # [d_core, d_total) -- un bin rare aux extrémités peut n'avoir jamais été
    # tiré).
    _frozen_core_ckpts = [f for f in os.listdir(SAVE_DIR) if f.startswith("p1_frozen_core_d")]
    assert len(_frozen_core_ckpts) == 1, (
        f"attendu exactement 1 checkpoint p1_frozen_core_d*.pt dans {SAVE_DIR}, "
        f"trouvé {_frozen_core_ckpts} -- d_core/D_EXTRA ambigus."
    )
    _ckpt = torch.load(os.path.join(SAVE_DIR, _frozen_core_ckpts[0]), map_location="cpu", weights_only=False)
    d_core = _ckpt["state_dict"]["core_sae.W_dec"].shape[0]
    d_extra = _ckpt["config"]["d_extra"]
    d_total = d_core + d_extra
    del _ckpt
    print(f"[n3] plage extension [{d_core}, {d_total}) -- lue du checkpoint frozen-core de {SAVE_DIR}")

    random.seed(SEED)   # même position dans le flux RNG global que b2_stratified_selection_rejudge.py
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train = len(train_texts)
    print(f"[n3] n_train={n_train}")

    selected, bin_info = feature_selection_stratified_by_frequency(
        TOKEN_FRAGMENTS_DIR, list(range(n_train)), d_total, N_FEATURES,
        lo=d_core, hi=d_total, seed=SEED, return_bin_info=True,
    )
    if set(selected) != set(ref_indices):
        print(f"[n3] ATTENTION : {len(set(selected) ^ set(ref_indices))} features de différence "
              "entre la reproduction et le cache de référence -- la reproduction n'est PAS "
              "bit-exacte (fragments modifiés depuis §79 ? D_EXTRA différent ?). "
              "Les métadonnées de bin ci-dessous restent celles de la reproduction, "
              "pas garanties alignées feature-par-feature avec le cache jugé.")
    else:
        print(f"[n3] reproduction bit-exacte confirmée : {len(selected)}/{len(selected)} features identiques.")

    # Taux par bin : n'utilise que les features présentes dans les DEUX (jugées
    # ET reproduites avec métadonnées) -- l'intersection est les 150 sauf
    # divergence signalée ci-dessus.
    common = set(selected) & set(ref_indices)
    per_bin: dict[int, dict] = {}
    values, probs = [], []
    for f in sorted(common):
        info = bin_info[f]
        b = info["bin"]
        interp = judged[str(f)].get("interp_score", 0)
        per_bin.setdefault(b, {"n": 0, "n_interp": 0, "bin_population": info["bin_population"],
                                "bin_n_sampled": info["bin_n_sampled"], "freq_range": [info["freq"], info["freq"]]})
        per_bin[b]["n"] += 1
        per_bin[b]["n_interp"] += interp
        per_bin[b]["freq_range"][0] = min(per_bin[b]["freq_range"][0], info["freq"])
        per_bin[b]["freq_range"][1] = max(per_bin[b]["freq_range"][1], info["freq"])
        # π_i approximée par bin_n_sampled/bin_population (constante au sein
        # d'une strate) -- cf. docstring horvitz_thompson_mean.
        pi = info["bin_n_sampled"] / info["bin_population"]
        values.append(float(interp))
        probs.append(pi)

    for b in sorted(per_bin):
        d = per_bin[b]
        d["interp_rate"] = d["n_interp"] / d["n"]

    raw_rate = sum(values) / len(values)
    ht_rate = horvitz_thompson_mean(values, probs)

    summary = {
        "n_features": len(common),
        "raw_interp_rate": raw_rate,
        "raw_interp_rate_pct": f"{raw_rate * 100:.1f}%",
        "horvitz_thompson_interp_rate": ht_rate,
        "horvitz_thompson_interp_rate_pct": f"{ht_rate * 100:.1f}%",
        "n_bins_populated": len(per_bin),
        "reference_scalar_cited_in_report": {"label": "89,3% (mixte, stratifié, §79/§81)",
                                              "value": sum(1 for f in common if judged[str(f)].get("interp_score") == 1) / len(common)},
    }

    print("\n" + "=" * 60)
    print("[n3] Taux d'interprétabilité PAR BIN de fréquence (log-espacés) :")
    for b in sorted(per_bin):
        d = per_bin[b]
        print(f"  bin {b}: {d['n_interp']}/{d['n']} = {d['interp_rate']*100:.1f}% "
              f"(freq [{d['freq_range'][0]:.4f}, {d['freq_range'][1]:.4f}], "
              f"population {d['bin_population']}, échantillonné {d['bin_n_sampled']})")
    print("\n[n3] Résumé :")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_bin": per_bin}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] {OUT_PATH}")


if __name__ == "__main__":
    main()
