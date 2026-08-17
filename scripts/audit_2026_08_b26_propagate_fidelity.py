"""
scripts/audit_2026_08_b26_propagate_fidelity.py — B.26 point 4 (le dernier
point encore ouvert de la réouverture B.26, cf. `docs/AUDIT_2026-08.md`) :
propage le correctif réellement corrigé de `INTENT_KEYWORDS_FR`
(`FIXED_PATTERNS_V2`, cf. `audit_2026_08_b26_round2_fix.py`, déjà appliqué à
`intent_urgency_probe.py` dans RESULTS_TESTS.md §60) aux trois derniers
consommateurs de `intent_*` encore non rejoués avec les labels corrigés :
`explanation_fidelity_test.py`, `steering_fidelity_test.py` (CPU/GPU léger,
activations déjà en cache) et `latent_retrieval_precision_eval.py` (GPU léger,
"quelques minutes" selon son propre docstring -- entraîne un petit
PhraseLevelSAE dédié mais aucune extraction Gemma-3-12B).

Ne modifie NI les trois scripts originaux NI `src/data/dataset.py` en
production : monkey-patch `src.data.dataset.INTENT_KEYWORDS_FR` en mémoire
avant d'appeler leurs `main()` respectifs (même technique que
`audit_2026_08_b26_round2_fix.py`), puis renomme immédiatement leur fichier
de sortie (écrit par construction à un chemin fixe codé en dur dans ces
scripts) pour ne PAS écraser le résultat original déjà publié -- l'original
est sauvegardé avant l'appel et restauré après.

Usage : sbatch slurm/validation/run_audit_b26_propagate_fidelity.slurm
"""
from __future__ import annotations

import os
import shutil

import src.data.dataset as dataset_mod
from src.config import SAVE_DIR
from scripts.audit_2026_08_b26_round2_fix import FIXED_PATTERNS_V2

ORIG_PATTERNS = dict(dataset_mod.INTENT_KEYWORDS_FR)
CACHE_DIR = os.path.join(SAVE_DIR, "cache")


def run_with_corrected_labels(module_name: str, out_filename: str) -> None:
    out_path = os.path.join(CACHE_DIR, out_filename)
    backup_path = out_path + ".orig_bug_backup.json"
    v2_path = out_path.replace(".json", "_v2_labels_corriges.json")

    if os.path.exists(out_path) and not os.path.exists(backup_path):
        shutil.copy2(out_path, backup_path)
        print(f"[propagate] Sauvegarde de l'original (labels buggés) : {backup_path}")
    elif os.path.exists(backup_path):
        print(f"[propagate] Sauvegarde déjà présente : {backup_path} (pas re-écrasée)")

    print(f"[propagate] === {module_name} avec labels V2 corrigés ===")
    dataset_mod.INTENT_KEYWORDS_FR = FIXED_PATTERNS_V2
    try:
        import importlib
        mod = importlib.import_module(module_name)
        mod.main()
    finally:
        dataset_mod.INTENT_KEYWORDS_FR = ORIG_PATTERNS

    if os.path.exists(out_path):
        shutil.move(out_path, v2_path)
        print(f"[propagate] Résultat V2 déplacé vers : {v2_path}")
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, out_path)
        print(f"[propagate] Original restauré à : {out_path}")


def main() -> None:
    assert set(FIXED_PATTERNS_V2) == set(ORIG_PATTERNS)
    run_with_corrected_labels("scripts.explanation_fidelity_test", "explanation_fidelity_results.json")
    run_with_corrected_labels("scripts.steering_fidelity_test", "steering_fidelity_results.json")
    run_with_corrected_labels("scripts.latent_retrieval_precision_eval", "latent_retrieval_precision_results.json")
    print("\n[propagate] Terminé. Comparer *_v2_labels_corriges.json aux *.json "
          "(labels originaux buggés, restaurés) et *.orig_bug_backup.json (identique, "
          "conservé par sécurité).")


if __name__ == "__main__":
    main()
