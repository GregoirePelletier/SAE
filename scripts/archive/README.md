# Scripts archivés (audits ponctuels)

Conclusion figée, aucune recette Slurm, test ni document actif ne les appelle.
Lancement inchangé depuis la racine : `PYTHONPATH=. .venv/bin/python scripts/archive/<script>.py`.
Résultats et méthode : sections correspondantes de `RESULTS_TESTS.md`.

| Script | Objet | Référence |
|---|---|---|
| `augmentation_rejection_length_bias_test.py` | biais de longueur du rejet d'augmentation (B.9) | `RESULTS_TESTS.md` |
| `seed_label_overlap_r0_test.py` | recouvrement de labels inter-graines sous R0 | `RESULTS_TESTS.md` §120 |

Conservés à leur place malgré l'absence de référence, par prudence :
`dictionary_width_quality_audit.py` et `compare_to_frozen_benchmark.py`
(chemins calculés depuis `__file__`), `plot_*.py` (régénèrent des figures du rapport).
