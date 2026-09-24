# Scripts archivés (audits ponctuels)

Conclusion figée, aucune recette Slurm, test ni document actif ne les appelle.
Lancement inchangé depuis la racine : `PYTHONPATH=. .venv/bin/python scripts/archive/<script>.py`.
Résultats et méthode : sections correspondantes de `RESULTS_TESTS.md`.

| Script | Objet | Référence |
|---|---|---|
| `augmentation_rejection_length_bias_test.py` | biais de longueur du rejet d'augmentation (B.9) | `RESULTS_TESTS.md` |
| `seed_label_overlap_r0_test.py` | recouvrement de labels inter-graines sous R0 | `RESULTS_TESTS.md` §120 |

| `plot_d1_intent_bars.py`, `plot_delta_fve_bars.py`, `plot_model_scale_curve.py` | figures du rapport de stage (écrivent sous `report/figures/`, retiré du dépôt) | rapport rendu hors dépôt |

Conservés à leur place malgré l'absence de référence, par prudence :
`dictionary_width_quality_audit.py` et `compare_to_frozen_benchmark.py`
(chemins calculés depuis `__file__`).
