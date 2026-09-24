# Scripts archivés

Scripts qui ne sont plus appelés par aucune recette Slurm, test ou document. Ils restent
utilisables depuis la racine : `PYTHONPATH=. .venv/bin/python scripts/archive/<script>.py`.

| Script | Rôle |
|---|---|
| `augmentation_rejection_length_bias_test.py` | biais de longueur dans le rejet des variantes augmentées (`RESULTS_TESTS.md`) |
| `seed_label_overlap_r0_test.py` | recouvrement des labels entre graines (`RESULTS_TESTS.md` §120) |
| `plot_d1_intent_bars.py`, `plot_delta_fve_bars.py`, `plot_model_scale_curve.py` | figures du rapport de stage ; écrivent dans `report/figures/`, dossier qui n'est plus dans le dépôt |
