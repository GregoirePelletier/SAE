# Logs SLURM

Sortie brute (`*.log`) de chaque job SLURM, dans le sous-dossier correspondant au
script `.slurm` qui l'a produite (cf. [`../slurm/README.md`](../slurm/README.md)) :
`pipeline_runs/`, `baseline_diffing/`, `augmentation/`, `analysis/`, `validation/`.

Ce dossier est **gitignoré** (`*.log`) — il n'existe que sur cette machine et n'est
jamais commité. Pour l'historique documenté des runs (paramètres, durée, résultats
et déductions), voir `RESULTS_TESTS.md` à la racine du dépôt plutôt que les logs
bruts eux-mêmes.
