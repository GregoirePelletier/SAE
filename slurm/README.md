# Scripts SLURM

Tous les scripts de soumission `.slurm` du dépôt, classés par catégorie. Chaque
script écrit sa sortie dans le sous-dossier `logs/` de même nom (cf.
[`../docs/ops.md`](../docs/ops.md)) — jamais à la racine du dépôt.

- **[`pipeline_runs/`](pipeline_runs/)** — Pipeline 1 (Gemma-3 + GemmaScope) et
  Pipeline 2 (F2LLM + PhraseLevelSAE), via `src/sae/saev5.py` : run principal, et
  toutes les ablations contrôlées (volume de tokens, largeur du SAE core, nombre
  d'époques, capacité de l'extension).
- **[`baseline_diffing/`](baseline_diffing/)** — `scripts/baseline_gemmascope.py` :
  diffing avec le SAE GemmaScope natif (sans extension), mails originaux vs
  augmentés par axe de perturbation.
- **[`augmentation/`](augmentation/)** — `scripts/run_augmentation.py` : génération
  des variantes de mails augmentées (+ script de fusion des shards).
- **[`analysis/`](analysis/)** — tests post-hoc qui réutilisent des activations déjà
  en cache (aucune réextraction Gemma-3 sauf mention contraire) : robustesse du
  protocole de jugement, fidélité/plausibilité de l'explication, labellisation
  contrastive directe, sondes intention/urgence, corrélations "intéressantes",
  comparaison de backbones d'embedding.
- **[`validation/`](validation/)** — smoke-tests ponctuels.

Usage type :

```bash
sbatch slurm/pipeline_runs/run_sae_v12_scaled.slurm
squeue -u $USER
tail -f logs/pipeline_runs/sae_v12_scaled_<jobid>.log
```

Le suivi chronologique détaillé (paramètres exacts, durée, résultats, déductions)
de chaque run est dans `RESULTS_TESTS.md` à la racine du dépôt — les logs bruts ne
sont qu'une trace d'exécution, pas la documentation de référence.

## Recettes actives et lancement

- **Actives** : `post_stage/` (campagne post-soutenance, numérotée par ordre de dépendance,
  ressources par recette dans son en-tête `#SBATCH`, budget dans `configs/post_stage/campaign_policy.yaml`).
  Les autres répertoires sont des campagnes historiques : ne pas les rejouer par défaut ; leurs
  chemins passent par `SAE_ROOT` mais leurs ressources n'ont pas été re-testées.
- **Lancement** : depuis la racine du clone, `export SAE_ROOT="$PWD"` puis `mkdir -p logs/post_stage`
  (et le sous-dossier de logs de la catégorie utilisée) **avant** `sbatch`. Les `#SBATCH --output`
  sont relatifs à ce répertoire ; les directives `#SBATCH` n'expansent pas les variables shell ; un
  script peut écraser une variable exportée par l'appelant.
- **`RUN_SUFFIX`** (recettes `post_stage/`) : suffixe ajouté aux dossiers de résultats pour un nouveau
  run (`sbatch --export=ALL,RUN_SUFFIX=_x ...`) ; vide = mêmes dossiers que l'historique. Un rejeu
  sous le manifeste corrigé ne recharge pas les anciens poids FIT.
- **Juge** : 1 H100/H100-bis suffit ; le repli 2×A100 n'existe que dans les scripts avec
  `--judge-device auto` (E02-E04, variantes `*_a100.slurm`). L'encodage CONFIRM (`05_*`) change la clé de
  cache : voir `docs/HANDOVER.md`, ne pas supprimer les liens de cache pour contourner l'erreur.
- `bash -n` ne valide que la syntaxe shell (pas droits, ressources, poids, chemins résolus).
