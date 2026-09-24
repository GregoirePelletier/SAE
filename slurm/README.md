# Recettes Slurm

Scripts de soumission `sbatch`, classés par catégorie. Chacun écrit ses logs dans le
sous-dossier de `logs/` correspondant (voir `docs/ops.md`).

- [`post_stage/`](post_stage/) : campagne post-soutenance E00 à E09 (recettes maintenues).
- [`pipeline_runs/`](pipeline_runs/) : runs du pipeline (`src/sae/saev5.py`) d'avant la
  soutenance, dont les ablations (volume de tokens, largeur du SAE, époques, capacité de
  l'extension, couche, taille du modèle).
- [`baseline_diffing/`](baseline_diffing/) : diffing avec le SAE GemmaScope seul, emails
  d'origine contre variantes.
- [`augmentation/`](augmentation/) : génération des variantes augmentées et fusion des morceaux.
- [`analysis/`](analysis/) : analyses qui réutilisent des activations déjà calculées (robustesse
  du juge, qualité des explications, sondes, corrélations, comparaison d'embeddings).
- [`validation/`](validation/) : vérifications ponctuelles.

Suivi d'un job : `squeue -u $USER`, puis `sacct -j <id>` une fois terminé. Les résultats de
chaque run sont consignés dans `RESULTS_TESTS.md` (avant la soutenance) ou
`docs/post_stage/` (après).

## Recettes actives et lancement

Seules les recettes de `post_stage/` sont maintenues ; les autres dossiers correspondent aux
campagnes d'avant la soutenance et ne sont pas à relancer par défaut (leurs ressources n'ont pas
été revérifiées). L'ordre des recettes `post_stage/` et leurs dépendances sont décrits dans
`docs/HANDOVER.md`.

Lancer depuis la racine du clone, après `export SAE_ROOT="$PWD"` et `mkdir -p logs/post_stage`
(ou le sous-dossier de logs de la catégorie utilisée). Les chemins `#SBATCH --output` sont
relatifs à ce répertoire, et les directives `#SBATCH` n'interprètent pas les variables shell.
Chaque recette exporte ses propres variables : une valeur exportée avant `sbatch` peut être
écrasée par le script.

- `RUN_SUFFIX` : suffixe ajouté aux dossiers de résultats des recettes `post_stage/`, pour écrire
  un nouveau run sans écraser les précédents.
- `EVAL_SUFFIX` : dossier distinct pour l'encodage de CONFIRM et les expériences E03, E04, E06 et
  E07 (voir `docs/HANDOVER.md`).
- Le modèle juge Qwen3.8-27B demande une H100 (`h100` ou `h100-bis`). La répartition sur 2 A100
  n'existe que dans les recettes `06b` et `07b`.
- `bash -n <recette>` vérifie seulement la syntaxe du script, pas les ressources ni les chemins.
