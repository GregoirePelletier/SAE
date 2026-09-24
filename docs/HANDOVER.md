# Passation

Ordre de lecture : `README.md`, ce document, puis `docs/RESULTS_STATUS.md` (état des résultats et
travail restant).

## Organisation du dépôt

**Code actif**

- `src/sae/saev5.py` et `src/config.py` : le pipeline (extraction, entraînement de l'extension,
  encodage, étiquetage des features). Toute valeur de `config.py` se surcharge par variable
  d'environnement.
- `src/post_stage/`, `scripts/post_stage/`, `slurm/post_stage/`, `configs/post_stage/` : la
  campagne post-soutenance E00 à E09 (contrat de données FIT / DEV / CONFIRM, une expérience par
  script).
- `src/visualization/dashboard.py` : le dashboard Streamlit.
- `tests/` : tests unitaires CPU.

**Historique**

- `scripts/` (hors `post_stage/`) et les autres dossiers de `slurm/` : audits et ablations d'avant
  la soutenance. Chacun correspond à une section de `RESULTS_TESTS.md`. Plusieurs restent utiles
  (`retrieval_demo.py`, `generate_diagnostic_plots.py`, `test_chargement_sae.py`) ; avant de
  retirer un script, chercher son nom dans `slurm/`, `docs/`, `tests/` et `RESULTS_TESTS.md`.
- `RESULTS_TESTS.md` : journal numéroté (§N) des expériences d'avant la soutenance. Les noms de
  dossiers `results_v*` qu'il cite désignent des runs qui ne sont pas dans le dépôt (la plupart
  ont été supprimés du disque ; voir `docs/archive/archived_runs_manifest.md`).
- `scripts/archive/`, `docs/archive/` : scripts et documents conservés pour mémoire, non maintenus
  (index dans chaque dossier).

**Documentation de référence** : `docs/architecture.md` (architecture et scripts),
`docs/ops.md` (cluster, réseau, environnement), `docs/references.md` (bibliographie et écarts aux
papiers), `docs/INTERP_EMBED_COVERAGE.md` (correspondance avec le code interp_embed),
`docs/post_stage/PLAN_E00-E09.md` (plan de la campagne, cité « plan §N » dans le code et les
comptes rendus).

**Outils pour Claude Code** : `CLAUDE.md` (règles du projet) et `.claude/settings.json` (relance
des tests après chaque modification d'un fichier Python). Sans effet si l'on n'utilise pas cet outil.

## Points d'entrée

- Une expérience post-soutenance : `python scripts/post_stage/eNN_*.py`, lancé par la recette
  correspondante de `slurm/post_stage/`. Il n'y a pas de CLI commune : `src/post_stage/cli.py` ne
  contient que `freeze-corpus`, qui a servi à geler le split.
- Le pipeline complet : `src/sae/saev5.py`, via une recette Slurm.
- Le dashboard et les tests : voir le README.

## Consulter les résultats sans GPU

Il faut Python 3.12, l'environnement installé (`uv sync --locked --python 3.12`) et une copie du
dossier de résultats voulu à la racine du clone. Commande de lancement : voir le README.

Runs disponibles sur le cluster d'origine :

| Dossier | Contenu |
|---|---|
| `results_post_stage_e01_fit_1b_layer13_k5_v2/` | rejeu avec le split corrigé : E01, E02, E05 |
| `results_post_stage_e01_fit_1b_layer13_k5_v2_eval/` | rejeu : encodage de CONFIRM, puis E03, E04, E06, E07 (en cours) |
| `results_post_stage_e01_fit_1b_layer13_k5/` | première exécution de E01 à E07 (split erroné, historique) |
| `results_post_stage_e05_{seed43,seed44,randinit45,randinit46}_v2/` | entraînements de E05 |

Ce que lit chaque page :

- la plupart des pages lisent les JSON et fichiers `umap_*_coords.parquet` à la racine du run ;
  les poids (`*.pt`) et le dossier `cache/` ne sont pas nécessaires, sauf pour la page Features,
  qui lit `cache/p1_judge_labels_extended.json` et son `.meta.json` ;
- la comparaison mail original / augmenté a besoin de `local_data/emails/Mails.tsv` et
  `augmented_mails.jsonl` ;
- la recherche par propriété et le diffing post-soutenance sont dans la page « Mini-pilote
  analyste (E08) » ; ils affichent des résultats calculés à l'avance. La page « Recherche » ne
  fait pas de recherche sémantique en direct.

Le sélecteur de run liste tous les dossiers `results_*` du clone ; un bandeau signale les runs
de la première exécution. Les pistes saisies dans le formulaire E08 sont enregistrées dans
`$SAE_DASHBOARD_STATE_DIR` (par défaut `local_data/dashboard_state/`), hors Git ; deux personnes
qui écrivent en même temps peuvent s'écraser. Pour un accès distant, ouvrir un tunnel SSH vers la
machine qui exécute Streamlit.

## Recalculer

Tout calcul passe par `sbatch`, lancé depuis la racine du clone (voir aussi `slurm/README.md`) :

```bash
export SAE_ROOT="$PWD"
mkdir -p logs/post_stage
sbatch --export=ALL,RUN_SUFFIX=_v3 slurm/post_stage/01_e01_fit_reference.slurm
```

Ordre des recettes de `slurm/post_stage/` (utiliser `--dependency=afterok:<id>`) :

1. `01` : entraînement de référence sur FIT.
2. Après `01` : `02`, `03` (sondes E01), `04` (registre E02), `10`, `11`, `13`, `14`
   (entraînements E05).
3. Après `02`, `03`, `04` : `05` (encodage de CONFIRM).
4. Après `05` : `06` (E03), `07` (E04), `08` (E06), `09` (E07).
5. Après `10` et `11` : `12` ; après `12`, `13` et `14` : `15` (analyses E05, CPU).

`00` (profilage mémoire) est facultatif.

Variables à connaître :

- `RUN_SUFFIX` ajoute un suffixe aux dossiers de résultats (`_v2` pour le rejeu actuel). Sans
  suffixe, les recettes écrivent dans les dossiers de la première exécution.
- `EVAL_SUFFIX` (recettes `05` à `09`) fait écrire l'encodage de CONFIRM et les expériences
  suivantes dans un dossier distinct (`_eval` pour le rejeu actuel), où `05` copie les
  checkpoints FIT. C'est nécessaire : l'ajout de CONFIRM change la clé du cache d'extraction, et
  `saev5.py` refuse un dossier dont les liens de cache pointent vers une autre clé. Ne jamais
  supprimer ces liens pour contourner l'erreur.
- `SAE_ROOT` vaut par défaut le répertoire depuis lequel `sbatch` est lancé.

GPU : le juge Qwen3.8-27B tient sur une H100 mais pas sur une A100 de 40 Go. Les recettes
`01`, `04` à `11`, `13` et `14` le chargent et doivent tourner sur `h100` ou `h100-bis` ; seules
`02` et `03` (bge-m3) peuvent aller sur `a100`. Les scripts E02, E03 et E04 acceptent
`--judge-device auto` pour répartir le juge sur 2 A100 (recettes `06b` et `07b`). Budget et
limites : `configs/post_stage/campaign_policy.yaml`.

## Transférer les artefacts

Données (`local_data/`), modèles (`models/`), résultats (`results_*`) et logs ne sont pas dans
Git. Pour transmettre un run :

- pour consulter, les fichiers listés plus haut suffisent (quelques dizaines de Mo) ;
- pour recalculer, il faut aussi les modèles, `local_data/emails/` et le cache d'extraction
  `local_data/activation_cache/<clé>/` (plusieurs dizaines à centaines de Go par clé). Les fichiers
  `cache/p1_all_doc_acts.pt`, `p1_raw_residuals.memmap` et `p1_token_fragments` d'un run sont des
  liens symboliques absolus vers ce cache : une copie du run sans le cache donne des liens cassés.

Vérifier les copies avec `sha256sum` sur les petits fichiers ; calculer les empreintes des gros
fichiers sur un nœud de calcul, pas sur le frontal.

## Pièges connus

- Sans `logits_to_keep=1`, `output_hidden_states=True` calcule les logits sur tout le vocabulaire
  et peut saturer la mémoire.
- `@torch.no_grad()` sur un générateur ne protège pas ses itérations : mettre un
  `with torch.no_grad():` dans la boucle.
- Passer les activations SAE à `LogisticRegression` en matrice creuse (CSR), jamais en dense.
- `SEED` n'entre pas dans la clé du cache d'extraction ; une reprise après interruption n'est pas
  reproductible au bit près.
- Pour les features mortes, lire `dead_pct_extension`, pas `dead_pct` (qui mélange CORE et EXTRA).
- Téléchargements HuggingFace via le proxy du cluster : `HF_HUB_DISABLE_XET=1`.

Détail de ces points et diagnostics d'un run : `docs/archive/CLAUDE_long_2026-09.md`.
