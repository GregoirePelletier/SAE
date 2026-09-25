# SAE : analyse interprétable d'emails clients par Sparse Autoencoders

Prototype de recherche réalisé pendant un stage de M2 à EDF R&D. L'objectif est de découvrir et
d'examiner des thèmes dans des emails de clients sans définir de catégories à l'avance. Pour cela,
on décompose les représentations internes d'un modèle de langue (Gemma-3) en features
interprétables à l'aide de Sparse Autoencoders (SAE), puis on s'en sert pour chercher, comparer et
regrouper des emails.

**Données.** Corpus synthétique d'emails de type client EDF (aucune donnée client réelle) :
3 474 emails d'origine et leurs variantes générées (émotion, urgence, registre, orthographe). Les
données, les modèles et les résultats ne sont pas dans ce dépôt.

**État.** Prototype non validé par des utilisateurs. Les résultats de la campagne
post-soutenance ont été recalculés après la correction d'une erreur dans le découpage des
données. Tout est détaillé dans [`docs/RESULTS_STATUS.md`](docs/RESULTS_STATUS.md).

**Pour démarrer** : lire ce fichier, puis [`docs/HANDOVER.md`](docs/HANDOVER.md) (organisation du
dépôt, consultation, recalcul), puis [`docs/RESULTS_STATUS.md`](docs/RESULTS_STATUS.md).

## Ce que fait le code

- **Pipeline principal.** Gemma-3 (gelé) produit des activations à une couche donnée. Un SAE
  préentraîné de GemmaScope-2 (gelé, appelé CORE) les décompose ; une petite extension entraînée
  sur ce que CORE ne reconstruit pas ajoute des features propres au domaine (EXTRA,
  `src/sae/frozen_core.py`). La représentation FULL réunit les deux et est agrégée par email.
  Les features EXTRA sont nommées par un modèle juge local (Qwen3.8-27B), différent du modèle
  analysé.
- **Pipeline alternatif.** Embeddings de phrases F2LLM-v2 et SAE entraîné de zéro
  (`PhraseLevelSAE`). Il n'a pas été repris après la soutenance.
- **Campagne post-soutenance (E00 à E09, Gemma-3-1B).** Comparaison de représentations (CORE,
  FULL, embeddings denses bge-m3, TF-IDF), catalogue de features, recherche d'emails par
  propriété, comparaison de populations, stabilité des features, associations, regroupement
  ciblé, et un dashboard pour un mini-pilote. Les emails d'origine sont répartis en trois
  ensembles fixes : FIT (entraînement), DEV (réglages) et CONFIRM (évaluation), définis dans
  `configs/post_stage/`.

Résultat de référence du rapport de stage : 65,7 % des features de l'extension jugées
interprétables (197 sur 300, Gemma-3-12B, `RESULTS_TESTS.md` §119).

## Installation

Prérequis : Python 3.12, [`uv`](https://docs.astral.sh/uv/), un token HuggingFace ayant accès aux
modèles Gemma, et un cluster Slurm avec GPU pour tout calcul (voir `docs/ops.md`).

```bash
git clone <url-du-dépôt> SAE && cd SAE
git submodule update --init          # external/interp_embed et external/sae-lens
uv sync --locked --python 3.12       # crée .venv/ avec les versions de uv.lock
cp .env.example .env                 # variables de configuration disponibles
```

Les modèles sont attendus sous `models/` : `gemma-3-1b-it` ou `gemma-3-12b-it`, `Qwen3.8-27B`,
`bge-m3`, `F2LLM-v2-*` (chemins modifiables par `MODEL_ID`, `JUDGE_MODEL_ID`, `EMB_MODEL`). Les
SAE GemmaScope-2 se téléchargent avec `MODEL_SIZE=1b .venv/bin/python download_sae.py` (ou `12b`,
`4b`, `270m`). Proxy, HuggingFace et dépannage : `docs/ops.md`.

## Utilisation

**Consulter les résultats**, sans GPU ni poids de modèle : copier un dossier de résultats
(`results_*`) à la racine du clone, puis lancer le dashboard.

```bash
export SAE_ROOT="$PWD"
export SAE_DASHBOARD_STATE_DIR="$HOME/.local/share/sae_dashboard"
mkdir -p "$SAE_DASHBOARD_STATE_DIR" && chmod 700 "$SAE_DASHBOARD_STATE_DIR"
.venv/bin/python -m streamlit run src/visualization/dashboard.py \
  --server.address=127.0.0.1 --server.port=8501 --server.headless=true \
  --browser.gatherUsageStats=false
```

Les fichiers nécessaires à chaque page sont listés dans `docs/HANDOVER.md`.

**Recalculer** : uniquement par `sbatch` (jamais sur le nœud frontal), depuis la racine du clone.

```bash
export SAE_ROOT="$PWD"
mkdir -p logs/post_stage
sbatch --export=ALL,RUN_SUFFIX=_v3 slurm/post_stage/01_e01_fit_reference.slurm
```

L'ordre des recettes, les ressources GPU et les précautions sont dans `docs/HANDOVER.md` et
`slurm/README.md`. Toutes les valeurs de `src/config.py` se surchargent par variable
d'environnement.

**Tests** (CPU, environ 5 minutes, sans données) :

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/check_docs.py     # vérification de la documentation
```

## Organisation

| Chemin | Contenu |
|---|---|
| `src/sae/` | pipeline (`saev5.py`), SAE et extension, modèle juge, cache d'extraction |
| `src/post_stage/` | découpage FIT / DEV / CONFIRM, représentations, stabilité, profilage mémoire |
| `src/analysis/`, `src/data/`, `src/storage/` | statistiques, corpus et augmentation, checkpoints |
| `src/visualization/dashboard.py` | dashboard Streamlit |
| `scripts/post_stage/` | une expérience par script (E01 à E07) |
| `scripts/` | audits et ablations d'avant la soutenance ; `scripts/archive/` pour ceux qui ne servent plus |
| `slurm/post_stage/` | recettes Slurm de la campagne ; les autres dossiers de `slurm/` sont historiques |
| `configs/post_stage/` | règles de la campagne et découpage gelé des données |
| `tests/` | tests unitaires CPU |
| `docs/` | documentation ; `docs/archive/` pour les documents historiques |
| `RESULTS_TESTS.md` | journal des expériences d'avant la soutenance |
| `external/` | code de référence (interp_embed, SAELens), en sous-modules |

Hors dépôt, sur le cluster d'origine : données (`local_data/`), modèles (`models/`), jeux de
données publics (`datasets/`), résultats (`results_*/`), caches d'extraction (plusieurs centaines
de Go) et logs. Leur transfert est décrit dans `docs/HANDOVER.md`.

Auteur : Grégoire Pelletier, stage de M2 Mathématiques et IA (Université Paris-Saclay), EDF R&D.
