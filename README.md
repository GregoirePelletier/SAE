# SAE — analyse interprétable d'emails clients par Sparse Autoencoders

Prototype de recherche (stage M2, EDF R&D) : **découvrir et examiner des thèmes dans des emails
clients** sans taxonomie définie à l'avance, en décomposant les représentations internes d'un
modèle de langue (Gemma-3) en features interprétables (Sparse Autoencoders), puis en s'en servant
pour chercher, comparer et regrouper des emails.

> **Statut : résultats antérieurs au correctif de filiation des emails parents. La séparation FIT/DEV/CONFIRM n'était pas effective pour les variantes. Ces résultats et checkpoints restent consultables comme historique, mais ne constituent pas une validation hors apprentissage. Un rejeu avec le manifeste corrigé est nécessaire. Les évaluations humaines n'ont pas été réalisées.**
>
> Un rejeu est engagé (résultats partiels) : état précis et limites dans
> [`docs/RESULTS_STATUS.md`](docs/RESULTS_STATUS.md).

**Données** : corpus d'emails de type client EDF **synthétiques** (aucune donnée client réelle),
3 474 mails d'origine + variantes générées (émotion, urgence, registre, orthographe). Ni les données,
ni les modèles, ni les résultats ne sont dans ce dépôt (voir « Hors dépôt »).

**Parcours de lecture** : ce README → [`docs/HANDOVER.md`](docs/HANDOVER.md) (carte du dépôt,
démarrage, reprise) → [`docs/RESULTS_STATUS.md`](docs/RESULTS_STATUS.md) (où en est chaque résultat).

---

## Ce que fait le code

- **Pipeline 1 (principal)** : Gemma-3 gelé → activations du residual stream (couche `LAYER`) →
  SAE GemmaScope-2 préentraîné gelé (**CORE**) + petite extension entraînée sur le résidu
  (**EXTRA** ; `SAEBoostResidualSAE`, `src/sae/frozen_core.py`) → représentation **FULL** =
  CORE + EXTRA, agrégée par email. Les features EXTRA sont nommées par un juge LLM local
  (Qwen3.8-27B, jamais le modèle d'extraction).
- **Pipeline 2 (alternative)** : embeddings de phrases F2LLM-v2 → `PhraseLevelSAE` entraîné de
  zéro. Non rejoué dans la campagne post-soutenance.
- **Campagne post-soutenance E00–E09** (1B, couche 13) : comparaison de représentations
  (CORE/FULL/dense bge-m3/TF-IDF), registre de features, recherche d'emails par propriété,
  diffing de populations, stabilité des features, corrélations, clustering ciblé, mini-pilote
  Streamlit. Split parent-aware **FIT / DEV / CONFIRM** (`configs/post_stage/`).

Taux d'interprétabilité de référence (historique, 12B) : **65,7 % (197/300)**,
`RESULTS_TESTS.md` §119 — protocole et réserves dans `docs/RESULTS_STATUS.md`.

---

## Installation

Prérequis : Python **3.12**, [`uv`](https://docs.astral.sh/uv/), accès aux dépôts HuggingFace
« gated » de Gemma (token), un cluster SLURM avec GPU pour tout calcul (voir `docs/ops.md`).

```bash
git clone <url-du-dépôt> SAE && cd SAE
git submodule update --init          # external/interp_embed, external/sae-lens (référence)
uv sync --locked --python 3.12       # crée .venv/ aux versions de uv.lock
cp .env.example .env                 # leviers de configuration documentés
```

Modèles attendus sous `models/` (`MODEL_ID`, `EMB_MODEL`, `JUDGE_MODEL_ID` surchargeables) :
`gemma-3-1b-it` / `gemma-3-12b-it`, `Qwen3.8-27B`, `bge-m3`, `F2LLM-v2-*`. SAE GemmaScope-2 :
`MODEL_SIZE=1b .venv/bin/python download_sae.py` (ou `12b`, `4b`, `270m`). Accès réseau/proxy,
HuggingFace et dépannage : `docs/ops.md`.

## Utiliser

**Consulter les résultats (sans GPU ni poids)** — dashboard Streamlit en lecture seule sur un
dossier `results_*` copié à la racine du clone :

```bash
export SAE_ROOT="$PWD"
export SAE_DASHBOARD_STATE_DIR="$HOME/.local/share/sae_dashboard"
mkdir -p "$SAE_DASHBOARD_STATE_DIR" && chmod 700 "$SAE_DASHBOARD_STATE_DIR"
.venv/bin/python -m streamlit run src/visualization/dashboard.py \
  --server.address=127.0.0.1 --server.port=8501 --server.headless=true \
  --browser.gatherUsageStats=false
```

Fichiers à copier par page et limites : `docs/HANDOVER.md`, parcours A.

**Recalculer** — uniquement par `sbatch`, jamais sur le nœud frontal, depuis la racine du clone :

```bash
export SAE_ROOT="$PWD"; mkdir -p logs/post_stage
sbatch --export=ALL,RUN_SUFFIX=_nouveau slurm/post_stage/01_e01_fit_reference.slurm
```

Ordre des recettes, ressources, préconditions (encodage CONFIRM, juge sur H100) :
`docs/HANDOVER.md` parcours B et `slurm/README.md`. Configuration : `src/config.py` (toutes les
valeurs sont surchargeables par variable d'environnement, voir `.env.example`).

**Tests** (CPU, ~3 min, aucune donnée réelle nécessaire) :

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/check_docs.py     # contrôle éditorial de la documentation
```

---

## Carte du dépôt

| Chemin | Contenu |
|---|---|
| `src/sae/` | pipeline (`saev5.py`), SAE et extension, juge, cache d'extraction |
| `src/post_stage/` | contrat de données FIT/DEV/CONFIRM, représentations, stabilité, profilage |
| `src/analysis/`, `src/data/`, `src/storage/` | métriques/statistiques, corpus et augmentation, checkpoints |
| `src/visualization/dashboard.py` | dashboard Streamlit |
| `scripts/post_stage/` | expériences E01–E07 (scripts autonomes) |
| `scripts/` | audits et ablations historiques (`scripts/archive/` : audits sans usage actif) |
| `slurm/post_stage/` | recettes actives ; autres sous-dossiers = campagnes historiques |
| `configs/post_stage/` | politique de campagne, manifeste et split gelés |
| `tests/` | tests unitaires CPU |
| `docs/` | documentation (voir `docs/HANDOVER.md`, section « Documents ») |
| `report/` | rapport de stage (voir `report/README.md`) |
| `RESULTS_TESTS.md` | journal d'expériences historique, sections §N citées par le rapport |
| `external/` | sous-modules de référence (interp_embed, SAELens) |

## Hors dépôt

Données (`local_data/`), modèles (`models/`), jeux publics (`datasets/`), résultats
(`results_*/`), caches d'extraction (`local_data/activation_cache/`, plusieurs centaines de Go) et
logs sont ignorés par Git. Ils sont sur le cluster d'origine ; leur transfert, sous contrôle
d'accès, et l'inventaire des artefacts nécessaires sont décrits dans `docs/HANDOVER.md`.

Auteur : Grégoire Pelletier (stage M2 Mathématiques et IA, Université Paris-Saclay, EDF R&D).
