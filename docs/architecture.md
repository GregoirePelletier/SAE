# Architecture

Vue d'ensemble technique du dépôt. Pour le détail des expériences et
résultats, voir `RESULTS_TESTS.md` et `docs/experiments.md`.

## Objectif du projet

Analyse interprétable de mails clients EDF via Sparse Autoencoders (SAE) : indexation,
recherche par concepts, clustering interprétable, détection d'urgence/d'intention,
comparaison de corpus, visualisation des concepts activés, explication des décisions.

## Deux pipelines

### Pipeline 1 — Gemma-3 + GemmaScope-2 (token-level)

```
mail/texte → Gemma-3-12B-it (hidden states, couche LAYER=24)
           → SAE GemmaScope-2 préentraîné ("core", largeur configurable --
             16 384 features pour le run principal et la plupart des
             ablations comparatives, 65 536 par défaut dans `src/config.py`
             depuis la vérification de couverture Neuronpedia, 262 144 testé
             une fois -- aucune des trois largeurs ne change
             significativement le taux d'interprétabilité, cf.
             `RESULTS_TESTS.md` §17/§29)
           → [optionnel] FrozenCoreResidualSAE/ExtendedSAE (résidu core → 1024
             features "extra", TopK+AuxK)
           → max-pool documentaire (max sur les tokens du document)
```

- Le SAE **core** est préentraîné par DeepMind (GemmaScope-2), gelé, jamais réentraîné.
  Ses features sont labellisées via le cache Neuronpedia local (`local_data/
  neuronpedia_labels/`, cf. `src/sae/neuronpedia_labels.py`).
- L'**extension** (`FrozenCoreResidualSAE`/`ExtendedSAE`, `src/sae/frozen_core.py`)
  encode le résidu (ce que le SAE core ne reconstruit pas) avec un second SAE de plus
  petite taille (`D_EXTRA=1024`, `K_EXTRA=32` actifs), entraîné **from-scratch** sur le
  corpus du projet (cf. section Corpus ci-dessous). Design spécifique au projet — jamais
  fourni par GemmaScope/SAELens.
- Les features d'extension n'existent sur aucune base externe (Neuronpedia ne les
  connaît pas) : elles sont labellisées par un **juge LLM local** (odd-one-out,
  `src/sae/judge.py::odd_one_out_judge`) — cf. `docs/experiments.md` pour le protocole
  et son taux de succès mesuré.

### Pipeline 2 — F2LLM + PhraseLevelSAE (phrase-level)

```
mail/texte → découpage en phrases
           → F2LLM-v2-80M (embeddings de phrase, dim Matryoshka=320)
           → PhraseLevelSAE entraîné from-scratch (BatchTopK+AuxK, D_SAE=8192, K=16)
           → max-pool documentaire (max sur les phrases du document)
```

Entraînement complet (pas de partie préentraînée) : `src/sae/phrase_sae.py`. Même
protocole de labellisation par juge LLM local (`local_gemma_judge`, `src/sae/judge.py`).

## Corpus (`src/data/preparation.py`)

Deux rôles bien séparés (cf. `RESULTS_TESTS.md` §12 pour le
diagnostic qui a motivé cette séparation) :

- **Corpus principal** (`build_email_train_test_corpus`) : mails originaux
  (`local_data/emails/Mails.tsv`) + variantes augmentées acceptées
  (`local_data/emails/augmented_mails.jsonl`, générées par `scripts/run_augmentation.py`
  via perturbations contrôlées — émotion, registre, orthographe, urgence, cf.
  `src/data/augmentation.py::AXES`). C'est le corpus qui **entraîne** le SAE
  d'extension (réservoir de résidus) et le `PhraseLevelSAE`. Split **group-aware** par
  mail d'origine (`parent_id`) : un mail et toutes ses variantes tombent du même côté
  train/test, pour éviter toute fuite de quasi-duplicata.
- **Corpus secondaire** (`prepare_domain_dataset`, energy/sports/support depuis
  FineWeb-2/Wikipedia FR) : encodé **post-hoc** par le SAE déjà entraîné, jamais utilisé
  pour l'entraînement. Sert uniquement à la démonstration préexistante de diffing
  cross-domaine (`corpus_diff_stats` energy vs sports).

## Stockage des activations (`src/storage/`)

- `fragment_store.py` : CSR fait-maison en tenseurs torch pour les activations
  token-level (un mail encodé en 16k+1024 dimensions par token serait ~400 Mo dense/doc
  à la largeur 262k historique — le format CSR ramène ça à quelques centaines de Ko).
- `shards.py` : sharding/mmap pour les gros tenseurs d'activations denses (embeddings
  de phrase P2, activations doc-level).

## Précision numérique

`DTYPE=bf16` partout par défaut, y compris en local — **pas** fp16. Gemma-3 a des
activations "massives" documentées dans le residual stream (outliers ~1e5) qui
dépassent le max représentable en fp16 (~65504), overflow silencieux vers inf/nan qui
contamine tout l'entraînement de l'extension (`Loss=nan` dès l'epoch 1, observé avant
correction). La branche "extra" de `ExtendedSAE` reste volontairement en fp32.

## Configuration (`src/config.py`)

Source unique de vérité, tout surchargeable par variable d'environnement (compat
`sbatch`). Presets modèle/SAE par taille (`MODEL_SIZE` ∈ {12b, 4b, 1b, 270m}), chemins
de données canoniques (`LOCAL_MAILS_PATH`, `LOCAL_AUGMENTED_MAILS_PATH`,
`NEURONPEDIA_LABELS_PATH` — tous trois partagés entre runs, indépendants de `SAVE_DIR`).
Table complète des variables : `README.md`.

Orchestration cluster (partitions, soumission, logs) : `docs/ops.md`.

## Arborescence

```
src/
  config.py              # Source unique de vérité (constantes, presets modèle/SAE)
  sae/
    saev5.py              # Orchestration Pipeline 1 + Pipeline 2 (point d'entrée)
    sae_shared.py          # Harnais d'entraînement ExtendedSAE, ré-exports
    gemma_scope_loader.py  # Chargement SAE GemmaScope-2 (disque local ou Hub)
    neuronpedia_labels.py  # Labels de features via bucket S3 Neuronpedia
    frozen_core.py         # FrozenCoreResidualSAE / ExtendedSAE
    phrase_sae.py          # PhraseLevelSAE + embeddings F2LLM (Pipeline 2)
    batch.py               # BatchTopKEncoder (seuil θ calibré)
    judge.py               # Labellisation LLM locale (odd-one-out)
    retrieval/latent_terms.py  # BM25 sur features SAE (Latent Terms)
    compare/                   # Comparaison cross-modèle d'embeddings
  analysis/
    activations.py         # Extraction/masquage activations, max-pooling
    cooccurrence.py         # NPMI, diffing de corpus (Fisher+BH), clustering
    metrics.py              # FVE, NMSE, L0, classification en aval
    stats.py                 # Tests statistiques partagés
    plotting.py               # Figures de diagnostic réutilisables (Plotly)
    visualization.py           # Exports Plotly HTML autonomes
  data/
    dataset.py               # Ingestion Mails.tsv, GoEmotions
    preparation.py           # build_email_train_test_corpus (corpus principal,
                              #   emails+augmentés, group-aware) + construction du
                              #   corpus secondaire domaine (FineWeb-2/Wikipedia)
    augmentation.py          # Génération de variantes perturbées
    keywords.py               # Listes de mots-clés par domaine
  storage/
    fragment_store.py       # Stockage CSR (torch) des activations token-level
    shards.py                # Sharding/mmap d'activations denses
  visualization/
    dashboard.py             # Dashboard interactif Streamlit
scripts/                   # Points d'entrée secondaires (cf. section Scripts)
tests/                     # Suite pytest
external/sae-lens/         # Submodule SAELens (comparaison d'implémentation)
local_data/
  emails/                  # Corpus EDF : Mails.tsv + augmented_mails.jsonl
  neuronpedia_labels/      # Cache labels Neuronpedia, partagé par tous les runs
  saes/                    # Poids SAE téléchargés (download_sae.py)
docs/                      # Référence technique (ce dossier)
report/                    # Rapport de stage M2
slurm/, logs/              # Soumission SLURM et sorties (cf. docs/ops.md)
results_v*/                # Répertoires de résultats par run (gitignorés),
                            #   cf. RESULTS_TESTS.md pour l'index
```

## Scripts

### `download_sae.py` (racine)

Télécharge le modèle Gemma-3 et le SAE GemmaScope-2 correspondant à `MODEL_SIZE` vers le
cache HF (`HF_HOME`) et `LOCAL_SAE_DIR` respectivement.

```bash
MODEL_SIZE=12b python download_sae.py            # modèle + SAE
python download_sae.py --model-only               # modèle seul
python download_sae.py --sae-only                  # SAE seul
```

### `src/sae/saev5.py` — orchestration principale

Point d'entrée des deux pipelines. `if __name__ == "__main__":` charge le
corpus principal (mails originaux + variantes augmentées, via
`build_email_train_test_corpus`, avec repli synthétique si `LOCAL_MAILS_PATH`
est absent), puis exécute Pipeline 1 et/ou Pipeline 2 selon `PIPELINES` (`p1`,
`p2`, ou `p1,p2`).

```bash
PYTHONPATH=. PIPELINES=p1,p2 python src/sae/saev5.py
```

Doit être lancé depuis la racine du dépôt avec `PYTHONPATH=.` (le script
mélange imports absolus, `from src.analysis...`, et imports relatifs "à
plat", `from sae_shared import ...`, résolus car son propre dossier est ajouté
à `sys.path`).

Variables utiles : `EMAIL_TEST_SPLIT`, `MAX_AUGMENTED_PER_MAIL`,
`N_TOTAL_ENERGY`/`N_TOTAL_SPORTS`/`N_TOTAL_SUPPORT` (taille du corpus
secondaire de diffing), `N_TOKENS_EXTRA_TRAIN` (budget d'entraînement de
l'extension), `N_FEATURES_TO_LABEL` (nombre de features jugées par le LLM
local — la puissance statistique de tout taux d'interprétabilité en dépend
directement).

Sorties dans `SAVE_DIR` : `p1_top_core_features.json` /
`p1_top_extended_features.json` (features labellisées), `umap_pipeline1_*.html`
/ `umap_pipeline2_*.html` (visualisations interactives — suffixe `_emails`
pour le corpus principal, `_diffcorpus` pour le corpus secondaire),
`p1_diff_energy_sports.csv` (diffing statistique), `cache/` (activations,
checkpoints SAE, labels Neuronpedia, réutilisés entre runs).

### `scripts/baseline_gemmascope.py`

Compare mails originaux vs augmentés (perturbations contrôlées) avec le SAE
GemmaScope natif, sans extension : quelles features de base changent
significativement de fréquence d'activation entre les deux groupes ?

```bash
python scripts/baseline_gemmascope.py local_data/emails/Mails.tsv local_data/emails/augmented_mails.jsonl
```

Nécessite `augmented_mails.jsonl` (produit par `run_augmentation.py`, lui-même
nécessite un `Mails.tsv`).

### `scripts/relabel_diff_csvs.py`

Réapplique offline (CPU, aucune réextraction) les labels Neuronpedia du cache
canonique aux `diff_*.csv`/`.html` déjà produits par `baseline_gemmascope.py`
— utile quand un run a tourné hors-ligne avant que le cache de labels ne soit
disponible localement.

### `scripts/run_augmentation.py`

Génère des variantes de mails perturbées (émotion, registre, orthographe,
urgence — grille `AXES` de `src/data/augmentation.py`) via Gemma-3, avec
garde-fous factuels (numéros de contrat, montants, dates préservés).
Nécessite un `Mails.tsv`.

```bash
LOCAL_MAILS_PATH=local_data/emails/Mails.tsv MODEL_ID=google/gemma-3-12b-it \
  SAVE_DIR=local_data/emails/ python scripts/run_augmentation.py
```

### `scripts/retrieval_demo.py`

Câble `src/sae/retrieval/latent_terms.py` (BM25 sur le vocabulaire latent d'un
SAE de phrases, Clavié et al. 2026) à un point d'entrée testable sans
`Mails.tsv` : corpus de substitution public (FineWeb-2, domaine "energy").

```bash
python scripts/retrieval_demo.py --n-docs 60 --query "énergie électrique renouvelable"
```

Sur le corpus original : `src/sae/retrieval/latent_terms.py --mails <Mails.tsv> --query "..."`.

### `src/sae/compare/pipeline.py`

Compare deux modèles d'embeddings de phrases via leurs SAE respectifs :
alignement cross-modèle des features (`model_compare.py`), détection de
features non alignées, clustering/diffing pour un seul modèle (`--mode
analysis`).

```bash
python -m src.sae.compare.pipeline --mails <Mails.tsv> --mode analysis
python -m src.sae.compare.pipeline --mails <Mails.tsv> --mode compare \
  --model-a codefuse-ai/F2LLM-v2-80M --model-b intfloat/multilingual-e5-small
```

### Diagnostics manuels (hors pytest)

- `test_chargement_sae.py` : charge le SAE configuré, affiche ses dimensions.
- `scripts/test_massive_acts.py` : corrélation Pearson entre pré-activations
  "extra" et norme du token — diagnostic de pollution par activations
  massives. Suppose des checkpoints déjà produits par un run complet.

### `src/visualization/dashboard.py` — dashboard interactif (Streamlit)

Lit uniquement des artefacts déjà produits sur disque (JSON, parquet, CSV) —
aucun modèle chargé, aucun GPU requis. Vue d'ensemble par run, UMAP
interactif, features (core Neuronpedia + extension + phrase-level, exemples
positifs/négatifs), diffing, recherche par mot-clé, diagnostics
d'entraînement, urgence/robustesse du juge. Sélecteur de run dans la barre
latérale (tous les `results_*/` du dépôt).

```bash
.venv/bin/python -m streamlit run src/visualization/dashboard.py
```
