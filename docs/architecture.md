# Architecture

Vue d'ensemble technique du dépôt. Pour l'historique des décisions et des bugs
corrigés, voir `Context.md` ; pour le détail des expériences et résultats, voir
`RESULTS_TESTS.md` et `docs/experiments.md`.

## Objectif du projet

Analyse interprétable de mails clients EDF via Sparse Autoencoders (SAE) : indexation,
recherche par concepts, clustering interprétable, détection d'urgence/d'intention,
comparaison de corpus, visualisation des concepts activés, explication des décisions
(cf. `Context.md`, section "Projet").

## Deux pipelines

### Pipeline 1 — Gemma-3 + GemmaScope-2 (token-level)

```
mail/texte → Gemma-3-12B-it (hidden states, couche LAYER=24)
           → SAE GemmaScope-2 préentraîné (16 384 features, "core")
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

Deux rôles bien séparés depuis la session v10 (cf. `RESULTS_TESTS.md` §12 pour le
diagnostic qui a motivé cette séparation) :

- **Corpus principal** (`build_email_train_test_corpus`) : mails réels
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
Voir `README.md` pour la table complète des variables.

## Orchestration cluster (SLURM)

Le cluster de calcul (3 partitions GPU : `a100`, `h100`, `h100-bis`, 8 GPU/nœud
chacune) n'a pas d'accès réseau direct sur les nœuds de calcul (`HF_HUB_OFFLINE=1`
systématique dans les scripts `.slurm`, `.venv/bin/python` plutôt que `uv run` qui
tenterait de re-résoudre l'environnement). Scripts principaux :

- `run_sae.slurm` : smoketest volumes réduits (référence stable, ne pas modifier).
- `run_sae_full.slurm` : run à l'échelle complète, corpus historique (pré-v10).
- `run_sae_v10_emails.slurm` / `run_sae_v10_ablation_tok{100k,2M}.slurm` : runs de
  validation du corpus emails-dominant (cf. `docs/experiments.md`).
- `run_augmentation(_full).slurm`, `run_baseline(_full).slurm` : génération du corpus
  augmenté et baseline SAE natif originaux-vs-augmentés.
