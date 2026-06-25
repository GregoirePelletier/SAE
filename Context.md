# CONTEXT.md

# Projet

Analyse interprétable de mails clients EDF à l'aide de Sparse Autoencoders (SAE).

Objectif :

Construire une plateforme permettant :

- indexation de mails
- recherche par concepts
- clustering interprétable
- détection d'urgence
- détection d'intentions
- comparaison de corpus
- visualisation des concepts activés
- retrieval par propriétés
- explication des décisions

Le projet doit réutiliser au maximum :

1. SAELens
2. GemmaScope
3. Interpretable Embeddings with Sparse Autoencoders
4. SAE Boost (optionnel)

Aucune réimplémentation ne doit être conservée lorsqu'une implémentation robuste existe déjà dans ces dépôts.

---

# Références

SAELens:
https://github.com/jbloomAus/SAELens

GemmaScope:
https://github.com/google-deepmind/gemma-scope

Interpretable Embeddings:
https://github.com/nickjiang2378/interp_embed

Article:
Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit

SAE Boost:
chercher l'implémentation officielle la plus récente.

---

# État actuel

## Fonctionnalités présentes

### Pipeline 1

Gemma-3

Layer residual stream

GemmaScope SAE

FrozenCoreResidualSAE

Feature extraction

Max pooling

Diffing

NPMI

Feature highlighting

Steering

UMAP

### Pipeline 2

F2LLM embeddings

Phrase splitting

BatchTopK SAE

Document pooling

Classification

### Infrastructure

bf16

mmap

sharding

cache

sauvegarde des activations

### Sous-modules externes

- `SAELens` est téléchargé localement sous `external/sae-lens` et utilisé par le code.
- `GemmaScope` n'est pas cloné comme sous-module dans ce dépôt ; le projet utilise des poids SAE préentraînés locaux / Hugging Face.
- `interp_embed` n'est pas téléchargé localement, il est référencé comme source d'inspiration mais pas comme dépendance installée.

---

# Problèmes connus

## Duplication

Une partie importante du code existe déjà dans :

- SAELens
- interp_embed

Supprimer progressivement les réimplémentations.

---

## Architecture monolithique

saev5.py

dual_pipeline_sae.py

sont trop volumineux.

Objectif :

factorisation.

---

## Sparse matrices

Utilisation historique :

SciPy CSR

Objectif :

remplacer par :

PyTorch COO

ou

top-k indices + valeurs

CSR uniquement pour export.

---

# Architecture cible

src/

models/
    gemma.py
    embedding_models.py

sae/
    frozen_core.py
    training.py
    extraction.py

analysis/
    diffing.py
    clustering.py
    retrieval.py
    correlations.py

storage/
    mmap.py
    shards.py
    sparse.py

visualization/
    umap.py
    dashboard.py

evaluation/
    metrics.py
    benchmarks.py

---

# Règles importantes

## 1

Ne jamais réimplémenter une fonctionnalité déjà présente dans :

- SAELens
- interp_embed

sans justification documentée.

---

## 2

Comparer systématiquement les implémentations locales avec :

SAELens

et documenter les différences.

---

## 3

Conserver FrozenCoreResidualSAE.

Cette classe est spécifique au projet.

---

## 4

Conserver compatibilité bf16.

Vérifier qu'aucune conversion implicite :

bf16 -> fp32 -> bf16

n'est introduite.

---

## 5

Préserver les résultats actuels.

Toute refactorisation doit passer les tests de non-régression.

---

# Fonctionnalités futures

## Retrieval interprétable

Recherche :

"mails urgents"

"mise en service"

"panne locale"

"facturation"

à partir des activations SAE.

---

## Diffing de corpus

Comparer :

région A vs région B

année N vs N+1

campagne avant/après

---

## Corrélations

Calcul NPMI.

Identifier :

urgence ↔ panne

colère ↔ facturation

etc.

---

## Dashboard

Interface Streamlit.

Visualisation :

- UMAP
- features activées
- exemples positifs
- exemples négatifs
- recherche

---

# Documentation

Maintenir automatiquement :

README.md

docs/architecture.md

docs/experiments.md

docs/references.md

---

# Rapport de recherche

Créer progressivement :

report/

avec :

- état de l'art
- architecture
- expériences
- résultats
- comparaison avec interp_embed
- comparaison avec GemmaScope
- comparaison avec SAE Boost

Mis à jour à chaque évolution importante.