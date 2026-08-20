# Sparse Autoencoders (SAE) for Interpretable Text Analysis

Analyse interprétable de mails clients EDF (et de corpus publics de substitution) via
Sparse Autoencoders sur les hidden states de Gemma-3, avec labellisation des features
par GemmaScope-2 / Neuronpedia et par un juge LLM local.

Deux pipelines :

- **Pipeline 1** : Gemma-3 (hidden states, couche `LAYER`) → SAE GemmaScope-2 préentraîné
  (+ extension `FrozenCoreResidualSAE`/`SAEBoostResidualSAE` optionnelle) → max-pool documentaire.
- **Pipeline 2** : F2LLM-v2 (embeddings de phrases) → `PhraseLevelSAE` entraîné from-scratch
  (BatchTopK + AuxK) → max-pool documentaire.

Détail de l'architecture : `docs/architecture.md`. Installation, cluster SLURM,
dépannage Windows/HuggingFace : `docs/ops.md`.

---

## Corpus d'entraînement

Le SAE d'extension (`SAEBoostResidualSAE`) et le `PhraseLevelSAE` s'entraînent sur les
**mails originaux + variantes augmentées** (`local_data/emails/`), qui
dominent le train (~41k/2,2k docs train/test). Le corpus generic
energy/sports/support (FineWeb-2/Wikipedia) sert uniquement à une
démonstration de diffing cross-domaine, encodée post-hoc, sans participer à
l'entraînement.

L'appariement de domaine entre le corpus d'entraînement de l'extension et le
corpus cible conditionne directement l'interprétabilité mesurée des features :
sur un corpus hors-domaine (energy/sports/support), le taux d'interprétabilité
(protocole odd-one-out) est de 20% (2/10) ; sur le corpus emails, il est de
45,3% (68/150). Le volume d'entraînement, testé de 100k à 2M tokens à corpus
identique, n'a lui aucun effet mesurable. Détail du diagnostic et des runs de
validation : `RESULTS_TESTS.md` §12.

---

## Démarrage rapide

```bash
# 1. Télécharger le modèle + le SAE (cible par défaut : 12b)
python download_sae.py

# 2. Récupérer les labels Neuronpedia (optionnel mais recommandé)
python -c "from src.sae.neuronpedia_labels import fetch_neuronpedia_labels; \
  fetch_neuronpedia_labels(model_id='gemma-3-12b-it', layer=24, width='16k', \
  cache_path='local_data/neuronpedia_labels/neuronpedia_labels_24-gemmascope-2-res-16k.json')"

# 3. Lancer la pipeline complète
PYTHONPATH=. python src/sae/saev5.py
```

Installation, accès HuggingFace (gated), dépannage Windows : `docs/ops.md`.

---

## Configuration (`src/config.py`)

Source unique de vérité pour toute la pipeline — toutes les valeurs sont surchargeables
par variable d'environnement. Voir `.env.example` pour un jeu de valeurs prêtes à copier,
avec un profil `12b` (principal) et un profil `270m` (validation rapide, commenté).
Conditions de référence pour les comparaisons expérimentales : `docs/evaluation_protocol.md`.

| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_SIZE` | `12b` | `12b` / `4b` / `1b` / `270m` — sélectionne modèle + SAE via `_PRESETS` |
| `MODEL_ID` | dérivé du preset | Repo HF du modèle (override direct possible) |
| `SAE_ID` | dérivé du preset | Sous-dossier GemmaScope (`layer_X_width_Y_l0_Z`) |
| `DTYPE` | `bf16` | bf16 obligatoire sur Gemma-3 (activations massives, cf. `docs/architecture.md`) |
| `LAYER` | dérivé du preset | Couche du residual stream extraite |
| `USE_FROZEN_CORE` | `1` | Active l'extension `SAEBoostResidualSAE` (Pipeline 1) |
| `D_EXTRA` / `K_EXTRA` | `1024` / `32` | Dimension / sparsité de l'extension |
| `D_SAE` / `K_SPARSE` | `8192` / `16` | Dimension / sparsité du `PhraseLevelSAE` (Pipeline 2) |
| `EMB_MODEL` | `codefuse-ai/F2LLM-v2-80M` | Modèle d'embeddings phrase (Pipeline 2) |
| `SAVE_DIR` | `./results/` | Racine des sorties (résultats + `cache/`) |
| `LOCAL_MAILS_PATH` | `./local_data/emails/Mails.tsv` | Corpus EDF (mails originaux) |
| `LOCAL_AUGMENTED_MAILS_PATH` | `./local_data/emails/augmented_mails.jsonl` | Variantes augmentées acceptées |
| `NEURONPEDIA_LABELS_PATH` | `./local_data/neuronpedia_labels/neuronpedia_labels_{layer}-gemmascope-2-res-{width}.json` | Cache labels Neuronpedia, partagé entre tous les runs |
| `EMAIL_TEST_SPLIT` | `0.05` | Fraction des mails réservée au test, split group-aware par mail d'origine |
| `MAX_AUGMENTED_PER_MAIL` | `13` | Nb max de variantes augmentées conservées par mail original |
| `CLUSTER_OFFLINE_MODE` | `0` | `1` = reproduit l'environnement cluster (cf. `docs/ops.md`) |
| `HF_TOKEN` | — | Token HF pour les repos gated |

---

## Scripts, dashboard, tests

Inventaire détaillé de tous les scripts (`download_sae.py`,
`scripts/baseline_gemmascope.py`, `scripts/run_augmentation.py`,
`scripts/retrieval_demo.py`, `src/sae/compare/pipeline.py`, diagnostics
manuels) : `docs/architecture.md`.

Dashboard interactif (Streamlit, lecture seule des artefacts déjà produits) :

```bash
.venv/bin/python -m streamlit run src/visualization/dashboard.py
```

Suite de tests :

```bash
pytest tests/ -v
```

---

## Rapport et résultats

- `report/` : rapport de stage (chapitres numérotés `00_*` à `07_*`, sources
  uniques ; `RAPPORT_STAGE_UNIVERSITE.tex` et `RAPPORT_STAGE_ENTREPRISE.tex`
  sont les livrables). Limites connues et pistes pour la suite :
  `report/04_limites_et_perspectives.md`.
- `RESULTS_TESTS.md` : cahier de laboratoire (une section par question posée,
  avec méthode statistique et résultat).
- `docs/evaluation_protocol.md` : configuration de référence pour comparer
  les runs entre eux.
