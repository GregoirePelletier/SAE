# Références

## Bibliothèques et dépôts réutilisés (règle n°1 de `Context.md`)

| Nom | Rôle dans le projet | Statut de la comparaison (règle n°2) |
|---|---|---|
| **SAELens** ([jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)) | Package pip (`sae-lens>=6.0.0`) utilisé pour charger/encoder le SAE GemmaScope-2 préentraîné (`src/sae/gemma_scope_loader.py`). Submodule `external/sae-lens` gardé comme référence d'implémentation. | Comparaison **de formule faite** (session v10, cf. note ci-dessous) ; comparaison **chiffrée en conditions identiques non faite** (nécessiterait de faire passer le chargement Gemma-3/GemmaScope par `HookedTransformer`+`ActivationsStore`, non fait par choix — cf. note). |
| **GemmaScope** ([google-deepmind/gemma-scope](https://github.com/google-deepmind/gemma-scope)) | Poids SAE préentraînés téléchargés depuis HuggingFace Hub (`download_sae.py`), pas cloné comme submodule. Fournit les features "core" du Pipeline 1. | N/A (poids utilisés tels quels, pas de réimplémentation). |
| **Interpretable Embeddings with Sparse Autoencoders** ([nickjiang2378/interp_embed](https://github.com/nickjiang2378/interp_embed)) | Inspiration méthodologique (papier : *Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit*), non installé/vendorisé. `tests/test_interp_embed_diff.py` compare optionnellement `corpus_diff_stats` à `diff_features` d'interp_embed si le package est présent. | Comparaison **partielle** (test optionnel, dépend de la présence du package non installé par défaut). |
| **SAE Boost** | Mentionné dans les objectifs initiaux (`Context.md`), implémentation officielle la plus récente à rechercher. | **Non fait.** Aucune intégration ni comparaison à date. |
| **Neuronpedia** ([neuronpedia.org](https://www.neuronpedia.org)) | Source des labels officiels des features GemmaScope "core", via téléchargement en masse des lots `.jsonl.gz` du bucket S3 public `neuronpedia-datasets` (l'ancienne route REST `/api/explanation/export` est cassée). Cache local canonique : `local_data/neuronpedia_labels/`. | N/A (source de données externe, pas de code à comparer). |
| **F2LLM-v2** (`codefuse-ai/F2LLM-v2-{80M,160M,330M}`) | Modèle d'embeddings de phrases pour le Pipeline 2 (`src/sae/phrase_sae.py`). | N/A. |
| **transformer_lens** | Dépendance de SAELens pour le hooking des activations. | N/A (dépendance transitive). |

## Comparaison FVE/variance expliquée avec SAELens (règle n°2)

`sae_lens.evals.get_sparsity_and_variance_metrics` (package pip installé,
`.venv/lib/.../sae_lens/evals.py`) calcule la variance expliquée de deux façons
différentes, maintenues en parallèle dans leur propre code :

- `explained_variance_legacy` : `1 - resid_sum_of_squares / batched_variance_sum`,
  calculé **par token** puis moyenné. `batched_variance_sum` centre chaque
  dimension sur sa moyenne **batch** avant de sommer sur les dimensions —
  structurellement la même idée que notre `compute_metrics` (résidu au carré
  normalisé par une variance centrée par dimension).
- `explained_variance` (qualifiée de "nouvelle formule correcte" dans leurs
  propres commentaires de code) : agrège d'abord `E[x²]` et `E[x]²` par
  dimension à l'échelle du jeu de données entier, PUIS calcule
  `1 - variance_résiduelle/variance_totale` une seule fois — pas une moyenne de
  ratios par token.

Notre `src/analysis/metrics.py::compute_metrics` calcule
`mse = mean_élémentwise((x - x̂)²)` et
`variance = mean_élémentwise((x - x.mean(dim=0))²)`, moyennés sur tokens ET
dimensions en une seule fois (pas de moyenne de ratios par token) — plus proche
dans sa structure d'agrégation de la "nouvelle formule" de SAELens que de leur
formule "legacy", bien que le détail de centrage diffère légèrement (SAELens
centre par dimension sur la moyenne du batch en cours ; notre formule fait de
même via `acts.mean(dim=0, keepdim=True)`).

**Conclusion** : les deux formules mesurent le même concept (variance expliquée
= 1 - variance résiduelle normalisée) et sont structurellement compatibles,
mais SAELens documente elle-même deux variantes légèrement différentes selon
l'ordre d'agrégation (par token vs global) — un rappel que ce n'est pas une
formule unique et stabilisée même dans la référence. Une comparaison chiffrée
directe sur les mêmes activations nécessiterait de faire passer le chargement
de Gemma-3 + GemmaScope-2 par `transformer_lens.HookedTransformer` +
`sae_lens.ActivationsStore` (l'API attendue par `run_evals`), ce que ce projet
évite délibérément (`src/sae/gemma_scope_loader.py` a été écrit spécifiquement
pour contourner des incompatibilités de chargement direct constatées avec
GemmaScope-2, cf. `Context.md`) — non fait dans cette session, proposé comme
piste dans `report/04_limites_et_perspectives.md`.

## Protocoles/méthodes issus de la littérature

- **Odd-one-out / auto-interprétation par juge LLM** : protocole inspiré de SAEBench
  (feature-detection) et de Bills et al. 2023 (ρ_interp, corrélation Spearman entre le
  score du juge et l'activation réelle) — implémenté dans
  `src/sae/judge.py::odd_one_out_judge`/`local_gemma_judge`.
- **Latent Terms (BM25 sur vocabulaire latent SAE)** : Clavié et al. 2026, implémenté
  dans `src/sae/retrieval/latent_terms.py`.
- **BatchTopK + AuxK** : architecture d'entraînement SAE utilisée pour `PhraseLevelSAE`
  et l'extension `ExtendedSAE` (`src/sae/batch.py`, `src/sae/frozen_core.py`).
- **Diffing de corpus (Fisher exact + correction Benjamini-Hochberg)** :
  `src/analysis/cooccurrence.py::corpus_diff_stats`, remplace un diffing naïf par
  écarts de fréquence sans contrôle du taux de faux positifs.

## Modèles

| Modèle | Rôle | Taille |
|---|---|---|
| `google/gemma-3-12b-it` | Modèle cible de production (extraction hidden states + juge LLM) | 12B |
| `google/gemma-3-{4b,1b,270m}-it` | Profils alternatifs (`MODEL_SIZE`), 270m pour validation rapide locale | 4B/1B/270M |
| `google/gemma-scope-2-{12b,4b,1b,270m}-it` | SAE préentraînés GemmaScope-2 correspondants | — |
| `codefuse-ai/F2LLM-v2-{80M,160M,330M}` | Embeddings de phrase (Pipeline 2) | 80M-330M |
| `BAAI/bge-m3` | Présent dans `models/`, non branché dans le pipeline actuel à date | — |
