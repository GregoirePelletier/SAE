# Références

## Bibliothèques et dépôts réutilisés (règle n°1 de `Context.md`)

| Nom | Rôle dans le projet | Statut de la comparaison (règle n°2) |
|---|---|---|
| **SAELens** ([jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)) | Package pip (`sae-lens>=6.0.0`) utilisé pour charger/encoder le SAE GemmaScope-2 préentraîné (`src/sae/gemma_scope_loader.py` — un converter, pas une réimplémentation : le SAE chargé EST un objet `sae_lens.SAE` natif). Submodule `external/sae-lens` gardé comme référence d'implémentation. | **Comparaison chiffrée faite** (`scripts/saelens_numeric_comparison.py`, cf. note ci-dessous) : désaccord numérique important entre notre formule et les deux formules maintenues par `sae_lens.evals` elles-mêmes, sur le même SAE et les mêmes activations. |
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
  dimension sur sa moyenne **batch** avant de sommer sur les dimensions.
- `explained_variance` (qualifiée de "nouvelle formule correcte" dans leurs
  propres commentaires de code) : agrège d'abord `E[‖x‖²]` et `E[x]²` à l'échelle
  du jeu de données entier, PUIS calcule `1 - variance_résiduelle/variance_totale`
  une seule fois — pas une moyenne de ratios par token.

Notre `src/analysis/metrics.py::compute_metrics` calcule
`mse = mean_élémentwise((x - x̂)²)` et
`variance = mean_élémentwise((x - x.mean(dim=0))²)`, moyennés sur tokens ET
dimensions en une seule fois.

### Comparaison chiffrée (session v10, `scripts/saelens_numeric_comparison.py`)

Les trois formules ont été calculées sur le **même** SAE (objet `sae_lens.SAE` natif,
chargé via `load_gemma_scope_sae`) et les **mêmes** activations (4096 tokens réels
d'emails déjà en cache, `p1_eval_raw_tokens.pt`) :

| Formule | Valeur |
|---|---|
| Notre `compute_metrics` (FVE) | **0,831** |
| `explained_variance_legacy` (sae_lens, par token) | **0,406** |
| `explained_variance` "corrigée" (sae_lens, agrégation globale) | **1,000** |

**Désaccord numérique important entre les trois formules sur les mêmes données** —
expliqué par les activations massives documentées de Gemma-3 (`Context.md`, section
bf16) : sur cet échantillon, une seule dimension atteint une magnitude ~74 752 contre
une magnitude moyenne ~53 (ratio >1400×), et domine la norme L2 de la quasi-totalité
des tokens (norme moyenne ~50 785, cohérente avec la dimension outlier seule). La
formule "corrigée" de sae_lens somme sur les dimensions AVANT de normaliser : si le
SAE reconstruit correctement cette unique dimension géante (en erreur absolue, même
une erreur relative non négligeable sur cette dimension reste petite comparée à sa
magnitude), la variance expliquée globale est mécaniquement écrasée vers 1,0, sans
refléter la qualité de reconstruction des dimensions "normales" (les 3839 autres). La
formule "legacy" (normalisation par token) et la nôtre (normalisation par dimension)
sont moins sensibles à ce phénomène mais restent sensiblement différentes entre elles
(0,41 vs 0,83), ce qui montre que le choix précis de normalisation n'est pas neutre
en présence d'activations aussi hétérogènes en magnitude.

**Conclusion** : la variance expliquée n'est pas une métrique unique et stable sur
Gemma-3 — le classement (0,41 / 0,83 / 1,00 selon la formule) dépend fortement de la
manière dont les dimensions à magnitude extrême sont pondérées dans l'agrégation.
Toute lecture de FVE/NMSE sur ce projet doit être accompagnée de la formule exacte
utilisée ; un score unique sans cette précision est peu interprétable. Recommandation
pour la suite : ajouter une métrique robuste aux outliers (médiane des ratios par
token plutôt que moyenne, ou variance expliquée par dimension pondérée uniformément)
plutôt que de choisir arbitrairement entre les trois formules existantes.

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
