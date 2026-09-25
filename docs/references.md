# Références

Code, papiers et modèles utilisés par le projet, et écarts connus avec eux.

## Bibliothèques et dépôts

| Nom | Usage dans le projet | Écarts et remarques |
|---|---|---|
| [SAELens](https://github.com/jbloomAus/SAELens) | Chargement et encodage du SAE GemmaScope-2 (`src/sae/gemma_scope_loader.py`, paquet `sae-lens`). Le sous-module `external/sae-lens` sert de référence. | La variance expliquée est calculée avec notre propre formule (voir plus bas). L'entraînement de l'extension n'utilise pas SAELens, qui ne prévoit pas un SAE secondaire entraîné sur le résidu d'un SAE gelé. |
| [GemmaScope](https://github.com/google-deepmind/gemma-scope) | Poids des SAE préentraînés (CORE), téléchargés par `download_sae.py`. | Poids utilisés tels quels. |
| [interp_embed](https://github.com/nickjiang2378/interp_embed) (Jiang, Sun et al. 2025, *Interpretable Embeddings with Sparse Autoencoders*) | Référence pour la recherche d'emails, le diffing, les corrélations et le nommage de features. Sous-module `external/interp_embed`. | Correspondance détaillée dans `docs/archive/INTERP_EMBED_COVERAGE.md`. Nous sélectionnons les features par similarité d'embeddings plutôt que par sous-chaîne, filtrons les corrélations triviales et nommons les features par test de l'intrus. Le score de silhouette, que le papier juge non pertinent, reste calculé par continuité avec les résultats anciens. Leurs exemples n'ont jamais été exécutés pour vérifier notre lecture. MAP, MP@K et RRF (k = 60) suivent les conventions usuelles, faute de code de référence. |
| [Neuronpedia](https://www.neuronpedia.org) | Descriptions des features CORE, téléchargées depuis le bucket public `neuronpedia-datasets` et mises en cache dans `local_data/neuronpedia_labels/`. | L'ancienne API d'export ne fonctionne plus. |
| transformers | Extraction des activations de Gemma-3 (`output_hidden_states` et crochets PyTorch). | transformer_lens n'est qu'une dépendance de SAELens ; nnsight et baukit n'ont pas été évalués. |

## Architectures de SAE

| Référence | Usage | Vérification |
|---|---|---|
| SAE Boost ([Koriagin et al., COLM 2025](https://arxiv.org/abs/2507.12990)) | Modèle de l'extension (`SAEBoostResidualSAE`) : SAE secondaire entraîné sur le résidu d'un SAE gelé, de même taille (1 024) que dans le papier. | Les alternatives du papier (autres initialisations, fusion de SAE, ajustement complet) n'ont pas été comparées. `K_EXTRA=5` suit leur réglage optimal. |
| BatchTopK ([Bussmann, Leask, Nanda](https://arxiv.org/abs/2412.06410)) | Parcimonie de l'extension et du PhraseLevelSAE (`src/sae/batch.py`) : top-k sur le lot pendant l'entraînement, seuil global à l'inférence. | Conforme au papier (coefficient de perte auxiliaire 1/32, seuil estimé pendant l'entraînement). Seule différence : taille de la perte auxiliaire adaptée à un dictionnaire plus petit. |
| JumpReLU (Rajamanoharan et al. 2024) | Architecture du SAE CORE, utilisé tel quel. | — |
| Sanity Checks for Sparse Autoencoders ([Korznikov et al. 2026](https://arxiv.org/abs/2602.14111)) | Contrôle avec un décodeur aléatoire figé (`SANITY_CHECK_FROZEN_DECODER=1`). | Reproduit (`RESULTS_TESTS.md` §19 et §119) : l'interprétabilité ne distingue pas le décodeur entraîné du décodeur aléatoire, la variance expliquée si. |
| Cunningham et al. 2023 ([arXiv:2309.08600](https://arxiv.org/abs/2309.08600)) | Référence de contexte sur l'usage des SAE pour l'interprétabilité. | — |

## Méthodes

- **Nommage des features par test de l'intrus** : inspiré de SAEBench et de Bills et al. 2023
  (`src/sae/judge.py`). Le juge doit retrouver, parmi des exemples, celui qui n'active pas la
  feature.
- **Latent Terms** (Clavié et al. 2026) : BM25 sur le vocabulaire d'un SAE entraîné sur des
  tokens d'un corpus général (`src/sae/retrieval/latent_terms.py`). Il n'existe pas de code
  officiel ; la seule réimplémentation tierce (en JAX) n'est pas reprise. Aucun résultat obtenu
  avec la version actuelle.
- **Diffing de corpus** : test exact de Fisher par feature et correction de Benjamini-Hochberg
  (`src/analysis/cooccurrence.py::corpus_diff_stats`).

## Variance expliquée : pourquoi notre formule

SAELens propose deux formules de variance expliquée ; la nôtre
(`src/analysis/metrics.py::compute_metrics`) centre l'erreur et la variance dimension par
dimension. Sur le même SAE et les mêmes 4 096 tokens d'emails
(`scripts/saelens_numeric_comparison.py`) :

| Formule | Valeur |
|---|---:|
| la nôtre | 0,831 |
| SAELens, calcul par token (`explained_variance_legacy`) | 0,406 |
| SAELens, normes globales (`explained_variance`) | 1,000 |

Gemma-3 concentre une grande partie de la norme de ses activations sur quelques dimensions
énormes et presque constantes (« massive activations », Sun et al. 2024 ; sur nos données, une
dimension atteint environ 75 000 contre environ 53 en moyenne). La formule globale est dominée
par ces dimensions, faciles à reconstruire, et donne 1,0 même pour un SAE médiocre. En centrant
chaque dimension, une dimension constante ne pèse presque plus dans la variance, et les erreurs
sur les autres restent visibles. Nous gardons donc notre formule. Pour étudier ces dimensions
séparément, voir `scripts/test_massive_acts.py`.

## Alternatives non retenues

| Besoin | Alternative existante | Choix |
|---|---|---|
| Nommage automatique des features | EleutherAI `delphi` (détection, fuzzing, simulation) | Protocole maison, jamais comparé à `delphi`. Ses défauts ont été trouvés et corrigés en cours de route (`RESULTS_TESTS.md` §113 à §119). |
| Entraînement de SAE | `SAETrainingRunner` (SAELens), `dictionary_learning` | Code maison, nécessaire pour l'architecture CORE gelé et extension. |
| Évaluation de SAE | SAEBench | Métriques maison : les taux obtenus ne sont pas directement comparables à la littérature. |

## Pistes de la littérature non intégrées

- **Matryoshka SAEs** ([arXiv:2503.17547](https://arxiv.org/abs/2503.17547)) : dictionnaires
  emboîtés contre la fragmentation des features. Sans rapport avec `MATRYOSHKA_DIM`, qui tronque
  les embeddings F2LLM. Non essayé ; doubler la taille de l'extension n'avait rien changé.
- **ClassifSAE** ([arXiv:2506.23951](https://arxiv.org/abs/2506.23951)) : SAE entraîné avec un
  classifieur, pour des concepts utiles à une tâche (par exemple l'urgence). Non implémenté.
- **Features instables, sous-espaces reproductibles**
  ([arXiv:2606.12138](https://arxiv.org/abs/2606.12138),
  [arXiv:2605.31245](https://arxiv.org/abs/2605.31245)) : observé aussi ici (`RESULTS_TESTS.md`
  §21 et §120, et E05).
- **LLM multilingues** (Resck et al., EMNLP 2025 ; [arXiv:2507.11230](https://arxiv.org/abs/2507.11230)) :
  un juge interrogé en français ou en anglais donne le même taux global, mais 39 % des features
  changent de statut (`RESULTS_TESTS.md` §22).
- **Revue des SAE** (Shu, Wu, Zhao et al., EMNLP 2025) : cadre pour situer les explications par
  les entrées (nos tests) et par les sorties (le pilotage d'activations, testé en §24 avec des
  résultats irréguliers).
- **SPLARE** (Formal et al., ICLR 2026) : moteur de recherche entraîné sur le vocabulaire d'un SAE.
  Non essayé ; cité pour le choix de la couche (environ aux deux tiers du modèle).
- **Beckmann et Queloz 2026** : réflexion sur ce que l'interprétabilité mécaniste permet de dire
  de la « compréhension » d'un modèle ; cadrage seulement.

## Modèles

| Modèle | Rôle |
|---|---|
| `google/gemma-3-{12b,4b,1b,270m}-it` | modèle analysé (`MODEL_SIZE`) ; 12B par défaut, 1B pour la campagne post-soutenance |
| `google/gemma-scope-2-*` | SAE CORE correspondants |
| Qwen3.8-27B | modèle juge (`JUDGE_MODEL_ID`), différent du modèle analysé |
| `BAAI/bge-m3` | similarité entre noms de features et requêtes ; représentation dense de comparaison |
| `codefuse-ai/F2LLM-v2-{80M,160M,330M}` | embeddings de phrases du pipeline alternatif |
