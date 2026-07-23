# Références

## Bibliothèques et dépôts réutilisés (règle n°1 de `Context.md`)

| Nom | Rôle dans le projet | Statut de la comparaison (règle n°2) |
|---|---|---|
| **SAELens** ([jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)) | Package pip (`sae-lens>=6.0.0`) utilisé pour charger/encoder le SAE GemmaScope-2 préentraîné (`src/sae/gemma_scope_loader.py` — un converter, pas une réimplémentation : le SAE chargé EST un objet `sae_lens.SAE` natif). Submodule `external/sae-lens` gardé comme référence d'implémentation. | **Comparaison chiffrée faite** (`scripts/saelens_numeric_comparison.py`, cf. note ci-dessous) : désaccord numérique important entre notre formule et les deux formules maintenues par `sae_lens.evals` elles-mêmes, sur le même SAE et les mêmes activations. |
| **GemmaScope** ([google-deepmind/gemma-scope](https://github.com/google-deepmind/gemma-scope)) | Poids SAE préentraînés téléchargés depuis HuggingFace Hub (`download_sae.py`), pas cloné comme submodule. Fournit les features "core" du Pipeline 1. | N/A (poids utilisés tels quels, pas de réimplémentation). |
| **Interpretable Embeddings with Sparse Autoencoders** ([nickjiang2378/interp_embed](https://github.com/nickjiang2378/interp_embed)) | Inspiration méthodologique (papier : *Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit*, Jiang/Sun et al. 2025, `pdf/InterpretableSAE_Embeddings.pdf`), non installé/vendorisé. `tests/test_interp_embed_diff.py` compare optionnellement `corpus_diff_stats` à `diff_features` d'interp_embed si le package est présent. | **Comparaison méthodologique détaillée faite** (session v10, `RESULTS_TESTS.md` §15) : relecture ligne à ligne (labellisation Appendix C, corrélations §4.2/Appendix E.1, clustering/retrieval §4.3/4.4/Appendix F.1) contre notre code. 4 écarts trouvés et corrigés : matching par sous-chaîne → similarité d'embedding (retrieval/clustering), filtre de corrélations "intéressantes" manquant (jamais câblé), marqueurs erronés sur les exemples négatifs, gate odd-one-out vs génération contrastive directe (piste documentée, non intégrée en production). Comparaison de `corpus_diff_stats` (test optionnel dépendant du package) reste inchangée. |
| **SAE Boost** ([*Teach Old SAEs New Domain Tricks with Boosting*](https://arxiv.org/abs/2507.12990), Koriagin et al., COLM 2025, `pdf/teacholdsaes.pdf`) | **Identifié a posteriori** (relecture des PDF de référence, cf. `RESULTS_TESTS.md` §18) : `FrozenCoreResidualSAE`/`ExtendedSAE` (`src/sae/frozen_core.py`) EST une implémentation de SAE Boost — même architecture exacte (SAE secondaire entraîné sur le résidu de reconstruction `e = x - x̂` d'un SAE core gelé, sommé à l'inférence), sans que le projet ne l'ait jamais identifiée ni citée comme telle. Coïncidence notable : la taille de dictionnaire résiduel du papier (1024) correspond exactement à notre `D_EXTRA` par défaut. | **Comparaison méthodologique faite** (`RESULTS_TESTS.md` §18) : notre `K_EXTRA=32` est nettement moins parcimonieux que leur k=5 optimal (leur étude de sensibilité, Table 12, montre un compromis interprétabilité/domain-EV en faveur d'un k faible) — piste d'ablation non testée. Écart plus important : leur étude montre qu'un SAE résiduel a besoin de **100-200M tokens** pour converger sans dégrader les performances générales (jusqu'à -31% d'EV générale en dessous de 100M) ; notre ablation volume (`RESULTS_TESTS.md` §5/§12) n'a testé que 100k-2M tokens, 50-100x en dessous de ce seuil — notre conclusion "le volume ne change rien" pourrait donc ne pas être valide dans le régime où le papier montre un effet réel. Aucune comparaison chiffrée directe avec leurs baselines alternatives (Extended SAE random/most-active init, SAE Stitching, full fine-tuning) n'a été menée à ce jour. |
| **Sanity Checks for Sparse Autoencoders** ([Korznikov et al., 2026](https://arxiv.org/abs/2602.14111), `pdf/sanitychecks.pdf`) | Non mentionné dans le cadrage initial, mais directement pertinent : leur baseline "Frozen Decoder" (décodeur figé à une initialisation aléatoire, jamais entraîné) égale un SAE réellement entraîné sur interprétabilité (AutoInterp), sparse probing et causal editing (RAVEL) dans leur étude — remettant en cause la validité de ces métriques comme preuve d'un apprentissage de features significatif. | **Sanity check reproduit sur notre extension P1** (`FrozenDecoderExtendedSAE`, `src/sae/frozen_core.py`, `SANITY_CHECK_FROZEN_DECODER=1`) : cf. `RESULTS_TESTS.md` §19 pour le protocole et les résultats (comparaison directe du taux d'interprétabilité odd-one-out et de la sonde de classification contre le run principal, à corpus/volume/largeur identiques). |
| **Neuronpedia** ([neuronpedia.org](https://www.neuronpedia.org)) | Source des labels officiels des features GemmaScope "core", via téléchargement en masse des lots `.jsonl.gz` du bucket S3 public `neuronpedia-datasets` (l'ancienne route REST `/api/explanation/export` est cassée). Cache local canonique : `local_data/neuronpedia_labels/`. | N/A (source de données externe, pas de code à comparer). |
| **F2LLM-v2** (`codefuse-ai/F2LLM-v2-{80M,160M,330M}`) | Modèle d'embeddings de phrases pour le Pipeline 2 (`src/sae/phrase_sae.py`). | N/A. |
| **transformer_lens** | Dépendance de SAELens pour le hooking des activations. | N/A (dépendance transitive). |

## Architectures SAE fondamentales (implémentées directement dans ce projet)

| Nom | Rôle dans le projet | Statut de vérification |
|---|---|---|
| **BatchTopK SAE** (Bussmann, Leask, Nanda, [arXiv:2412.06410](https://arxiv.org/abs/2412.06410), `pdf/BatchTopK.pdf`) | Mécanisme de parcimonie de `ExtendedSAE`/`PhraseLevelSAE` (`src/sae/batch.py::BatchTopKEncoder`) : sélection des top-(k·B) pré-activations sur le batch en entraînement, conversion en seuil global θ (JumpReLU) pour l'inférence déterministe par échantillon. | **Vérifié fidèle** (relecture du papier face au code) : notre `AUX_ALPHA=1/32` correspond exactement à leur coefficient recommandé pour la loss auxiliaire anti-features-mortes ; notre estimation de θ par EMA pendant l'entraînement puis application `z·1[z>θ]` à l'inférence reproduit exactement leur protocole "on convertit BatchTopK en JumpReLU global pour lever la dépendance au batch". Écart mineur assumé : `k_aux = min(2·k_extra, d_extra/2)` chez nous contre une valeur absolue fixe (512) chez eux — adaptation nécessaire vu notre `D_EXTRA` (1024) bien plus petit que leurs dictionnaires (12k-16k). |
| **JumpReLU SAE** (Rajamanoharan et al., 2024, `pdf/jumpRELU.pdf`) | Architecture du SAE core GemmaScope-2 préentraîné (Pipeline 1), utilisé tel quel via SAELens — pas de code à nous dans ce cas, seulement la lecture des poids. | N/A (poids tiers, aucune réimplémentation de notre part à vérifier). |
| **Sparse Autoencoders Find Highly Interpretable Features** (Cunningham et al., 2023, [arXiv:2309.08600](https://arxiv.org/abs/2309.08600), `pdf/2309.08600v3.pdf`) | Un des deux papiers fondateurs de l'usage des SAE pour l'interprétabilité des LLM (avec Bricken et al. 2023), déjà cité de façon informelle dans tout le projet. | Référence de contexte, pas de comparaison de code (papier fondateur, pas une implémentation spécifique à comparer). |

## Littérature complémentaire consultée (relecture `pdf/`, perspectives non intégrées)

| Nom | Pertinence pour ce projet | Statut |
|---|---|---|
| **Matryoshka SAEs** (Bussmann, Nabeshima, Karvonen, Nanda, [arXiv:2503.17547](https://arxiv.org/abs/2503.17547), `pdf/Matryoshka.pdf`) | **Attention à ne pas confondre** avec `MATRYOSHKA_DIM` (`src/config.py`) : ce dernier ne fait que tronquer les embeddings F2LLM à une dimension donnée (propriété du modèle d'embedding lui-même, "Matryoshka Representation Learning"), sans rapport avec les "Matryoshka SAEs" du papier (dictionnaires SAE emboîtés entraînés simultanément, pour éviter le *feature splitting*/*absorption* aux grandes tailles de dictionnaire). Piste pertinente non testée : le résidu non-interprété (~45-59%, chapitre 3) pourrait en partie provenir de *feature splitting/absorption* dans `ExtendedSAE` (notre ablation capacité, `D_EXTRA` 1024→2048, montre déjà qu'agrandir le dictionnaire seul n'aide pas — cohérent avec le phénomène que ce papier documente). | **Non implémenté** (changement d'architecture d'entraînement plus substantiel que le sanity check Frozen Decoder) — piste de poursuite documentée en `report/04_limites_et_perspectives.md`. |
| **ClassifSAE** (Le Bail, Dentan, Buscaldi, Vanier, [arXiv:2506.23951](https://arxiv.org/abs/2506.23951), `pdf/UnveilingDecision-MakinginLLMsforTextClassification.pdf`) | SAE **supervisé**, entraîné conjointement avec un classifieur sur un sous-ensemble du dictionnaire + une pénalité de parcimonie sur le taux d'activation — conçu spécifiquement pour extraire des concepts interprétables ET causalement influents pour une tâche de classification de texte. Directement pertinent pour les objectifs "détection d'urgence"/"détection d'intention" de ce projet, qui utilise actuellement une sonde de classification post-hoc (`downstream_classification`) sur un SAE entraîné de façon totalement non supervisée. | **Non implémenté** — piste de poursuite documentée en `report/04_limites_et_perspectives.md` (nécessiterait une boucle d'entraînement SAE+classifieur jointe, absente de l'architecture actuelle). |
| **A Survey on Sparse Autoencoders** (Shu, Wu, Zhao et al., EMNLP 2025 Findings, `pdf/SurveySAE.pdf`) | Taxonomie utile pour situer nos méthodes : explications "input-based" (nos protocoles odd-one-out et labellisation contrastive) vs "output-based" (steering — `steer_activations`/`steer_and_decode`, `src/sae/sae_shared.py`, déjà implémenté mais jamais évalué comme méthode d'explication à part entière) ; métriques "structurelles" (NMSE/FVE/L0) vs "fonctionnelles" (nos sondes de classification, fidélité/plausibilité). | Utilisé pour enrichir le cadrage de `report/01_etat_de_lart.md` ; le steering comme méthode d'explication output-based reste une piste d'évaluation non exploitée. |
| **Explainability and Interpretability of Multilingual LLMs: A Survey** (Resck, Augenstein, Korhonen, EMNLP 2025, `pdf/2025.emnlp-main.1033.pdf`) | Rappel pertinent pour ce projet (corpus et juge en français) : les LLM traitent souvent les concepts multilingues via une représentation dominée par l'anglais, un facteur de confusion potentiel jamais quantifié ici pour le juge d'auto-interprétation (Gemma-3-12B-it, prompts en français). | Cité comme limite/piste de vigilance, aucune mesure quantitative faite sur ce point dans ce projet. |
| **Mechanistic Indicators of Understanding in Large Language Models** (Beckmann & Queloz, 2026, `pdf/MechanisticIndicatorsinLLM.pdf`) | Cadrage philosophique de l'interprétabilité mécaniste (à quel titre l'interprétabilité mécaniste permet-elle de parler de "compréhension" chez un LLM) — utilisé pour la motivation générale du projet, pas de technique actionnable. | Cité en introduction (`report/00_introduction.md`) pour le cadrage motivationnel uniquement. |

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
