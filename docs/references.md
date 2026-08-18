# Références

## Bibliothèques et dépôts réutilisés

| Nom | Rôle dans le projet | Différences avec ce projet |
|---|---|---|
| **SAELens** ([jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)) | Package pip (`sae-lens>=6.0.0`) utilisé pour charger/encoder le SAE GemmaScope-2 préentraîné (`src/sae/gemma_scope_loader.py` — un converter, pas une réimplémentation : le SAE chargé EST un objet `sae_lens.SAE` natif). Submodule `external/sae-lens` gardé comme référence d'implémentation. | Écart chiffré sur la métrique de variance expliquée (FVE), cf. section dédiée plus bas — la formule maison est retenue pour la raison qui y est expliquée, pas par défaut faute d'alternative. |
| **GemmaScope** ([google-deepmind/gemma-scope](https://github.com/google-deepmind/gemma-scope)) | Poids SAE préentraînés téléchargés depuis HuggingFace Hub (`download_sae.py`), pas cloné comme submodule. Fournit les features "core" du Pipeline 1. | N/A (poids utilisés tels quels, pas de réimplémentation). |
| **Interpretable Embeddings with Sparse Autoencoders** ([nickjiang2378/interp_embed](https://github.com/nickjiang2378/interp_embed)) | Guide de conception pour le retrieval/clustering, les corrélations et la labellisation (papier : *Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit*, Jiang/Sun et al. 2025, `pdf/InterpretableSAE_Embeddings.pdf`). Submodule `external/interp_embed` installé et peuplé. `tests/test_interp_embed_diff.py` compare optionnellement `corpus_diff_stats` à `diff_features` d'interp_embed si le package est présent. | Choix faits en s'écartant du papier (`RESULTS_TESTS.md` §15) : similarité d'embedding plutôt que matching par sous-chaîne pour le retrieval/clustering, filtre de corrélations "intéressantes" ajouté (absent de notre implémentation initiale), gate odd-one-out retenue plutôt que génération contrastive directe (alternative documentée, non intégrée en production). Écart non résolu : la comparaison au papier s'est faite par lecture, jamais en exécutant leur code de référence sur son propre cas jouet — le submodule est disponible localement pour ça, pas encore fait. |
| **SAE Boost** ([*Teach Old SAEs New Domain Tricks with Boosting*](https://arxiv.org/abs/2507.12990), Koriagin et al., COLM 2025, `pdf/teacholdsaes.pdf`) | Papier connu et utilisé comme référence de conception pendant le développement de `FrozenCoreResidualSAE`/`ExtendedSAE` (`src/sae/frozen_core.py`) : même architecture (SAE secondaire entraîné sur le résidu de reconstruction `e = x - x̂` d'un SAE core gelé, sommé à l'inférence) et même taille de dictionnaire résiduel par défaut (`D_EXTRA=1024`) que leur configuration. | Alternatives évaluées par le papier (Extended SAE random/most-active init, SAE Stitching, full fine-tuning) non testées ici faute de temps/ressources — le choix retenu s'appuie sur leur papier sans comparaison chiffrée directe à ces autres options. Deux ablations partielles depuis : `K_EXTRA=5` (§25, direction cohérente avec leur k=5 optimal, non significatif seul, n=150) ; volume testé jusqu'à 25M tokens (§23.4, même conclusion qualitative) mais le run à 100-200M (leur borne exacte, job 41658) reste à faire — cf. `docs/evaluation_protocol.md`. |
| **Sanity Checks for Sparse Autoencoders** ([Korznikov et al., 2026](https://arxiv.org/abs/2602.14111), `pdf/sanitychecks.pdf`) | Non mentionné dans le cadrage initial, adopté en cours de route : leur baseline "Frozen Decoder" (décodeur figé à une initialisation aléatoire, jamais entraîné) égale un SAE réellement entraîné sur interprétabilité (AutoInterp), sparse probing et causal editing (RAVEL) dans leur étude — remettant en cause la validité de ces métriques comme preuve d'un apprentissage de features significatif. | Sanity check reproduit sur notre extension P1 (`FrozenDecoderExtendedSAE`, `src/sae/frozen_core.py`, `SANITY_CHECK_FROZEN_DECODER=1`) — cf. `RESULTS_TESTS.md` §19 pour le protocole et les résultats. |
| **Neuronpedia** ([neuronpedia.org](https://www.neuronpedia.org)) | Source des labels officiels des features GemmaScope "core", via téléchargement en masse des lots `.jsonl.gz` du bucket S3 public `neuronpedia-datasets` (l'ancienne route REST `/api/explanation/export` est cassée). Cache local canonique : `local_data/neuronpedia_labels/`. | N/A (source de données externe, pas de code à comparer). |
| **F2LLM-v2** (`codefuse-ai/F2LLM-v2-{80M,160M,330M}`) | Modèle d'embeddings de phrases pour le Pipeline 2 (`src/sae/phrase_sae.py`). | N/A. |
| **transformer_lens** | Dépendance de SAELens elle-même (hooking interne côté `sae_lens`), pas utilisée directement par ce projet : l'extraction des activations Gemma-3 (`src/sae/saev5.py`) passe par `output_hidden_states=True` et un `register_forward_hook` bruts de `transformers`, pas par `HookedTransformer`. Ni `nnsight` ni `baukit` ne sont mentionnés ailleurs dans ce document ou dans le code — non évalués comme alternative au hook manuel actuel. | N/A (dépendance transitive de SAELens ; pas de comparaison faite avec un hooking direct via transformer_lens/nnsight pour ce projet). |

## Architectures SAE fondamentales (implémentées directement dans ce projet)

| Nom | Rôle dans le projet | Statut de vérification |
|---|---|---|
| **BatchTopK SAE** (Bussmann, Leask, Nanda, [arXiv:2412.06410](https://arxiv.org/abs/2412.06410), `pdf/BatchTopK.pdf`) | Mécanisme de parcimonie de `ExtendedSAE`/`PhraseLevelSAE` (`src/sae/batch.py::BatchTopKEncoder`) : sélection des top-(k·B) pré-activations sur le batch en entraînement, conversion en seuil global θ (JumpReLU) pour l'inférence déterministe par échantillon. | **Vérifié fidèle** (relecture du papier face au code) : notre `AUX_ALPHA=1/32` correspond exactement à leur coefficient recommandé pour la loss auxiliaire anti-features-mortes ; notre estimation de θ par EMA pendant l'entraînement puis application `z·1[z>θ]` à l'inférence reproduit exactement leur protocole "on convertit BatchTopK en JumpReLU global pour lever la dépendance au batch". Écart mineur assumé : `k_aux = min(2·k_extra, d_extra/2)` chez nous contre une valeur absolue fixe (512) chez eux — adaptation nécessaire vu notre `D_EXTRA` (1024) bien plus petit que leurs dictionnaires (12k-16k). |
| **JumpReLU SAE** (Rajamanoharan et al., 2024, `pdf/jumpRELU.pdf`) | Architecture du SAE core GemmaScope-2 préentraîné (Pipeline 1), utilisé tel quel via SAELens — pas de code à nous dans ce cas, seulement la lecture des poids. | N/A (poids tiers, aucune réimplémentation de notre part à vérifier). |
| **Sparse Autoencoders Find Highly Interpretable Features** (Cunningham et al., 2023, [arXiv:2309.08600](https://arxiv.org/abs/2309.08600), `pdf/2309.08600v3.pdf`) | Un des deux papiers fondateurs de l'usage des SAE pour l'interprétabilité des LLM (avec Bricken et al. 2023), déjà cité de façon informelle dans tout le projet. | Référence de contexte, pas de comparaison de code (papier fondateur, pas une implémentation spécifique à comparer). |

## Littérature complémentaire consultée (relecture `pdf/`, perspectives non intégrées)

| Nom | Pertinence pour ce projet | Statut |
|---|---|---|
| **Matryoshka SAEs** (Bussmann, Nabeshima, Karvonen, Nanda, [arXiv:2503.17547](https://arxiv.org/abs/2503.17547), `pdf/Matryoshka.pdf`) | **Attention à ne pas confondre** avec `MATRYOSHKA_DIM` (`src/config.py`) : ce dernier ne fait que tronquer les embeddings F2LLM à une dimension donnée (propriété du modèle d'embedding lui-même, "Matryoshka Representation Learning"), sans rapport avec les "Matryoshka SAEs" du papier (dictionnaires SAE emboîtés entraînés simultanément, pour éviter le *feature splitting*/*absorption* aux grandes tailles de dictionnaire). Piste pertinente à garder en tête : le résidu non-interprété (~45-59%, chapitre 3) pourrait en partie provenir de *feature splitting/absorption* dans `ExtendedSAE` (notre ablation capacité, `D_EXTRA` 1024→2048, montre déjà qu'agrandir le dictionnaire seul n'aide pas — cohérent avec le phénomène que ce papier documente). | **Non implémenté** (changement d'architecture d'entraînement plus substantiel que le sanity check Frozen Decoder) — piste de poursuite documentée en `report/04_limites_et_perspectives.md`. |
| **ClassifSAE** (Le Bail, Dentan, Buscaldi, Vanier, [arXiv:2506.23951](https://arxiv.org/abs/2506.23951), `pdf/UnveilingDecision-MakinginLLMsforTextClassification.pdf`) | SAE **supervisé**, entraîné conjointement avec un classifieur sur un sous-ensemble du dictionnaire + une pénalité de parcimonie sur le taux d'activation — conçu spécifiquement pour extraire des concepts interprétables ET causalement influents pour une tâche de classification de texte. Directement pertinent pour les objectifs "détection d'urgence"/"détection d'intention" de ce projet, très intéressant à garder en tête même sans y consacrer de temps d'implémentation pour l'instant. | **Non implémenté** — piste de poursuite documentée en `report/04_limites_et_perspectives.md` (nécessiterait une boucle d'entraînement SAE+classifieur jointe, absente de l'architecture actuelle). |
| **A Survey on Sparse Autoencoders** (Shu, Wu, Zhao et al., EMNLP 2025 Findings, `pdf/SurveySAE.pdf`) | Taxonomie utile pour situer nos méthodes : explications "input-based" (nos protocoles odd-one-out et labellisation contrastive) vs "output-based" (steering — `steer_activations`/`steer_and_decode`, `src/sae/sae_shared.py`) ; métriques "structurelles" (NMSE/FVE/L0) vs "fonctionnelles" (nos sondes de classification, fidélité/plausibilité). | Utilisé pour enrichir le cadrage de `report/01_etat_de_lart.md`. Le steering comme méthode d'explication output-based **a été évalué** (`RESULTS_TESTS.md` §24, `scripts/steering_fidelity_test.py`) : résultat hétérogène et non trivial (round-trip decode/ré-encodage neutralise l'intervention pour 2 intentions sur 4, la préserve pour une troisième, l'amplifie pour la dernière) -- pas un mécanisme d'intervention causale fiable et prévisible. |
| **Explainability and Interpretability of Multilingual LLMs: A Survey** (Resck, Augenstein, Korhonen, EMNLP 2025, `pdf/2025.emnlp-main.1033.pdf`) | Rappel pertinent pour ce projet (corpus et juge en français) : les LLM traitent souvent les concepts multilingues via une représentation dominée par l'anglais, un facteur de confusion potentiel pour le juge d'auto-interprétation (Gemma-3-12B-it, prompts en français). | **Mesuré** (`RESULTS_TESTS.md` §22, `report/03_experiences_et_resultats.md` §13, `scripts/multilingual_judge_bias_test.py`, job 41119) : taux d'interprétabilité non significativement différent entre juge en français (46,9%) et en anglais traduit (45,5%), z=0,24 — mais 38,6% des features changent de statut interprétable/non-interprétable selon la langue (> 31,3% déjà mesuré pour le simple ré-ordonnancement, §13.1). Conclusion : pas de biais anglais systématique détecté, mais confirme que le protocole odd-one-out à décision greedy unique est bruyant face à toute perturbation de surface, pas seulement la langue. |
| **Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages** ([arXiv:2507.11230](https://arxiv.org/abs/2507.11230)) | Motive directement le test de biais multilingue ci-dessus : montre que les SAE peuvent isoler des features spécifiques à une langue plutôt que purement sémantiques, un facteur de confusion supplémentaire pour un juge LLM interrogé dans une langue différente de celle du corpus source. | Cité comme motivation du test (`RESULTS_TESTS.md` §22), pas de réplication directe de leur protocole de détection de features langue-spécifiques dans ce projet. |
| **Unstable Features, Reproducible Subspaces** ([arXiv:2606.12138](https://arxiv.org/abs/2606.12138)) et **Toward Identifiable Sparse Autoencoders** ([arXiv:2605.31245](https://arxiv.org/abs/2605.31245)) | Montrent que les features individuelles d'un SAE varient selon la graine d'entraînement, alors que le sous-espace de bas rang qu'elles couvrent reste, lui, reproductible — très pertinent pour juger de la reproductibilité de ce projet, motive directement l'ablation de variance de seed. | **Ablation reproduite** (`RESULTS_TESTS.md` §21, `report/03_experiences_et_resultats.md` §12, job 41118, `SEED=123` vs `SEED=42` via `CORPUS_SPLIT_SEED` découplé) : taux d'interprétabilité agrégé stable (45,3% vs 47,3%, z=0,35, non significatif) mais seulement 28,2% (22/78) de recouvrement exact des libellés de features entre les deux graines — confirme empiriquement, sur ce projet, la thèse "features instables / sous-espace reproductible" des deux papiers. |
| **Mechanistic Indicators of Understanding in Large Language Models** (Beckmann & Queloz, 2026, `pdf/MechanisticIndicatorsinLLM.pdf`) | Cadrage philosophique de l'interprétabilité mécaniste (à quel titre l'interprétabilité mécaniste permet-elle de parler de "compréhension" chez un LLM) — utilisé pour la motivation générale du projet, pas de technique actionnable. | Cité en introduction (`report/00_introduction.md`) pour le cadrage motivationnel uniquement. |

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

## Métrique de fidélité de reconstruction (FVE) — pourquoi notre formule est retenue

`sae_lens.evals.get_sparsity_and_variance_metrics` calcule la variance expliquée de
deux façons différentes, maintenues en parallèle dans son propre code :
`explained_variance_legacy` (ratio résidu/variance calculé **par token** puis moyenné,
en centrant chaque dimension sur sa moyenne batch) et `explained_variance` (agrégation
`E[‖x‖²]`/`E[x]²` à l'échelle du jeu de données entier, ratio calculé une seule fois).
`src/analysis/metrics.py::compute_metrics` centre et moyenne l'erreur et la variance
**par dimension** (`x.mean(dim=0)`), ni par token ni par norme globale.

Comparaison chiffrée sur le même SAE (`sae_lens.SAE` natif) et les mêmes activations
(`scripts/saelens_numeric_comparison.py`, 4096 tokens réels d'emails,
`p1_eval_raw_tokens.pt`) :

| Formule | Valeur |
|---|---|
| Notre `compute_metrics` (FVE) | 0,831 |
| `explained_variance_legacy` (sae_lens, par token) | 0,406 |
| `explained_variance` (sae_lens, agrégation par norme globale) | 1,000 |

**Cause du désaccord** : Gemma-3 concentre une part disproportionnée de la norme de son
residual stream sur une poignée de dimensions à magnitude énorme et quasi constante
d'un token à l'autre — le phénomène des "massive activations", documenté
indépendamment de ce projet (Sun et al., COLM 2024, *Massive Activations in Large
Language Models* : ratios jusqu'à ~100 000× rapportés, dimensions input-agnostic ; un
suivi 2026, arXiv:2606.20743, identifie spécifiquement 4 dimensions à activation
massive sur Gemma-3-12b-it, dont la dimension 2339). Sur notre échantillon, une seule
dimension atteint ~74 752 contre une magnitude moyenne ~53 (ratio ~1400×) — bien en
dessous des ratios rapportés dans la littérature : ce n'est pas un artefact de notre
pipeline d'extraction, c'est le comportement documenté du modèle à ce point du réseau.

La formule `explained_variance` de sae_lens somme la norme AVANT de normaliser : une
dimension à la fois énorme et quasi constante (donc facile à reconstruire, y compris
pour un SAE médiocre) domine mécaniquement `E[‖x‖²]`, écrasant le ratio vers 1,0 sans
refléter la qualité de reconstruction des 3839 autres dimensions. Notre formule centre
la variance **par dimension** sur sa moyenne batch avant de sommer — une dimension
quasi constante a, par construction, une variance proche de zéro une fois centrée, donc
un poids proche de zéro dans le dénominateur ; toute erreur de reconstruction
résiduelle sur cette même dimension, même modeste en relatif, reste alors visible dans
le numérateur plutôt que noyée par sa propre magnitude. C'est la normalisation qui
correspond à la question posée ("le SAE reconstruit-il la variation réelle du signal
?"), pas celle qui traite la magnitude brute comme le signal à expliquer.

**Conclusion retenue** : `compute_metrics` (notre formule) est la métrique de référence
de ce projet, pour la raison ci-dessus — pas par défaut faute d'alternative. On ne
retient pas l'option d'une métrique "robuste aux outliers" (médiane des ratios,
pondération uniforme par dimension) : les dimensions en cause ne sont pas du bruit à
lisser, ce sont des dimensions identifiables et fonctionnellement significatives du
modèle (Sun et al. les caractérisent comme des termes de biais implicites, pas des
artefacts de mesure) — les traiter comme un outlier générique masquerait un signal réel
plutôt que de corriger un artefact de mesure. Si une décomposition plus fine devient
utile, la bonne direction est d'isoler explicitement les dimensions à activation
massive identifiées par la littérature (comme le fait déjà `scripts/test_massive_acts.py`
pour le diagnostic de pollution), pas une statistique robuste générique.

## Réutilisations envisageables et non retenues

Pour chaque capacité qu'un dépôt tiers aurait pu fournir : ce qui est fait, et ce qui
justifie de ne pas l'avoir adopté.

| Brique | Existant | Choix retenu et raison |
|---|---|---|
| Auto-interprétation | `EleutherAI/delphi` (ex-`sae-auto-interp`) : scorers *detection*/*fuzzing*/*simulation*, prompts et protocoles publiés, comparables à la littérature | Protocole odd-one-out maison (`src/sae/judge.py`), pas benchmarké contre `delphi` faute de temps. Des défauts propres à notre protocole ont été trouvés et documentés en cours de route (biais de sélection du contrôle négatif, ρ_interp non conforme à Bills et al., reconstruction non-déterministe des exemples entre scripts) — corrigés ou explicitement notés, mais rien n'indique qu'adopter `delphi` dès le départ les aurait évités : cette hypothèse reste non testée. |
| Entraînement de SAE | `SAELens` (`SAETrainingRunner`), `saprmarks/dictionary_learning` | Harnais maison (`sae_shared.py`, `phrase_sae.py`), `FrozenCoreResidualSAE`/`ExtendedSAE`. L'architecture centrale du projet (core gelé + résidu entraîné à part, cf. SAE Boost ci-dessus) n'a pas d'équivalent direct dans `SAETrainingRunner`, conçu pour entraîner un SAE unique de bout en bout. SAELens reste utilisé là où c'est pertinent (chargement/encodage du core GemmaScope-2 préentraîné, `gemma_scope_loader.py`) — pas une réimplémentation par principe, un harnais complémentaire pour un besoin que l'existant ne couvre pas. |
| Évaluation de SAE | `SAEBench` (sparse probing, absorption, unlearning, RAVEL) | Métriques maison (`src/analysis/metrics.py`) + odd-one-out maison. Les taux publiés ne sont comparables à aucun chiffre de la littérature faute d'un harnais commun (limite déjà notée au chapitre 4 du rapport) ; le coût réel d'adoption de `SAEBench` (temps de migration, quels chiffres actuels resteraient valides) n'a pas été chiffré. |
| Latent Terms | `x-tabdeveloping/latent_terms` (dépôt JAX tiers) | Réimplémentation ~40 lignes (`src/sae/retrieval/latent_terms.py`) — algorithme petit et self-contained (BM25 sur un vocabulaire latent SAE), le dépôt tiers en JAX aurait ajouté une dépendance lourde pour ce volume de logique. Les écarts protocolaires assumés (SAE entraîné en domaine, phrase-level plutôt que token-level) comptent plus que la question de la réimplémentation elle-même. |
| interp_embed | Dépôt officiel (Jiang, Sun et al.), submodule installé et peuplé localement | Cf. tableau principal ci-dessus pour le détail des écarts. Point encore ouvert : exécuter leur exemple de référence depuis `external/interp_embed/` pour vérifier que notre lecture du papier correspond bien à leur comportement réel — pas encore fait. |

## Modèles

| Modèle | Rôle | Taille |
|---|---|---|
| `google/gemma-3-12b-it` | Modèle cible de production (extraction hidden states + juge LLM) | 12B |
| `google/gemma-3-{4b,1b,270m}-it` | Profils alternatifs (`MODEL_SIZE`), 270m pour validation rapide locale | 4B/1B/270M |
| `google/gemma-scope-2-{12b,4b,1b,270m}-it` | SAE préentraînés GemmaScope-2 correspondants | — |
| `codefuse-ai/F2LLM-v2-{80M,160M,330M}` | Embeddings de phrase (Pipeline 2) | 80M-330M |
| `BAAI/bge-m3` | Pooling `[CLS]`, multilingue — branché via `src/config.py::LATENT_LABEL_EMB_MODEL` pour la similarité de labels (`select_latents_by_similarity`, `find_interesting_pairs`, §15.2/15.3 `RESULTS_TESTS.md`), après comparaison empirique où F2LLM (pooling dernier-token) donnait des résultats sans rapport sur des labels courts. | — |
