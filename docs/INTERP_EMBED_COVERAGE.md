# Inventaire interp_embed ↔ dépôt ↔ papier — Étape 0

Source de fidélité pour tout prompt/formule : `pdf/InterpretableSAE_Embeddings.pdf`
(Appendices A–M extraits verbatim dans `docs/PDF_APPENDICES_EXTRACT.md`, avec
traçabilité ligne/page). `external/interp_embed` est le code de référence des auteurs,
une implémentation **partielle** du papier — jamais l'inverse. Catégories A/B/C/D
définies dans la mission (remplacer / forcer la leur / double / garder la mienne).

**Corrections triviales faites au passage** (demandées dans la mission) :
- `src/sae/sae_shared.py:29` importe `interp_embed.sae.utils.get_reconstruction_error` :
  fonction absente de `external/interp_embed/interp_embed/sae/utils.py` (vérifié —
  le fichier ne contient que `process_device_config`, `ensure_loaded`,
  `try_to_load_feature_labels`, `get_goodfire_d_sae`, `get_goodfire_config_from_hf`,
  `goodfire_sae_loader`, `get_goodfire_config`, `store_activations_hook`). Import mort
  confirmé, à corriger à l'étape 1. **Conséquence plus grave que "juste un import mort" :**
  ce sous-import est dans le MÊME bloc `try` (lignes 28–32) que
  `from interp_embed import Dataset as InterpDataset` (ligne 30) — l'`ImportError`
  levée par `get_reconstruction_error` (ligne 29, exécutée en premier) est capturée par
  l'`except ImportError: InterpDataset = None` (ligne 31–32) AVANT que la ligne 30 ne
  s'exécute. Or `interp_embed/__init__.py:6` exporte bien `Dataset` (`from
  .dataset_analysis import Dataset`, `__all__ = ["Dataset"]`) — cet import réussirait
  seul. Résultat : `InterpDataset` reste `None` en permanence, MÊME MAINTENANT que le
  submodule est peuplé, à cause d'un import mort sans rapport qui masque silencieusement
  un import valide dans le même `try`. Le commentaire `sae_shared.py:20-22` ("interp_embed
  reste volontairement non peuplé... ce chemin ne prend effet que si le submodule est un
  jour initialisé") est lui-même obsolète — le submodule EST initialisé, mais le code ne
  peut pas le voir à cause de ce bug. Même famille de désynchronisation documentée en
  désaccord #6 (`tests/test_interp_embed_diff.py`).
- `examples/analysis.ipynb` cellule 0 (`from interp_embed.saes import GoodfireSAE`) :
  non vérifié directement (notebook non lu, hors périmètre lecture de code), mais le
  module réel est bien `interp_embed.sae.local_sae.GoodfireSAE` (classe dépréciée,
  confirmée présente, `DeprecationWarning` à l'import du module). Cohérent avec le
  signalement de la mission.

## Désaccords avec le classement hypothèse de la mission (à trancher avant l'étape 1)

1. **`odd_one_out_judge` vs protocole contrastif (App. C) — proposé en catégorie B,
   je recommande C.** `RESULTS_TESTS.md` §15.4 a déjà testé exactement ce remplacement
   (`scripts/contrastive_labeling_test.py`, mêmes 150 features) : bug de marqueurs sur
   négatifs trouvé et corrigé, bug d'écho de template (48/82 "récupérations" étaient un
   artefact), puis sur les 82 features non-interprétées seulement 58 labels distincts
   (45% de doublons), `confident=true` 150/150 dans les deux runs (signal inutilisable).
   Verdict déjà écrit noir sur blanc : *"le protocole contrastif n'est pas prêt à le
   remplacer... aucun changement appliqué au pipeline de production"*. Passer en défaut
   la version dont on a déjà la preuve empirique qu'elle produit un taux de récupération
   gonflé par la complaisance du juge serait une régression, pas une amélioration. Sauf
   qu'un gate anti-doublon soit ajouté d'abord, ce reste une alternative documentée
   (catégorie C, défaut `odd_one_out`), pas un nouveau défaut.
2. **`diff_features_multi` et `limit_feature_differences` ne sont PAS "À construire".**
   Les deux existent déjà, verbatim, dans `external/interp_embed/paper/diffing/sae_utils.py`
   (lignes 406–456 et 367–403 respectivement) — implémentations complètes, testables
   telles quelles. Le classement "aucun équivalent des deux côtés" de la mission est
   faux pour ces deux symboles ; ils passent en adaptateur (étape 1), pas en
   construction neuve (étape "À CONSTRUIRE").
3. **Le "harnais de vérification du diffing (App. K.1)" existe déjà en code**, pas
   seulement en spécification PDF : `paper/diffing/hypothesis_verifier.py::HypothesisVerifier`
   implémente le prompt App K.1 quasi mot pour mot (`verify_hypothesis_response`,
   comparé ligne à ligne à `docs/PDF_APPENDICES_EXTRACT.md` §K.1 — même structure
   REASONING/ANSWER, mêmes 7 instructions). Ce qui manque réellement (à construire) :
   le calcul des métriques (taux de vérification >1%, couverture) au-dessus de la
   matrice retournée par `verify_all` — la mission les avait correctement identifiées
   comme absentes, mais le classement du harnais lui-même comme "aucun équivalent" est
   à corriger : c'est un adaptateur, pas une construction neuve.
4. **App. K contient bien K.1–K.4** (contrairement au doute initial de la recherche par
   regex) : K.3 CLUSTERING et K.4 RETRIEVAL existent, courts (4-6 lignes chacun),
   `docs/PDF_APPENDICES_EXTRACT.md` lignes 846–888. Les prompts de jugement clustering
   et retrieval de l'étape "À CONSTRUIRE" ont donc une spécification texte complète.
5. **Deux implémentations divergentes du prompt de labellisation coexistent dans
   interp_embed lui-même** : `interp_embed/llm/prompts.py::build_labeling_prompt`
   (non-paired, "a clear, easily-understandable property") vs
   `paper/diffing/sae_utils.py::build_gpt4_labeling_prompt` (paired, "the most specific
   and concise property" — **identique mot pour mot** à l'Appendix C du PDF, vérifié
   ligne à ligne). Le PDF fait foi (règle mission) → `build_gpt4_labeling_prompt` de
   `sae_utils.py` est la version à utiliser pour tout remplacement de catégorie A, PAS
   `llm/prompts.py`. `llm/prompts.py::build_labeling_prompt` est un état antérieur/dérivé
   du même prompt, à ignorer.
6. **`tests/test_interp_embed_diff.py` a un docstring obsolète** : il affirme qu'
   interp_embed est *"volontairement non vendorisée... jamais installée en package"* —
   faux depuis que le submodule est cloné et peuplé (`external/interp_embed`, confirmé
   présent et lisible). À réécrire à l'étape 1/2 une fois les adaptateurs en place. Le
   test actuel tente `from interp_embed import diff_features as external_diff` qui
   échoue silencieusement (`except Exception: external_diff = None`) puisque le package
   n'est pas sur le path — la comparaison chiffrée n'a donc jamais tourné malgré le
   docstring qui laisse penser le contraire.

## A. `examples/functions.py`

| Symbole | Lignes | §/App. papier | Équivalent chez moi | Cat. | Écart papier↔dépôt |
|---|---|---|---|---|---|
| `diff_features(ds1, ds2, metric, min_coverage, max_coverage)` | 5–61 | §4.1 (dataset diffing, cas binaire simple) | `src/analysis/cooccurrence.py::corpus_diff_stats` (Fisher+BH, plus riche statistiquement) | C (déjà en place, `DIFF_STAT` flag à créer étape 3) | Aucun — dense/simple, cas de base uniquement (pas de p-value) |
| `calculate_npmi(X, Y=None)` | 63–103 | §4.2/App. E.1 | `src/analysis/cooccurrence.py::compute_npmi` (dense torch, symétrique seul) | **A** (test d'équivalence requis étape 2) | Aucun — mais gère le cas croisé X×Y (asymétrique) que la mienne ne gère pas |

## B. `paper/clustering/algorithms.py`

| Symbole | Lignes | §/App. | Équivalent chez moi | Cat. | Écart |
|---|---|---|---|---|---|
| `compute_clusters(dataset, n_clusters, active_features, top_n)` | 5–114 | App. F (Jaccard affinity + SpectralClustering `affinity="precomputed"`) | `src/sae/saev5.py::analyze_with_umap` + `src/analysis/cooccurrence.py::cluster_in_feature_space` (UMAP→HDBSCAN, confirmés tous deux présents et actifs) | **B** | Aucun — mais ne fait PAS la génération de mots-clés LLM ni l'union top-k=100/mot-clé (App. F, "à construire") ; `compute_clusters` prend un `active_features` déjà filtré en entrée |

## C. `paper/diffing/sae_utils.py`

| Symbole | Lignes | §/App. | Équivalent chez moi | Cat. | Écart |
|---|---|---|---|---|---|
| `FeatureLabelingRequest/Response`, `SingleSampleScoringRequest/Response` (pydantic) | 14–50 | App. C | Schémas JSON informels dans `judge.py` (pas de pydantic) | D (garder mon format léger, pas de gain pydantic ici) | — |
| `Hypothesis` (dataclass) | 52–59 | §4.1 | Dict brut dans mon pipeline | D | — |
| `extract_json_from_response` | 62–83 | — (utilitaire) | Parsing `re.search(r"\{.*?\}", ...)` inline dans `judge.py` | **A** candidate (utilitaire pur, aucun état) | — |
| `build_gpt4_labeling_prompt` (paired pos/neg) | 119–188 | **App. C — verbatim confirmé mot pour mot** | `judge.py::odd_one_out_judge` (prompt FR différent, gate préalable) | **A pour le texte du prompt EN**, garder le gate FR par-dessus en catégorie C (cf. désaccord #1) | `llm/prompts.py::build_labeling_prompt` (autre copie dans le même repo) diverge de cette version-ci ; **celle-ci est la fidèle au PDF** |
| `build_single_sample_prompt` (paired scoring) | 191–229 | App. C (scoring) | `judge.py` n'a pas de scoring pos/neg pairé — seulement le score contrastif LLM interne à `odd_one_out_judge` (étape rho_interp) | C (à construire un scorer séparé si App. D.3/K.1 l'exige) | — |
| `build_middle_out_batch_prompt` / `build_middle_out_final_prompt` | 232–322 | App. D.1/D.2 (résumé "middle-out" pour beaucoup de features) | Rien — mon pipeline traite les features directement, sans étape de résumé intermédiaire par lot | À CONSTRUIRE si le volume de features dépasse un budget contexte (cf. étape "modules neufs") | — |
| `build_hypotheses_prompt` | 325–365 | App. D.2 | `saev5.py::generate_llm_diff_hypothesis` (prompt FR, une seule hypothèse texte libre, pas de format JSON structuré ni budget "num_hypotheses") | C (garder la version FR simple en défaut interne, ajouter la version structurée pour comparaison chiffrée) | — |
| `limit_feature_differences` | 367–403 | App. D | Rien — pas de troncature symétrique pos/neg dans mon pipeline | **Adaptateur (pas "à construire")** — cf. désaccord #2 | — |
| `diff_features_multi` | 406–456 | §4.1 ("target vs max de fréquence parmi K autres corpus") | Rien | **Adaptateur (pas "à construire")** — cf. désaccord #2 | — |

## D. `paper/diffing/hypothesis_verifier.py`

| Symbole | Lignes | §/App. | Équivalent chez moi | Cat. | Écart |
|---|---|---|---|---|---|
| `class HypothesisVerifier` | 34–624 | **App. K.1 — prompt vérifié quasi-verbatim** (cf. désaccord #3) | Rien (aucun harnais de vérification équivalent) | **Adaptateur prioritaire**, pas "à construire" | Le prompt du dépôt dit *"the document exhibits..."* pour le texte à analyser dans le corps de la tâche mais garde *"RESPONSE TEXT TO ANALYZE"* comme en-tête — micro-incohérence interne au prompt déjà présente dans le PDF lui-même (vérifié dans `docs/PDF_APPENDICES_EXTRACT.md` — pas une erreur d'extraction) |
| `.verify_all` / `.verify_multiple_fields` | 237–333 | App. K.1 | — | Adaptateur | Async + `asyncio.Semaphore`, à réutiliser tel quel (pas de calcul, juste de l'orchestration API) |
| `.compute_multi_field_results` / `.save_multi_field_results` | 335–517 | — (reporting) | `RESULTS_TESTS.md` (format markdown numéroté, pas CSV) | C (garder mon format de reporting, réutiliser leur calcul de matrice) | — |

## E. `paper/diffing/generate_sae_hypotheses.py`

| Symbole | Lignes | §/App. | Équivalent chez moi | Cat. | Écart |
|---|---|---|---|---|---|
| `class HypothesisGenerator` | 65–767 | §4.1 (pipeline complet SAE→hypothèses) | `saev5.py` (fonctions séparées : `corpus_diff_stats` → `generate_llm_diff_hypothesis`, pas de classe orchestratrice) | C (pipeline entier différent, garder le mien, emprunter les prompts individuels en A) | `.label_feature`/`.score_single_sample` réutilisent les prompts de `sae_utils.py`, pas de nouveau texte |
| `.analyze_feature_differences` / `.generate_hypotheses` | 255–621 | §4.1, App. D | — | C | Bornes de concurrence (`max_concurrency=8` défaut) à vérifier avant tout run API à grande échelle |

## F. `paper/diffing/generate_baseline_hypotheses.py` + `baseline_utils.py`

| Symbole | Lignes | §/App. | Équivalent chez moi | Cat. | Écart |
|---|---|---|---|---|---|
| `class MultiModelDiffAnalyzer` (LLM-S/LLM-C baselines, App. D.1) | 60–639 | App. D.1 | Rien — mon pipeline n'a pas de baseline "LLM lit tout et résume" pour comparaison | À CONSTRUIRE si baseline LLM-only demandée en comparaison (pas listé explicitement dans les livrables actuels — à confirmer) | — |
| `baseline_utils.py::create_pairwise_analysis_prompt` etc. | 79–284 | App. D.1 | — | À CONSTRUIRE (idem, si baseline demandée) | — |

## G. `interp_embed/sae/*` (chargement modèle + SAE)

| Symbole | Lignes | §/App. | Équivalent chez moi | Cat. | Écart |
|---|---|---|---|---|---|
| `class BaseSAE(ABC)` (`base_sae.py`) | 6–75 | App. A (méthodes) | Rien d'équivalent en ABC — mon pipeline extrait les activations directement (`src/analysis/activations.py`) | **D pour l'architecture existante**, mais `HFHookSAE` (étape 1) doit hériter de celle-ci pour être exécutable telle quelle par `paper/*` sans modification | — |
| `class LocalSAE(BaseSAE)` | 13–120 (`local_sae.py`) | — | — | Incompatible tel quel (mission) — `transformer_lens.HookedTransformer` ne supporte pas Gemma-3 | Confirmé par lecture directe : `load_models` appelle `HookedTransformer.from_pretrained` sans option de fallback |
| `class GoodfireSAE(BaseSAE)` [DEPRECATED] | 123–258 (`local_sae.py`) | — | — | **Patron à copier pour `HFHookSAE`** (mission, étape 1) | `AutoModelForCausalLM` + `register_forward_hook` sur `model.model.layers[layer]` + troncature `layers[:layer+1]` — exactement le patron demandé, confirmé ligne à ligne (`local_sae.py:164-178`) |
| `class NeuronpediaApiSAE(ApiSAE)` | 55–235 (`neuronpedia_sae.py`) | — | — | D (hors périmètre — API distante, pas de poids locaux) | Utile seulement comme référence de `sae_id` Llama-3.1-8b/70b si jamais un jour comparé à Neuronpedia |
| `class ApiSAE(BaseSAE)` (`api_sae.py`) | 7–56 | — | — | D | `retry_api_with_backoff` réutilisable tel quel pour tout appel LLM juge/labellisation à grande échelle (backoff exponentiel + sémaphore) |
| `load_sae_from_metadata` (`load_sae.py`) | 3–17 | — | — | D (mon `FragmentDataset`/`HFHookSAE` n'a pas ce besoin de dispatch multi-backend) | — |
| `interp_embed/sae/utils.py::try_to_load_feature_labels` | 35–46 | — | Rien — pas de labels précalculés HF chez moi (P0 en a besoin) | **Adaptateur direct pour P0** (`nickjiang/feature_labels`, `goodfire/meta-llama/Llama-3.1-8B-Instruct.json`) | — |
| `get_goodfire_config` / `get_goodfire_config_from_hf` / `goodfire_sae_loader` | 49–163 (`utils.py`) | — | — | **Adaptateur direct pour P0** | `get_goodfire_d_sae` a un bug de logique déjà visible : `if "meta-llama/Llama-3.1-8B-Instruct":` est TOUJOURS vrai (chaîne non vide, pas de comparaison `==`) — la branche 70B (`elif`) n'est jamais atteinte. **Signalé, pas corrigé** (hors périmètre P0 qui n'utilise que le 8B ; le 70B est de toute façon abandonné par la mission) |

## H. `interp_embed/llm/*` + `utils/helpers.py` + `utils/data_models.py`

| Symbole | Lignes | §/App. | Équivalent chez moi | Cat. | Écart |
|---|---|---|---|---|---|
| `llm/prompts.py::build_scoring_prompt` | 3–56 | App. C (scoring non-paired) | — | Superseded — préférer `sae_utils.py::build_single_sample_prompt` (paired, confirmé plus proche du texte App. C) | — |
| `llm/prompts.py::build_labeling_prompt` | 58–100 | App. C | — | **Ne pas utiliser** — cf. désaccord #5, diverge du PDF | *"a clear, easily-understandable property"* (dépôt) vs *"the most specific and concise property"* (PDF, confirmé aussi dans `sae_utils.py::build_gpt4_labeling_prompt`) |
| `llm/utils.py::get_llm_client/call_async_llm/extract_json_from_response` | 7–98 | — | Mon pipeline appelle le juge local (Gemma-3 via `transformers.generate`, pas d'API OpenRouter/OpenAI) | D pour mon usage (juge local), **A candidate** si un jour un juge API externe est ajouté en comparaison | — |
| `utils/helpers.py::CHAT_TEMPLATE_END_POSITION_TOKENS/ACTIVATIONS` (30/29) | 7–8 | — | Rien — mon extraction ne tronque pas par position fixe | **Ne PAS copier tel quel** (mission) — constantes Llama-3-spécifiques, à recalculer dynamiquement pour Gemma-3 si `READER_CHAT_TEMPLATE=assistant` est activé (étape 3) | Confirmé : aucune détection dynamique dans le dépôt, valeurs codées en dur |
| `utils/helpers.py::highlight_activations_as_string` | 68–81 | App. C (format `<<>>` "full span") | `judge.py::extract_causal_context` (fenêtre causale gauche, mot complet) | C (déjà correctement classé par la mission) | La leur marque TOUT span contigu où `activation > 0`, y compris après le token cible — la mienne s'arrête au mot cible (fenêtre causale gauche uniquement) |
| `utils/data_models.py::SingleSampleScoringResponse/FeatureLabelResponse` | 3–21 | App. C | — | D (doublon de `sae_utils.py`, pas de valeur ajoutée à adapter séparément) | — |

## I. `interp_embed/dataset_analysis.py`

| Symbole | Lignes | §/App. | Équivalent chez moi | Cat. | Écart |
|---|---|---|---|---|---|
| `class Dataset.latents(aggregation_method)` — {max, mean, sum, binarize, count, all} | 218–281 | — (structure de données de référence) | `src/storage/fragment_store.py::doc_maxpool` (max seul) | **Spécification pour `FragmentDataset`** (étape 1) — je n'expose que max | `count` = nnz par colonne (ligne 557 `DatasetRow.latents`, `all_activations.getnnz(axis=0)`) — confirmé non dérivable d'un vecteur déjà max-poolé, doit être calculé depuis le CSR brut |
| `Dataset.top_documents_for_feature(select_top, include_active/nonactive_samples)` | 283–302 | App. C | `judge.py::build_feature_examples_with_control` (logique similaire : top par magnitude + négatif à quantile bas) | C (déjà proche, garder la mienne en défaut) | La leur échantillonne k parmi `argpartition` direct sur toute la colonne dense ; la mienne itère les fragments triés et déduplique sur le mot-cible (garde-fou absent chez eux) |
| `Dataset.score_feature(feature, label, k=10, superset=3k)` | 304–358 | App. C, D.3 | Rien d'équivalent — mon pipeline n'a pas de scoring contrastif séparé du gate odd-one-out | **Spec pour le "protocole contrastif" (désaccord #1)** — confirme le détail exact : k=10 pos + 10 neg, superset 3k, négatifs `include_nonactive_samples=True, include_active_samples=False` (donc **activation nulle stricte pour la borne basse du superset**, pas juste "sous un quantile") | Confirme la ligne de la mission "négatifs à activation strictement nulle, échantillonnés dans un superset 3k" — vérifié exact dans le code, pas une supposition |
| `DatasetRow.token_activations(feature, left_marker, right_marker)` | 565–574 | App. C | `judge.py::extract_causal_context` | C | Utilise `highlight_activations_as_string` (full-span), cf. ligne H ci-dessus |
| `Dataset.save_to_file` / `load_from_file(resume=True)` | 145–209 | — | `src/storage/fragment_store.py` (CSR fp32, par-document, `.pt`) | **D pour le format** (le mien nécessaire pour fp32/massive activations Gemma-3), **A pour la sémantique de reprise** (mission) | Leur `resume` reprend un seul fichier pickle monolithique (tout le dataset) ; le mien reprend déjà par fragment individuel (`fragment_exists`) — la sémantique à récupérer est le *pattern* (quels indices manquent), pas le format de fichier |

## Résumé quantitatif (pour `docs/DELTA_CODE.md`, étape 5)

- Catégorie A confirmée (remplacement, test d'équivalence requis avant tout code) :
  `calculate_npmi`, prompts `build_gpt4_labeling_prompt`/`build_single_sample_prompt`
  (texte seul), `extract_json_from_response`.
- Catégorie B confirmée : `compute_clusters` (Jaccard+Spectral) contre
  `analyze_with_umap`/`cluster_in_feature_space`.
- Catégorie C (double, flag env) : `corpus_diff_stats` vs `diff_features`,
  `TOKEN_MASK_MODE`, `READER_CHAT_TEMPLATE`, `JUDGE_CONTEXT`, négatifs quantile vs
  strict-zéro, **et `JUDGE_PROTOCOL` (odd_one_out vs contrastif) — révisé depuis B**.
  `DatasetRow.latents()` : 5 agrégations à exposer dans `FragmentDataset` derrière la
  même interface `Dataset`-compatible.
- Adaptateurs prioritaires (ni A/B au sens strict, ni "à construire" — code déjà
  écrit chez eux, juste à rendre exécutable sur mes données) : `HypothesisVerifier`
  (App. K.1), `diff_features_multi`, `limit_feature_differences`, patron `GoodfireSAE`
  pour `HFHookSAE`, chargement labels HF (`try_to_load_feature_labels` +
  `get_goodfire_config*`).
- Catégorie D confirmée : `src/storage/fragment_store.py` (format), `scatter_maxpool`/
  `maxpool_sae_docs`, `src/analysis/stats.py` (aucun équivalent côté interp_embed —
  vérifié, aucun test statistique dans tout le dépôt cloné), scripts de robustesse du
  juge (`judge_robustness_check.py` etc., aucun équivalent).
- Réellement "à construire" (pas de code existant ni chez moi ni chez eux, seulement
  une spec papier) : agrégation `mean`/`sum`/`count` sur CSR (moi : spec connue, code à
  écrire), prompts de génération de mots-clés clustering (App. F.1, prompt non trouvé
  verbatim dans l'extraction — à vérifier dans `docs/PDF_APPENDICES_EXTRACT.md` §F),
  métriques de retrieval (MAP/MP@50/MP@10/RBO — formules non retrouvées explicitement
  dans l'extraction, cf. points d'incertitude du fichier), protocole App. I complet
  (F1 latent-classifieur, aucune implémentation ni chez moi ni chez eux), baseline
  LLM-S/LLM-C (`MultiModelDiffAnalyzer` existe mais sert un usage différent — comparer
  sans SAE, pas une brique de mon pipeline SAE).

## Limites de cet inventaire

- `paper/diffing/generate_sae_hypotheses.py` et `generate_baseline_hypotheses.py` sont
  documentés par leurs signatures publiques (grep `def`/`class`) et un survol de leurs
  prompts déjà lus indirectement via `sae_utils.py`/`baseline_utils.py` — pas relus
  ligne à ligne en entier (1406 lignes à eux deux). Si l'étape 1 les utilise comme
  adaptateurs directs, une relecture complète est nécessaire avant d'écrire le test
  d'équivalence.
- `docs/PDF_APPENDICES_EXTRACT.md` liste 13 points d'incertitude propres (formules RBO/
  MAP/RRF non trouvées explicitement dans le texte extrait, mapping numérique {1,0.5,0}
  de D.3 introuvable dans les Appendices — probablement dans le corps principal, hors
  périmètre de cette extraction). À vérifier manuellement sur le PDF avant d'implémenter
  App. G/App. D.3.
- `examples/analysis.ipynb` non lu directement (mission le signale déjà comme cassé,
  pas utilisé comme spécification).
