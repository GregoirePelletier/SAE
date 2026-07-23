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

# État actuel (mis à jour après audit + refonte + validation locale)

## Refactor : statut

Le refactor structurel décrit plus bas ("Architecture cible") a été **partiellement**
réalisé : `src/sae/`, `src/analysis/`, `src/data/`, `src/storage/` existent et séparent
correctement les responsabilités. Il manque encore `src/models/` (gemma.py,
embedding_models.py — actuellement fondus dans `saev5.py`/`phrase_sae.py`), et
`src/sae/saev5.py` reste volumineux (~1470 lignes, orchestration des deux pipelines) —
factorisation non terminée. `evaluation/` et `visualization/dashboard.py` (Streamlit)
n'existent pas.

## Cible modèle/SAE

**Gemma-3-12B-it** + GemmaScope-2 (`layer_24_width_16k_l0_medium`, residual stream) est
la cible principale (`MODEL_SIZE=12b`, défaut de `src/config.py`). Largeur **16k** choisie
plutôt que la 262k utilisée historiquement par `slurm/pipeline_runs/run_sae.slurm` : la couverture des labels
Neuronpedia sur `24-gemmascope-2-res-262k` est faible (~10 000 features labellisées sur
262 144, constaté manuellement) ; 16k est bien plus dense en proportion.

**Gemma-3-270M-it** + GemmaScope-2 (`layer_12_width_65k_l0_medium`) reste disponible
comme profil de **validation rapide** (`MODEL_SIZE=270m`) — c'est ce qui a servi à valider
tout le pipeline de bout en bout sur une machine à 6 Go VRAM avant de lancer un run complet
sur 12b. Détails et résultats de cette validation ci-dessous.

## Sous-modules externes

- `SAELens` est initialisé sous `external/sae-lens` (submodule). Le code utilise en
  pratique le **package pip** `sae-lens>=6.0.0` (pas le submodule directement) ; le
  submodule sert de référence pour la comparaison d'implémentation (règle n°2).
- `GemmaScope` n'est pas cloné comme sous-module ; les poids SAE sont téléchargés depuis
  HuggingFace Hub via `download_sae.py`.
- `interp_embed` n'est pas téléchargé localement (inspiration seulement, non installé).

---

# Audit et corrections apportées (session d'audit + validation locale)

## Bugs corrigés

### Chargement SAE (`src/sae/gemma_scope_loader.py`)
- `hook_layer` était figé à 24 en dur (biais historique 12b/layer-24) → dérivé
  dynamiquement du `hook_name` résolu (supporte les deux conventions observées :
  `blocks.N.hook_resid_post` et `model.layers.N.output`).
- Le fichier de config attendu (`cfg.json`) a en réalité été renommé `config.json` par
  GemmaScope-2 → les deux noms sont supportés.
- `d_in` était supposé 1024 pour 270m (specs génériques publiques) → **640**, confirmé
  empiriquement sur les poids réellement téléchargés.

### Labellisation Neuronpedia (`src/sae/neuronpedia_labels.py`)
- L'ancienne route REST `/api/explanation/export` était cassée (non fiable / retours
  vides). Remplacée par le téléchargement direct des lots `.jsonl.gz` publics sur le
  bucket S3 `neuronpedia-datasets` (approche validée manuellement par l'utilisateur avant
  la généralisation). Format de cache inchangé, aucun autre appelant à modifier.
- **Validé empiriquement** : 64 377 labels récupérés pour `gemma-3-270m-it/12-gemmascope-2-res-65k`
  (~98% de couverture sur 65 536 features) ; ~10 000/262 144 pour
  `gemma-3-12b-it/24-gemmascope-2-res-262k` (d'où le choix de 16k pour 12b, cf. ci-dessus).

### Dtype (bf16/fp16) — le bug le plus significatif trouvé par exécution réelle
Gemma-3 a des **activations massives** documentées dans le residual stream (outliers
d'amplitude ~1e5). En **fp16** (max représentable ~65504), ces outliers overflowent
silencieusement vers `inf`/`nan`, qui contaminent tout l'entraînement de `ExtendedSAE`
(`Loss=nan` dès la première epoch — observé sur un run 270m complet avant correction).
`DTYPE` est donc passé à **bf16 par défaut**, y compris en local (plage d'exposant
identique à fp32, jusqu'à ~3e38 — pas d'overflow), au prix d'un calcul plus lent sur GPU
sans tensor cores bf16 natifs (Turing).

Une fois bf16 activé, un second bug est apparu : `FrozenCoreResidualSAE`/`ExtendedSAE`
était castée en bloc via `.to(TORCH_DTYPE)` après construction dans `saev5.py`, alors que
la branche "extra" (`W_dec_extra`, `W_enc_extra`, `b_enc_extra`, `threshold`,
`input_scale`) est conçue pour rester **fp32 partout** (`.float()` systématique dans
`frozen_core.py`). Le cast module-wide la basculait en bf16, cassant le backward
(`RuntimeError: Found dtype BFloat16 but expected Float`). Corrigé en retirant le cast
module-wide (seul `core_sae`, déjà casté séparément, doit être en `TORCH_DTYPE`) et en
castant `residual` en fp32 immédiatement après calcul dans `forward()` (la promotion
implicite fp32/bf16 dans `mse_loss`/`var_residual` fonctionne en forward mais pas de
façon fiable pour le gradient — confirmé par isolation empirique d'un cas minimal).

### Robustesse (petits corpus)
- `analyze_with_umap` (UMAP + HDBSCAN) plantait sur 0 feature active (`ValueError` sklearn)
  et sur l'initialisation spectrale pour de très petits corpus (`TypeError` scipy eigsh,
  `k >= N`) → gardes ajoutées : dégradation propre (features actives = 0) et bascule
  `init="random"` pour `N_DOCS < 15`.
- `corpus_diff_stats` plantait (`KeyError`) quand aucune feature n'était active dans les
  deux sous-groupes comparés → retourne un `DataFrame` vide mais bien formé.
- `scripts/baseline_gemmascope.py` : même classe de crash sur diff vide, gardée de la
  même façon.

### Autres corrections
- `download_sae.py` était figé sur `release="gemma-scope-2-4b-it-res"` (mauvais modèle
  ET mauvais nom de repo) → réécrit pour lire `src/config.py` (`MODEL_ID`/`RELEASE_ID`/
  `SAE_ID`), télécharge modèle + SAE via `snapshot_download` (repo HF réel, sans le
  suffixe `-res` hérité de GemmaScope v1 — vérifié empiriquement via l'API HF
  authentifiée pour 12b/4b/1b/270m).
- Org HuggingFace de F2LLM corrigée : `Alibaba-NLP/F2LLM-v2-80M` (404, org inexistante)
  → `codefuse-ai/F2LLM-v2-80M`, corrigée dans `src/config.py` et
  `src/sae/compare/pipeline.py`.
- `src/config.py` : `SAE_ID` n'était overridable par env que pour `MODEL_SIZE=12b` (bug) ;
  duplication du dispatch par taille de modèle entre `config.py` et `saev5.py`
  (`saev5.py` importe maintenant tout depuis `config.py`, source unique de vérité).
- `src/sae/sae_shared.py` : le `sys.path.insert` vers `external/interp_embed` pointait
  vers `src/sae/external/interp_embed` (inexistant) au lieu de la racine du dépôt.
- `src/sae/compare/pipeline.py` : `from . import visualization as viz` cassé (le module
  est dans `src/analysis/`, pas dans `src/sae/compare/`) — import relatif corrigé.
- `src/sae/batch.py::batch_topk_encode` : fonction dépréciée confirmée morte (aucun appel
  dans tout le dépôt) → supprimée avec son import inutilisé dans `saev5.py`.
- `tests/test_interp_embed_diff.py` : importait `diff_features` depuis
  `src/analysis/metrics.py`, symbole inexistant (échec de collection pytest) → réécrit
  pour exercer `corpus_diff_stats` (l'équivalent projet réellement maintenu), avec un
  second test optionnel comparant à `interp_embed` si jamais installé.
- `scripts/retrieval_demo.py` (nouveau) : câble `src/sae/retrieval/latent_terms.py`
  (jamais appelé nulle part ailleurs dans le dépôt) à un point d'entrée testable, avec un
  corpus de substitution public FineWeb-2 en l'absence de `Mails.tsv`.
- Chemins cluster (`/home/h21486/SAE/...`) remplacés par des chemins locaux relatifs ou
  des repo IDs HuggingFace directs (`MODEL_ID = "google/gemma-3-12b-it"`, résolu depuis
  le cache HF après `download_sae.py`) — portable entre machines, override par env
  toujours possible pour retrouver le comportement cluster.
- Environnement réseau (`HF_HUB_OFFLINE`, désactivation SSL) rendu conditionnel
  (`CLUSTER_OFFLINE_MODE`, défaut `0`) au lieu d'être forcé inconditionnellement à
  l'import de `saev5.py` — sinon aucun téléchargement initial n'est possible hors cluster.

## Validation empirique (Gemma-3-270M-it, machine locale 6 Go VRAM)

- `pytest tests/` : 8/8 passants.
- `download_sae.py` : modèle + SAE téléchargés et chargés avec les bonnes formes
  (`d_in=640, d_sae=65536`).
- Pipeline 1 exécuté de bout en bout (corpus réduit puis élargi, FineWeb-2/Wikipedia) :
  features **core** GemmaScope réellement interprétables obtenues, ex. `F65535 →
  "standards, specifications, variations"`, `F21853 → "save or process data"`.
- **Limite observée** : les features d'**extension** `ExtendedSAE` (juge LLM local, non
  couvertes par Neuronpedia) restent `dead_feature` avec un budget d'entraînement modeste
  (15-200k tokens sur un corpus de quelques centaines de documents) — insuffisant pour
  que ces features s'activent. À revalider avec un corpus/budget plus large sur 12b.
  Le juge LLM lui-même est probablement moins fiable sur un modèle 270M que sur 12B pour
  la tâche d'auto-interprétation odd-one-out.
- Pipeline 2 (F2LLM + `PhraseLevelSAE`) : entraînement propre (NMSE 0.65→0.06 sur un
  corpus de ~4000 phrases), UMAP/clustering fonctionnels.
- `scripts/retrieval_demo.py` : `PhraseLevelSAE` entraîné (NMSE 0.70→0.35 sur un petit
  corpus), recherche BM25 fonctionnelle sur le vocabulaire latent.
- `src/sae/compare/pipeline.py` : import et fonctions (`run_analysis`, `run_compare`)
  vérifiés propres après correction.

---

## Validation à l'échelle complète (Gemma-3-12B-it, corpus réel EDF) — session v10

Confirme et RÉSOUT la limite ci-dessus ("features d'extension restent `dead_feature`
avec un budget d'entraînement modeste") : ce n'était pas (seulement) un problème de
budget, mais surtout de **contenu du corpus**. Détail complet du diagnostic et des 3
runs de validation dans `RESULTS_TESTS.md` §12. Résumé :

- **Diagnostic** : `train_texts` (corpus servant à échantillonner le réservoir de
  résidus ET à entraîner `ExtendedSAE`) était bâti uniquement depuis
  energy/sports/support (FineWeb-2/Wikipedia) — les emails n'entraient JAMAIS dans le
  train, seulement en post-hoc pour la visualisation UMAP. Sur le dernier run complet
  avant fix (`results_v9_full`, job 39531) : 0/10 features d'extension `dead_feature`
  (donc pas un problème de volume brut) mais seulement 2/10 (20%) passaient le test
  odd-one-out — les "exemples positifs" présentés au juge étaient des extraits
  Wikipedia sans rapport entre eux (aucun concept commun à trouver).
- **Fix** : `build_email_train_test_corpus()` (nouveau, `src/data/preparation.py`) —
  corpus principal = mails réels + variantes augmentées acceptées, split group-aware
  par mail d'origine (`parent_id`, empêche toute fuite train/test). energy/sports/
  support réduit et repositionné en corpus secondaire post-hoc (démonstration de
  diffing cross-domaine uniquement, plus jamais utilisé pour l'entraînement).
- **Résultat (3 runs, `N_FEATURES_TO_LABEL=150` pour la puissance statistique)** :
  taux d'interprétabilité odd-one-out **20% → ~41-45%** (0 `dead_feature` dans les 3
  cas). Labels obtenus directement alignés métier : `Réclamations Clients`, `Litiges
  Factures`, `Résiliation Énergie`, `Menace Résiliation`, `Demande Urgente`.
- **Ablation volume** (`N_TOKENS_EXTRA_TRAIN` ∈ {100k, 500k, 2M}, corpus identique) :
  40,7% / 45,3% / 44,7% — **aucun écart significatif** (écart-type binomial attendu
  ≈4,1 pts sur n=150). Conclusion : une fois le domaine corrigé, le volume de tokens
  n'est déjà plus limitant à 100k, et le porter à 2M n'apporte rien de mesurable. Le
  goulot d'étranglement observé était bien un problème de **contenu du corpus**, pas
  de **volume brut**.
- **Effet de bord positif** : la sonde de classification sur les axes email (14
  classes, nouvelle métrique `clf_acc_email_axes`) donne acc_SAE=93,5% (P1) / 79,3%
  (P2) — les codes latents séparent très bien émotion/urgence/registre/original,
  résultat encourageant pour les cas d'usage détection d'urgence/intention visés par
  le projet (cf. section Objectif).
- **Reste non résolu** : ~55-59% des features d'extension restent non interprétables
  même corpus corrigé — piste à explorer : robustesse du protocole de jugement
  (génération greedy unique, pas de vote/ensemble) plutôt que le corpus ou le volume.

---

# Problèmes connus restants

## Duplication

Une partie du code existe déjà dans SAELens / interp_embed (règle n°1 — comparaison
documentée requise, pas encore faite systématiquement). `src/analysis/metrics.py`
réimplémente délibérément FVE/NMSE/L0 "en alignement strict avec SAELens et interp_embed"
(justifié : nécessaire pour scorer à la fois un `SAE` natif sae-lens et le
`FrozenCoreResidualSAE`/`ExtendedSAE` custom du projet).

## Architecture monolithique

`saev5.py` reste volumineux malgré l'extraction de `frozen_core.py`, `phrase_sae.py`,
`judge.py`, `batch.py`, `neuronpedia_labels.py`, `gemma_scope_loader.py`. Factorisation
supplémentaire possible vers `src/models/` (chargement Gemma-3 / F2LLM) et
`src/sae/training.py` / `src/sae/extraction.py` (architecture cible).

## Sparse matrices

`src/storage/fragment_store.py` a migré du SciPy CSR dense vers un **CSR fait-maison en
tenseurs torch** (pas la cible littérale COO/top-k du document original, mais atteint le
même objectif mémoire). `src/analysis/cooccurrence.py` et
`src/sae/retrieval/latent_terms.py` utilisent encore `scipy.sparse` (TF-IDF/UMAP/BM25 —
attendu par ces bibliothèques en aval, pas un problème en soi).

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

**Mise à jour** : bf16 est maintenant le défaut **partout** (y compris local, cf. section
Bugs corrigés — fp16 cause des overflows sur les activations massives de Gemma-3). La
branche "extra" de `FrozenCoreResidualSAE`/`ExtendedSAE` reste volontairement en fp32
(design initial confirmé correct par le débogage empirique) ; seul `core_sae` (poids
GemmaScope) est en bf16.

---

## 5

Préserver les résultats actuels.

Toute refactorisation doit passer les tests de non-régression.

**Statut** : `pytest tests/` = 8/8 passants après toutes les corrections ci-dessus.

---

# Prochaines étapes

1. ~~**Run complet sur Gemma-3-12B-it**~~ **FAIT** (session v10, cf. section
   "Validation à l'échelle complète" ci-dessus + `RESULTS_TESTS.md` §12) : features
   d'extension non `dead_feature`, taux d'interprétabilité 20%→~45% après correction
   du corpus.
2. ~~**`run_augmentation.py` puis `baseline_gemmascope.py`** sur la machine disposant du
   vrai `Mails.tsv`~~ **FAIT** (`RESULTS_TESTS.md` §0 et §6-10 : corpus complet
   augmenté — 45 240 générations, 39 949 acceptées — et baseline exécutés avec succès).
3. **Piste ouverte par le diagnostic v10** : ~55-59% des features d'extension restent
   non interprétables malgré le corpus corrigé, à volume de tokens non limitant —
   investiguer la robustesse du protocole de jugement lui-même (`odd_one_out_judge`) :
   vote majoritaire sur plusieurs générations plutôt qu'une décision greedy unique,
   qualité du contrôle négatif, prompt/format de la question.
4. Poursuivre la factorisation de `saev5.py` vers l'architecture cible (`src/models/`,
   séparation training/extraction).
5. Dashboard Streamlit (fonctionnalité future, non commencée).
6. Comparaison documentée avec SAELens (règle n°2), pas encore faite systématiquement.

---

# Fonctionnalités futures

## Retrieval interprétable

Recherche :

"mails urgents"

"mise en service"

"panne locale"

"facturation"

à partir des activations SAE.

*Amorcé* : `src/sae/retrieval/latent_terms.py` (BM25 sur features SAE) + démo
`scripts/retrieval_demo.py`, validé sur corpus de substitution public.

---

## Diffing de corpus

Comparer :

région A vs région B

année N vs N+1

campagne avant/après

*Amorcé* : `src/analysis/cooccurrence.py::corpus_diff_stats` (Fisher exact + BH),
utilisé par `saev5.py` (énergie vs sport) et `scripts/baseline_gemmascope.py` (originaux
vs augmentés par axe de perturbation).

---

## Corrélations

Calcul NPMI.

Identifier :

urgence ↔ panne

colère ↔ facturation

etc.

*Amorcé* : `src/analysis/cooccurrence.py::compute_npmi` + `cooccurrence_graph`
(communautés Louvain).

---

## Dashboard

Interface Streamlit.

Visualisation :

- UMAP
- features activées
- exemples positifs
- exemples négatifs
- recherche

*Non commencé.* Actuellement : exports HTML Plotly statiques autonomes
(`src/analysis/visualization.py`), pas d'interface serveur.

---

# Documentation

Maintenir automatiquement :

README.md

docs/architecture.md

docs/experiments.md

docs/references.md

*Statut* : `README.md` à jour (description de tous les scripts, installation, choix de
config). `docs/architecture.md`, `docs/experiments.md`, `docs/references.md` créés
(session v10).

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

*Statut* : `report/` créé (session v10) avec état de l'art, architecture, expériences/
résultats et limites/perspectives — matériel de base pour le rapport de stage. Pas
encore de section comparaison chiffrée avec interp_embed/SAE Boost (cf.
`docs/references.md` pour le détail des comparaisons faites/manquantes).
