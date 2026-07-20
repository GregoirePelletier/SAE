# Sparse Autoencoders (SAE) for Interpretable Text Analysis

Analyse interprétable de mails clients EDF (et de corpus publics de substitution) via
Sparse Autoencoders sur les hidden states de Gemma-3, avec labellisation des features
par GemmaScope-2 / Neuronpedia et par un juge LLM local.

Deux pipelines :

- **Pipeline 1** : Gemma-3 (hidden states, couche `LAYER`) → SAE GemmaScope-2 préentraîné
  (+ extension `FrozenCoreResidualSAE`/`ExtendedSAE` optionnelle) → max-pool documentaire.
- **Pipeline 2** : F2LLM-v2 (embeddings de phrases) → `PhraseLevelSAE` entraîné from-scratch
  (BatchTopK + AuxK) → max-pool documentaire.

---

## État actuel (résumé — détail complet dans `Context.md` et `RESULTS_TESTS.md`)

Le pipeline a été audité et rendu fonctionnel de bout en bout : bugs de chargement SAE
corrigés, labellisation Neuronpedia réparée (ancienne route REST cassée → téléchargement
S3 direct, cache local canonique réutilisé sans jamais retenter d'appel réseau),
plusieurs bugs de dtype (fp16/bf16) découverts et corrigés par exécution réelle, code
mort supprimé, modules orphelins raccordés. Validé à l'échelle complète sur
**Gemma-3-12B-it** (cible de production) sur le corpus réel EDF (mails + variantes
augmentées).

**Corpus d'entraînement (v10)** : le SAE d'extension (`ExtendedSAE`) et le
`PhraseLevelSAE` s'entraînent désormais sur les **mails réels + variantes augmentées**
(`local_data/emails/`), qui dominent le train (~41k/2.2k docs train/test). Le corpus
generic energy/sports/support (FineWeb-2/Wikipedia) est réduit à un petit corpus
secondaire, encodé post-hoc uniquement pour la démonstration de diffing cross-domaine
préexistante — il ne participe plus à l'entraînement. Ce changement a fait passer le
taux d'interprétabilité des features d'extension (protocole odd-one-out) de **20%
(2/10, corpus générique) à ~41-45% (n=150, corpus emails)** ; des runs d'ablation à
corpus identique (100k/500k/2M tokens) montrent que le volume d'entraînement n'a lui
quasiment aucun effet une fois le domaine corrigé — cf. `RESULTS_TESTS.md` §12 pour le
détail complet du diagnostic et des 3 runs de validation.

---

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate   |   Linux/Mac : source .venv/bin/activate
pip install -e .
```

Dépendances principales : `torch`, `transformers`, `sae-lens`, `transformer-lens`,
`datasets`, `scipy`, `scikit-learn`, `networkx`, `hdbscan`, `plotly`, `umap-learn`,
`statsmodels`. Voir `pyproject.toml` pour le détail et les contraintes de version.

### Accès HuggingFace (obligatoire)

`google/gemma-3-*-it` et `google/gemma-scope-2-*-it` sont des repos **gated** :

1. Créer un compte sur [huggingface.co](https://huggingface.co).
2. Accepter la licence Gemma sur la page du modèle ciblé (ex.
   [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)).
3. Générer un token sur <https://huggingface.co/settings/tokens>.
4. Copier `.env.example` en `.env` et y placer `HF_TOKEN=hf_...` (fichier gitignored).

### Windows uniquement

- Si le nom d'utilisateur Windows contient un caractère accentué ou que le chemin `HOME`
  est long, le cache HuggingFace peut dépasser `MAX_PATH` (260 caractères) sur les
  fichiers de verrou → `OSError [Errno 22]`. Solution : `HF_HOME=C:\hfcache` (chemin
  court, sans accent).
- Si les liens symboliques ne sont pas autorisés (mode développeur désactivé) :
  `HF_HUB_DISABLE_SYMLINKS=1`.
- Forcer l'UTF-8 pour l'affichage console (le code utilise des caractères comme `→`) :
  `PYTHONUTF8=1` et `PYTHONIOENCODING=utf-8`.

---

## Démarrage rapide

```bash
# 1. Télécharger le modèle + le SAE (cible par défaut : 12b)
python download_sae.py

# 2. Récupérer les labels Neuronpedia (optionnel mais recommandé — features interprétables)
#    Cache canonique partagé par TOUS les runs (src/config.py::NEURONPEDIA_LABELS_PATH) --
#    une fois présent, plus jamais d'appel réseau, quel que soit SAVE_DIR utilisé ensuite.
python -c "from src.sae.neuronpedia_labels import fetch_neuronpedia_labels; \
  fetch_neuronpedia_labels(model_id='gemma-3-12b-it', layer=24, width='16k', \
  cache_path='local_data/neuronpedia_labels/neuronpedia_labels_24-gemmascope-2-res-16k.json')"

# 3. Lancer la pipeline complète (PYTHONPATH=. requis, cf. ci-dessous)
PYTHONPATH=. python src/sae/saev5.py
```

**Important** : `saev5.py` doit être lancé depuis la **racine du dépôt** (pas depuis
`src/sae/`), avec `PYTHONPATH=.` — c'est ce que fait `run_sae.slurm`. Le script mélange
volontairement imports absolus (`from src.analysis...`) et imports relatifs "à plat"
(`from sae_shared import ...`, résolus car le dossier du script est ajouté
automatiquement à `sys.path`).

---

## Configuration (`src/config.py`)

Source unique de vérité pour toute la pipeline — toutes les valeurs sont surchargeables
par variable d'environnement. Voir `.env.example` pour un jeu de valeurs prêtes à copier,
avec un profil `12b` (principal) et un profil `270m` (validation rapide, commenté).

| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_SIZE` | `12b` | `12b` / `4b` / `1b` / `270m` — sélectionne modèle + SAE via `_PRESETS` |
| `MODEL_ID` | dérivé du preset | Repo HF du modèle (override direct possible) |
| `SAE_ID` | dérivé du preset | Sous-dossier GemmaScope (`layer_X_width_Y_l0_Z`) |
| `DTYPE` | `bf16` | **bf16 obligatoire** sur Gemma-3 (cf. section Bugs corrigés) |
| `LAYER` | dérivé du preset | Couche du residual stream extraite |
| `USE_FROZEN_CORE` | `1` | Active l'extension `ExtendedSAE` (Pipeline 1) |
| `D_EXTRA` / `K_EXTRA` | `1024` / `32` | Dimension / sparsité de l'extension |
| `D_SAE` / `K_SPARSE` | `8192` / `16` | Dimension / sparsité du `PhraseLevelSAE` (Pipeline 2) |
| `EMB_MODEL` | `codefuse-ai/F2LLM-v2-80M` | Modèle d'embeddings phrase (Pipeline 2) |
| `SAVE_DIR` | `./results/` | Racine des sorties (résultats + `cache/`) |
| `LOCAL_MAILS_PATH` | `./local_data/emails/Mails.tsv` | Corpus EDF réel (absent hors machine de calcul) |
| `LOCAL_AUGMENTED_MAILS_PATH` | `./local_data/emails/augmented_mails.jsonl` | Variantes augmentées acceptées (produites par `scripts/run_augmentation.py`) |
| `NEURONPEDIA_LABELS_PATH` | `./local_data/neuronpedia_labels/neuronpedia_labels_{layer}-gemmascope-2-res-{width}.json` | Cache labels Neuronpedia, **partagé entre tous les runs** (indépendant de `SAVE_DIR`) |
| `EMAIL_TEST_SPLIT` | `0.05` | Fraction des mails (et de leurs variantes) réservée au test, split group-aware par mail d'origine |
| `MAX_AUGMENTED_PER_MAIL` | `13` | Nb max de variantes augmentées conservées par mail réel |
| `CLUSTER_OFFLINE_MODE` | `0` | `1` = reproduit l'environnement cluster (SSL désactivé, HF offline) |
| `HF_TOKEN` | — | Token HF pour les repos gated |

### Choix du SAE GemmaScope-2 (largeur)

Le preset `12b` cible **`layer_24_width_16k_l0_medium`** (16 384 features), pas la
largeur 262k utilisée historiquement par `run_sae.slurm` : la couverture des labels
Neuronpedia sur `24-gemmascope-2-res-262k` est faible (~10 000 features labellisées sur
262 144, constaté manuellement), alors que 16k a une couverture bien plus dense en
proportion (comparable à la couverture ~98% mesurée empiriquement sur `270m`/65k lors de
la validation locale). `run_sae.slurm` a été mis à jour en conséquence.

---

## Scripts — description et usage

### `download_sae.py` (racine)

Télécharge le modèle Gemma-3 et le SAE GemmaScope-2 correspondant à `MODEL_SIZE` vers le
cache HF (`HF_HOME`) et `LOCAL_SAE_DIR` respectivement.

```bash
MODEL_SIZE=12b python download_sae.py            # modèle + SAE
python download_sae.py --model-only               # modèle seul
python download_sae.py --sae-only                  # SAE seul
```

### `src/sae/saev5.py` — orchestration principale

Point d'entrée des deux pipelines. `if __name__ == "__main__":` charge le corpus
**principal** (mails réels + variantes augmentées, via
`build_email_train_test_corpus` de `src/data/preparation.py`, avec fallback
synthétique si `LOCAL_MAILS_PATH` est absent), qui domine désormais l'entraînement du
SAE d'extension et du `PhraseLevelSAE` (cf. `RESULTS_TESTS.md` §12 pour le diagnostic
qui a motivé ce choix). Un petit corpus **secondaire** (energy/sports/support,
FineWeb-2/Wikipedia) est gardé uniquement pour la démonstration de diffing
cross-domaine préexistante, encodé post-hoc, jamais utilisé pour l'entraînement.
Exécute ensuite Pipeline 1 et/ou Pipeline 2 selon `PIPELINES` (`p1`, `p2`, ou `p1,p2`).

```bash
PYTHONPATH=. PIPELINES=p1,p2 python src/sae/saev5.py
```

Variables utiles : `EMAIL_TEST_SPLIT` (fraction mails+augmentés réservée au test,
split group-aware par mail d'origine), `MAX_AUGMENTED_PER_MAIL`, `N_TOTAL_ENERGY`/
`N_TOTAL_SPORTS`/`N_TOTAL_SUPPORT` (taille du corpus secondaire de diffing),
`N_TOKENS_EXTRA_TRAIN` (budget d'entraînement de l'extension FrozenCore — cf.
`RESULTS_TESTS.md` §12 : n'a plus d'effet mesurable une fois le corpus dominé par les
emails, de 100k à 2M tokens), `N_FEATURES_TO_LABEL` (150 dans les runs `v10`, contre 10
par défaut — nécessaire pour une puissance statistique correcte sur le taux
d'interprétabilité observé).

Sorties dans `SAVE_DIR` : `p1_top_core_features.json` / `p1_top_extended_features.json`
(features labellisées), `umap_pipeline1_*.html` / `umap_pipeline2_*.html`
(visualisations interactives — suffixe `_emails` pour le corpus principal test-split,
`_diffcorpus` pour le corpus secondaire post-hoc), `p1_diff_energy_sports.csv`
(diffing statistique sur le corpus secondaire), `cache/` (activations, checkpoints SAE,
labels Neuronpedia, réutilisés entre runs).

### `scripts/baseline_gemmascope.py`

Compare mails originaux vs augmentés (perturbations contrôlées) avec le SAE GemmaScope
**natif, sans extension FrozenCore** — question posée : quelles features de base changent
significativement de fréquence d'activation entre les deux groupes ? Zéro nouveau code
d'inférence, n'orchestre que l'existant (`gemma_scope_loader`, `activations`,
`cooccurrence.corpus_diff_stats`, `neuronpedia_labels`, `augmentation.load_augmented`).

```bash
python scripts/baseline_gemmascope.py local_data/emails/Mails.tsv local_data/emails/augmented_mails.jsonl
```

Nécessite `augmented_mails.jsonl`, produit par `run_augmentation.py` (lui-même nécessite
un `Mails.tsv` réel). Les deux fichiers vivent dans `local_data/emails/` (emplacement
canonique, cf. `src/config.py::LOCAL_MAILS_PATH`/`LOCAL_AUGMENTED_MAILS_PATH`).

### `scripts/relabel_diff_csvs.py`

Réapplique offline (CPU, aucun GPU, aucune réextraction) les labels Neuronpedia du
cache canonique aux `diff_*.csv`/`.html` déjà produits par `baseline_gemmascope.py` —
utile quand un run a tourné hors-ligne (cluster sans accès réseau) avant que le cache
de labels ne soit disponible localement.

```bash
python scripts/relabel_diff_csvs.py results_v9_test/cache_baseline_full
```

### `scripts/run_augmentation.py`

Génère des variantes de mails perturbées (émotion, registre, orthographe, urgence — grille
`AXES` dans `src/data/augmentation.py`) via Gemma-3, tracées et validées (garde-fous
factuels : numéros de contrat, montants, dates préservés). **Nécessite un `Mails.tsv`
réel** — non exécuté lors de la validation locale (données EDF absentes de cette
machine) ; à lancer sur la machine disposant du corpus réel.

```bash
LOCAL_MAILS_PATH=local_data/emails/Mails.tsv MODEL_ID=google/gemma-3-12b-it \
  SAVE_DIR=local_data/emails/ python scripts/run_augmentation.py
```

### `scripts/retrieval_demo.py`

Câble `src/sae/retrieval/latent_terms.py` (Latent Terms — BM25 sur le vocabulaire latent
d'un SAE de phrases, Clavié et al. 2026) à un point d'entrée testable sans `Mails.tsv` :
génère un corpus de substitution public (FineWeb-2, domaine "energy") au format attendu,
entraîne un `PhraseLevelSAE`, et exécute une recherche de démonstration.

```bash
python scripts/retrieval_demo.py --n-docs 60 --query "énergie électrique renouvelable"
```

Sur données réelles (une fois `Mails.tsv` disponible), utiliser directement
`src/sae/retrieval/latent_terms.py --mails <Mails.tsv> --query "..."`.

### `src/sae/compare/pipeline.py`

Compare deux modèles d'embeddings de phrases (ex. F2LLM vs `intfloat/multilingual-e5-small`)
via leurs SAE respectifs : alignement cross-modèle des features (`model_compare.py`),
détection de "pollution" par des features non alignées, et — en mode `analysis` —
clustering, graphe de co-occurrence NPMI et diffing de corpus pour un seul modèle.

```bash
python -m src.sae.compare.pipeline --mails <Mails.tsv> --mode analysis
python -m src.sae.compare.pipeline --mails <Mails.tsv> --mode compare \
  --model-a codefuse-ai/F2LLM-v2-80M --model-b intfloat/multilingual-e5-small
```

### `test_chargement_sae.py` / `scripts/test_massive_acts.py` (diagnostics manuels)

Scripts de smoke-test ad hoc (pas de `def test_*`, pas collectés par pytest) :
- `test_chargement_sae.py` : charge le SAE configuré et affiche ses dimensions.
- `scripts/test_massive_acts.py` : corrélation Pearson entre pré-activations "extra" et
  norme du token, pour diagnostiquer la pollution par massive activations. Suppose des
  checkpoints déjà produits par un run complet (`results_v9_test/`).

```bash
python test_chargement_sae.py
python scripts/test_massive_acts.py
```

### Suite de tests (`pytest`)

```bash
pytest tests/ -v
```

8 tests, tous passants. `tests/test_frozen_core.py` est le plus substantiel (vérifie les
formes de sortie de `FrozenCoreResidualSAE` avec un SAE mocké) ; les autres
(`test_bfloat16`, `test_checkpoint`, `test_pooling`, `test_retrieval`,
`test_sparse_storage`) sont des tests unitaires génériques ; `test_interp_embed_diff.py`
exerce `corpus_diff_stats` (l'équivalent projet du `diff_features` d'`interp_embed`, non
vendorisé — cf. `Context.md`).

---

## Architecture du code

```
src/
  config.py              # Source unique de vérité (constantes, presets modèle/SAE)
  sae/
    saev5.py              # Orchestration Pipeline 1 + Pipeline 2 (point d'entrée)
    sae_shared.py          # Harnais d'entraînement ExtendedSAE, ré-exports
    gemma_scope_loader.py  # Chargement SAE GemmaScope-2 (disque local ou Hub)
    neuronpedia_labels.py  # Labels de features via bucket S3 Neuronpedia
    frozen_core.py         # FrozenCoreResidualSAE / ExtendedSAE
    phrase_sae.py          # PhraseLevelSAE + embeddings F2LLM (Pipeline 2)
    batch.py               # BatchTopKEncoder (seuil θ calibré)
    judge.py               # Labellisation LLM locale (odd-one-out)
    retrieval/latent_terms.py  # BM25 sur features SAE (Latent Terms)
    compare/                   # Comparaison cross-modèle d'embeddings
  analysis/
    activations.py         # Extraction/masquage activations, max-pooling
    cooccurrence.py         # NPMI, diffing de corpus (Fisher+BH), clustering
    metrics.py              # FVE, NMSE, L0, classification en aval
    visualization.py        # Exports Plotly HTML autonomes
  data/
    dataset.py               # Ingestion Mails.tsv, GoEmotions
    preparation.py           # build_email_train_test_corpus (corpus PRINCIPAL,
                              #   emails+augmentés, group-aware) + construction du
                              #   corpus SECONDAIRE domaine (FineWeb-2/Wikipedia)
    augmentation.py          # Génération de variantes perturbées
    keywords.py               # Listes de mots-clés par domaine
  storage/
    fragment_store.py       # Stockage CSR (torch) des activations token-level
    shards.py                # Sharding/mmap d'activations denses
scripts/                   # Points d'entrée secondaires (cf. section Scripts)
tests/                     # Suite pytest
external/sae-lens/         # Submodule SAELens (comparaison d'implémentation)
local_data/
  emails/                  # Corpus EDF canonique : Mails.tsv + augmented_mails.jsonl
    archive/                #   shards/fichiers intermédiaires de l'augmentation
  neuronpedia_labels/      # Cache labels Neuronpedia, partagé par tous les runs
  saes/                    # Poids SAE téléchargés (download_sae.py)
docs/                      # architecture.md / experiments.md / references.md
report/                    # Matériel de rapport de stage (état de l'art, résultats...)
```

---

## Limites connues et pistes pour la suite

Voir `Context.md` pour le détail complet (bugs corrigés, historique, prochaines étapes)
et `RESULTS_TESTS.md` §12 pour le diagnostic complet du taux d'interprétabilité. En
résumé :

- **Taux d'interprétabilité des features d'extension (RÉSOLU À ~45%, pas 100%)** :
  corriger le corpus d'entraînement (emails dominants) a fait passer le taux
  d'odd-one-out de 20% à ~41-45% (n=150) — un gain net important, mais qui laisse
  encore ~55% des features d'extension non interprétables par le juge. Pistes non
  explorées faute de temps : robustesse du protocole de jugement lui-même (une seule
  génération greedy par décision, pas de vote majoritaire/ensemble ; qualité du
  contrôle négatif dans `build_feature_examples_with_control`), ou architecture de
  l'extension elle-même (`D_EXTRA`/`K_EXTRA`, actifs non testés dans cette session).
- **Budget d'entraînement de l'extension FrozenCore** : confirmé **non limitant** une
  fois le domaine du corpus corrigé — 100k/500k/2M tokens donnent des taux
  d'interprétabilité statistiquement indistinguables (cf. `RESULTS_TESTS.md` §12).
  L'ancienne recommandation ("augmenter `N_TOKENS_EXTRA_TRAIN`") ne tient plus : le
  problème n'était pas le volume.
- **`run_augmentation.py` / `baseline_gemmascope.py`** : exécutés avec succès à
  l'échelle complète sur le corpus réel EDF (cf. `RESULTS_TESTS.md` §0 et §12) —
  n'est plus une limite.
- **Dashboard interactif** (Streamlit, mentionné dans `Context.md` comme fonctionnalité
  future) : non implémenté.
- **Comparaison documentée avec SAELens** (règle n°2 de `Context.md`) : toujours pas
  faite systématiquement.
