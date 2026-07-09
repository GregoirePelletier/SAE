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

## État actuel (résumé — détail complet dans `Context.md`)

Le pipeline a été audité et rendu fonctionnel de bout en bout : bugs de chargement SAE
corrigés, labellisation Neuronpedia réparée (ancienne route REST cassée → téléchargement
S3 direct), plusieurs bugs de dtype (fp16/bf16) découverts et corrigés par exécution
réelle, code mort supprimé, modules orphelins raccordés. Validé end-to-end en local sur
**Gemma-3-270M-it** (6 Go VRAM) ; la cible de production est **Gemma-3-12B-it**, à
exécuter sur une machine avec plus de VRAM (cf. `run_sae.slurm` pour le profil cluster).

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
python -c "from src.sae.neuronpedia_labels import fetch_neuronpedia_labels; \
  fetch_neuronpedia_labels(model_id='gemma-3-12b-it', layer=24, width='16k', \
  cache_path='results/cache/neuronpedia_labels_core.json')"

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
| `LOCAL_MAILS_PATH` | `./local_data/Mails.tsv` | Corpus EDF réel (absent hors machine de calcul) |
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
(FineWeb-2/Wikipedia par domaine + emails, avec fallback synthétique si
`LOCAL_MAILS_PATH` est absent), puis exécute Pipeline 1 et/ou Pipeline 2 selon
`PIPELINES` (`p1`, `p2`, ou `p1,p2`).

```bash
PYTHONPATH=. PIPELINES=p1,p2 python src/sae/saev5.py
```

Variables utiles pour un run réduit (smoke test) : `N_TOTAL_ENERGY`/`N_TOTAL_SPORTS`/
`N_TOTAL_SUPPORT` (taille du corpus par domaine), `N_TOKENS_EXTRA_TRAIN` (budget
d'entraînement de l'extension FrozenCore — doit être assez grand pour que les features
d'extension ne restent pas `dead_feature`, cf. section Limites connues),
`N_FEATURES_TO_LABEL`.

Sorties dans `SAVE_DIR` : `p1_top_core_features.json` / `p1_top_extended_features.json`
(features labellisées), `umap_pipeline1_*.html` / `umap_pipeline2_*.html`
(visualisations interactives), `p1_diff_energy_sports.csv` (diffing statistique),
`cache/` (activations, checkpoints SAE, labels Neuronpedia, réutilisés entre runs).

### `scripts/baseline_gemmascope.py`

Compare mails originaux vs augmentés (perturbations contrôlées) avec le SAE GemmaScope
**natif, sans extension FrozenCore** — question posée : quelles features de base changent
significativement de fréquence d'activation entre les deux groupes ? Zéro nouveau code
d'inférence, n'orchestre que l'existant (`gemma_scope_loader`, `activations`,
`cooccurrence.corpus_diff_stats`, `neuronpedia_labels`, `augmentation.load_augmented`).

```bash
python scripts/baseline_gemmascope.py <Mails.tsv> <augmented_mails.jsonl>
```

Nécessite `augmented_mails.jsonl`, produit par `run_augmentation.py` (lui-même nécessite
un `Mails.tsv` réel).

### `scripts/run_augmentation.py`

Génère des variantes de mails perturbées (émotion, registre, orthographe, urgence — grille
`AXES` dans `src/data/augmentation.py`) via Gemma-3, tracées et validées (garde-fous
factuels : numéros de contrat, montants, dates préservés). **Nécessite un `Mails.tsv`
réel** — non exécuté lors de la validation locale (données EDF absentes de cette
machine) ; à lancer sur la machine disposant du corpus réel.

```bash
LOCAL_MAILS_PATH=/chemin/Mails.tsv MODEL_ID=google/gemma-3-12b-it \
  python scripts/run_augmentation.py
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
    preparation.py           # Construction corpus domaine (FineWeb-2/Wikipedia)
    augmentation.py          # Génération de variantes perturbées
    keywords.py               # Listes de mots-clés par domaine
  storage/
    fragment_store.py       # Stockage CSR (torch) des activations token-level
    shards.py                # Sharding/mmap d'activations denses
scripts/                   # Points d'entrée secondaires (cf. section Scripts)
tests/                     # Suite pytest
external/sae-lens/         # Submodule SAELens (comparaison d'implémentation)
```

---

## Limites connues et pistes pour la suite

Voir `Context.md` pour le détail complet (bugs corrigés, historique, prochaines étapes).
En résumé :

- **Qualité de labellisation sur petits modèles** : le juge LLM local (features
  d'extension `ExtendedSAE`, non couvertes par Neuronpedia) est nettement moins fiable
  sur `gemma-3-270m-it` que sur un modèle plus grand — à revalider sur `12b`.
- **Budget d'entraînement de l'extension FrozenCore** : avec un corpus/nombre de tokens
  trop faible, les features "extra" restent `dead_feature` (jamais activées) — augmenter
  `N_TOKENS_EXTRA_TRAIN` et la taille du corpus.
- **`run_augmentation.py` / `baseline_gemmascope.py`** : nécessitent un `Mails.tsv` réel,
  non disponible sur la machine de validation locale — à exécuter sur la machine de
  calcul cible.
- **Dashboard interactif** (Streamlit, mentionné dans `Context.md` comme fonctionnalité
  future) : non implémenté.
