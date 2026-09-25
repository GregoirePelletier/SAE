# Architecture

Description technique du code. Pour l'organisation du dépôt et la reprise, voir
`docs/HANDOVER.md` ; pour les résultats, `docs/RESULTS_STATUS.md`.

## Objectif

Rendre explorables des emails de clients : retrouver des emails selon une propriété, comparer
deux groupes d'emails, regrouper selon une question, et expliquer ces résultats par des features
nommées. Les features viennent de Sparse Autoencoders (SAE) appliqués aux représentations
internes d'un modèle de langue.

## Pipeline principal (Gemma-3 et GemmaScope-2)

```
email → Gemma-3 gelé (activations du residual stream à la couche LAYER)
      → SAE GemmaScope-2 préentraîné et gelé : features CORE
      → extension entraînée sur le résidu non reconstruit par CORE : features EXTRA
      → agrégation par email (maximum sur les tokens) : vecteur FULL = CORE + EXTRA
```

- Configuration par défaut (`src/config.py`) : Gemma-3-12B, couche 31, SAE de largeur 16 384.
  La campagne post-soutenance utilise Gemma-3-1B, couche 13. Les préréglages par taille de
  modèle sont dans `_PRESETS` (`MODEL_SIZE` = `12b`, `4b`, `1b`, `270m`).
- CORE est le SAE de DeepMind, jamais réentraîné. Ses features sont nommées à partir des
  descriptions publiques de Neuronpedia (`src/sae/neuronpedia_labels.py`).
- L'extension (`SAEBoostResidualSAE`, `src/sae/frozen_core.py`) est un second SAE de 1 024
  features (`D_EXTRA`), dont 5 actives par token (`K_EXTRA`), entraîné de zéro sur le résidu.
  Son décodeur est initialisé par ACP (par défaut) ou aléatoirement (`EXTRA_DECODER_INIT`).
- Les features EXTRA n'existent dans aucune base publique : elles sont nommées par un modèle
  juge local (`src/sae/judge.py`, test de l'intrus `odd_one_out_judge`). Le juge
  (`JUDGE_MODEL_ID`, Qwen3.8-27B) est toujours différent du modèle analysé.

## Pipeline alternatif (F2LLM et PhraseLevelSAE)

```
email → découpage en phrases → embeddings F2LLM-v2 (défaut : 330M)
      → PhraseLevelSAE entraîné de zéro (8 192 features, 16 actives) → maximum sur les phrases
```

Code : `src/sae/phrase_sae.py`. Nommage des features par `local_gemma_judge`
(`src/sae/judge.py`). Non utilisé dans la campagne post-soutenance.

## Données

- Corpus principal : emails d'origine (`local_data/emails/Mails.tsv`) et variantes générées
  par `scripts/run_augmentation.py` (`local_data/emails/augmented_mails.jsonl`), selon les axes
  émotion, registre, orthographe et urgence (`src/data/augmentation.py`, `AXES`). C'est ce
  corpus qui entraîne l'extension et le PhraseLevelSAE.
- Découpage : un email et toutes ses variantes tombent toujours dans le même ensemble.
  Avant la soutenance, `build_email_train_test_corpus` (`src/data/preparation.py`) fait un
  découpage train / test ; après, `src/post_stage/dataset_contract.py` lit un découpage figé
  FIT / DEV / CONFIRM (`configs/post_stage/split_assignments.json`, 60 / 15 / 25 %). Le
  rattachement d'une variante à son email d'origine passe par `parent_sha1` si le fichier des
  variantes le contient, sinon par la position (`parent_id`, traduite par
  `load_and_clean_emails(return_positions=True)`).
- Corpus secondaire (énergie, sport, support, depuis FineWeb-2 ou Wikipédia) : encodé après
  l'entraînement, seulement pour une démonstration de comparaison entre domaines.

## Stockage et cache

- Activations par token : format creux maison (`src/storage/fragment_store.py`), utilisé pour
  retrouver les exemples d'une feature, pas pendant l'entraînement.
- Entraînement de l'extension : réservoir de résidus en mémoire mappée (`open_mmap_reservoir`),
  ce qui permet d'aller jusqu'à 100 ou 200 millions de tokens.
- Cache d'extraction partagé entre runs (`local_data/activation_cache/<clé>/`, clé calculée par
  `compute_activation_cache_key` dans `src/sae/sae_shared.py`) : voir `docs/ops.md`.
- Reprise après interruption : `src/storage/checkpoint.py`.

Pistes d'accélération repérées mais non faites : les lots de l'extension sont construits
ligne par ligne sur le réservoir mappé (un `DataLoader` sur `TensorDataset`), alors que le
PhraseLevelSAE indexe ses lots en une fois ; le réservoir passe toujours par la mémoire mappée
même quand il tiendrait sur le GPU ; la taille de lot d'extraction (`EXTRACTION_BATCH_SIZE`,
4 par défaut) n'a jamais été optimisée ; aucune utilisation de `torch.compile`.

## Précision numérique

bf16 pour Gemma-3 : ses activations comportent des valeurs extrêmes (de l'ordre de 10⁵) qui
débordent en fp16. La branche EXTRA de l'extension et le PhraseLevelSAE restent en fp32. Ne pas
convertir un module entier après sa construction.

## Recherche, cooccurrence et analyses

- `src/analysis/cooccurrence.py` : NPMI entre features, comparaison de fréquences entre deux
  groupes (`corpus_diff_stats`), graphe de cooccurrence. Avant la soutenance, le regroupement de
  features par Louvain sur ce graphe n'était pas distinguable du hasard (`RESULTS_TESTS.md` §66).
- `src/analysis/diff_hypothesis_generator.py`, `hypothesis_verifier.py`,
  `correlations_verified.py`, `clustering_llm.py` : génération et vérification d'hypothèses,
  associations vérifiées, nommage de groupes, repris du papier interp_embed
  (voir `docs/archive/INTERP_EMBED_COVERAGE.md`).
- `src/analysis/stats.py` : tests statistiques partagés (McNemar, proportions et intervalles,
  Benjamini-Hochberg, puissance).
- `src/sae/retrieval/latent_terms.py` : réimplémentation de *Latent Terms* (Clavié et al. 2026,
  BM25 sur le vocabulaire d'un SAE). Aucun résultat avec cette version : l'entraînement n'a
  jamais abouti dans le temps alloué.
- `src/post_stage/representations.py` : représentations de comparaison (TF-IDF, bge-m3) ;
  `src/post_stage/stability.py` : appariement et groupes de features pour E05.

## Code

```
src/
  config.py                 configuration, surchargeable par variable d'environnement
  sae/
    saev5.py                point d'entrée des deux pipelines
    sae_shared.py           entraînement de l'extension, cache d'extraction
    frozen_core.py          CORE gelé et extension (SAEBoostResidualSAE)
    gemma_scope_loader.py   chargement des SAE GemmaScope-2
    neuronpedia_labels.py   noms des features CORE
    phrase_sae.py           PhraseLevelSAE et embeddings F2LLM
    batch.py                activation BatchTopK
    judge.py                modèle juge et nommage des features
    retrieval/              Latent Terms
    compare/                comparaison de modèles d'embeddings
  analysis/                 métriques, statistiques, cooccurrence, hypothèses, figures
  data/                     lecture des emails, augmentation, découpage
  storage/                  stockage creux, reprise après interruption
  post_stage/               découpage FIT/DEV/CONFIRM, représentations, stabilité, profilage
  visualization/dashboard.py
scripts/post_stage/         expériences E01 à E07
scripts/                    scripts d'avant la soutenance
slurm/                      recettes sbatch (voir slurm/README.md)
tests/                      tests CPU
```

## Scripts principaux

`src/sae/saev5.py` exécute le pipeline principal et/ou l'alternatif (`PIPELINES` = `p1`, `p2`
ou `p1,p2`), depuis la racine avec `PYTHONPATH=.`, via une recette Slurm. Il écrit dans
`SAVE_DIR` : features nommées (`p1_top_core_features.json`, `p1_top_extended_features.json`),
projections UMAP, checkpoints et `cache/`. Variables utiles : `N_TOKENS_EXTRA_TRAIN` (taille du
réservoir), `N_FEATURES_TO_LABEL` (nombre de features jugées), `MAX_AUGMENTED_PER_MAIL`,
`EMAIL_TEST_SPLIT`.

```bash
MODEL_SIZE=1b .venv/bin/python download_sae.py     # modèle et SAE (--model-only, --sae-only)
```

Autres scripts :

- `scripts/run_augmentation.py` : génère les variantes des emails avec Gemma-3, en conservant
  numéros de contrat, montants et dates.
- `scripts/baseline_gemmascope.py` : compare emails d'origine et variantes avec le seul SAE CORE ;
  `scripts/relabel_diff_csvs.py` réapplique ensuite les noms Neuronpedia sans recalcul.
- `scripts/retrieval_demo.py` : démonstration de Latent Terms sur un corpus public.
- `python -m src.sae.compare.pipeline` : comparaison de deux modèles d'embeddings de phrases par
  leurs SAE.
- `scripts/generate_diagnostic_plots.py` : courbes d'entraînement et diagnostics d'un run, sans
  recalcul.
- `scripts/test_chargement_sae.py`, `scripts/test_massive_acts.py` : vérifications manuelles du
  chargement du SAE et des activations extrêmes.

## Dashboard

`src/visualization/dashboard.py` (Streamlit) lit uniquement des fichiers de résultats : aucune
inférence, aucun GPU. Pages : vue d'ensemble d'un run, UMAP, features (exemples positifs et
négatifs), diffing, recherche par mot-clé, diagnostics d'entraînement, robustesse du juge, et une
page par expérience post-soutenance (E05 à E08). Lancement et fichiers nécessaires :
`docs/HANDOVER.md`.
