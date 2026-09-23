# Passation — carte du dépôt

Ce document est une carte, pas un journal. Pour le "pourquoi" de chaque
choix, `CLAUDE.md` (règles actives) et `RESULTS_TESTS.md`/`docs/post_stage/`
(résultats) restent les sources de vérité — celui-ci n'en recopie aucun
chiffre.

## Parcours actif vs référence historique vs exploration archivée

- **Parcours actif (post-soutenance, E00-E09)** : `src/post_stage/`,
  `scripts/post_stage/`, `slurm/post_stage/`, `configs/post_stage/`,
  `docs/post_stage/`. Statut par expérience : `docs/RESULTS_STATUS.md`.
- **Pipeline de production (les deux pipelines, utilisé par le parcours actif
  ET par la référence historique)** : `src/sae/saev5.py` + `src/config.py`.
  Ne pas dupliquer sa configuration ailleurs — toute valeur est surchargeable
  par variable d'environnement, cf. README section Configuration.
- **Référence historique (rapport soutenu)** : `results_v10_emails_main/` et
  la famille `results_v*` à la racine (ablations layer/K_EXTRA/volume/seed),
  documentées par `docs/experiments.md` (index par question de recherche) et
  `docs/archived_runs_manifest.md` (quels `results_v*/` supprimés du disque
  correspondent à quel `.slurm` pour les reproduire). Rapport lui-même :
  `report/RAPPORT_STAGE_ENTREPRISE.tex` / `RAPPORT_STAGE_UNIVERSITE.tex`.
  **Ne pas modifier le fond de ces deux fichiers** ni les chemins de figures
  qu'ils référencent.
- **Exploration ponctuelle** : la plupart des ~50 scripts directement sous
  `scripts/` (hors `scripts/post_stage/`) sont des audits/ablations à
  conclusion figée, chacun documenté par sa propre entrée narrative dans
  `RESULTS_TESTS.md` plutôt que par un `.slurm` actif. Ne pas supposer qu'un
  script sans recette Slurm active est mort avant d'avoir grep sa présence
  dans `slurm/`, `docs/`, `tests/` et `RESULTS_TESTS.md` — plusieurs (ex.
  `retrieval_demo.py`, `test_chargement_sae.py`, `generate_diagnostic_plots.py`)
  sont documentés comme utilitaires actifs sans avoir de recette dédiée.

## Points d'entrée réels (pas d'invention)

- Pipeline complet : `PYTHONPATH=. python src/sae/saev5.py` (via un des
  `.slurm` de `slurm/pipeline_runs/` en pratique).
- Post-soutenance E01-E07 : scripts **autonomes**, pas de CLI unifiée —
  `python scripts/post_stage/eNN_*.py`, invoqués par `slurm/post_stage/
  *.slurm` (numérotés dans l'ordre de dépendance, cf. leur préfixe `00_` à
  `15_`).
- `src/post_stage/cli.py` : une seule sous-commande réelle, `freeze-corpus`
  (déjà exécutée, cf. `configs/post_stage/`). Testée
  (`tests/post_stage/test_cli.py`) mais non invoquée par aucune recette Slurm
  active — ne pas présenter comme une CLI plus large qu'elle ne l'est.
- Dashboard : `.venv/bin/python -m streamlit run src/visualization/
  dashboard.py` — lecture seule des artefacts déjà produits, aucune
  extraction/inférence au chargement d'une page.
- Tests : `pytest tests/ -q` (doit rester 100% vert, cf. CLAUDE.md).

## Environnement

Installation, accès HuggingFace gated, dépannage Windows/cluster :
`docs/ops.md` (ne pas dupliquer ici). Racine du projet configurable via
`SAE_ROOT` (défaut : chemin absolu de ce clone) — introduit par ce nettoyage
pour que `slurm/*.slurm` et les scripts `interp_embed` fonctionnent depuis
n'importe quel clone, pas seulement `/home/h21486/SAE`. Modèles/données
(`MODEL_ID`, `EMB_MODEL`, `JUDGE_MODEL_ID`, `LOCAL_MAILS_PATH`, `SAVE_DIR`,
...) restent individuellement surchargeables par variable d'environnement
(`src/config.py`) — ne pas les recentraliser dans un second mécanisme.

## Artefacts — deux inventaires

### 1. Minimum pour ouvrir la démo CPU (dashboard, sans poids ni extraction)

Run recommandé : `results_post_stage_e01_fit_1b_layer13_k5/` (campagne
post-soutenance complète, E01-E07 sur la même config). Pour cette
démonstration, seuls les JSON/HTML/parquet à la racine du répertoire de run
sont nécessaires (petits, quelques Mo à ~26 Mo pour les HTML UMAP) :
`e01_representation_comparison.json`, `e01_pooling_and_length_analysis.json`,
`e02_feature_registry.json`, `e03_property_retrieval.json`,
`e04_diffing.json`, `e05_stability.json`, `e05_stability_random_init.json`,
`e06_correlations.json`, `e07_clustering.json`, `results.json`,
`p1_top_core_features.json`, `p1_top_extended_features.json`,
`umap_pipeline1_*.html/.parquet`. **Pas besoin** de `p1_extended_sae.pt` /
`p1_frozen_core_d1024_k5.pt` (poids, ~85-88 Mo chacun) ni de
`cache/p1_all_doc_acts*.pt`/`cache/p1_raw_residuals.memmap` pour la simple
consultation — ces derniers ne servent qu'à ré-encoder/reprendre un calcul.
Si un artefact listé ici manque pour un run donné, le dashboard doit
l'indiquer par son nom exact et l'étape qui le produit (`page_pilot_e08` et
les autres pages `eNN` le font déjà) — jamais un zéro silencieux.

### 2. Dépendances pour reproduire une expérience (au-delà de la consultation)

- Rejouer E01-E07 sur le run existant : les `.pt`/`.memmap` de `cache/`
  ci-dessus (déjà présents pour `results_post_stage_e01_fit_1b_layer13_k5`)
  + le modèle Gemma-3 correspondant sous `models/` (ou cache HF).
- Reproduire depuis zéro (nouveau corpus/seed/config) : corpus source
  (`local_data/emails/`, ~non mesuré ici, gros), modèle extracteur
  (`models/gemma-3-{taille}-it`), juge (`models/Qwen3.8-27B`), et repasser
  par `slurm/post_stage/00_profile.slurm` → `01_e01_fit_reference.slurm` →
  ... dans l'ordre numéroté (dépendances documentées dans le plan
  `Plan_execution_SAE_15_jours_Claude_Code.md` §16.5).
- Chaque script Slurm exporte explicitement les chemins qu'il consomme
  (`LOCAL_MAILS_PATH`, `MODEL_ID`, etc., tous dérivés de `SAE_ROOT` depuis ce
  nettoyage) — lire l'en-tête du `.slurm` concerné pour la liste exacte
  plutôt que de la deviner.

**Liens symboliques** : 4 sous `results_post_stage_e01_fit_1b_layer13_k5/cache/`
(`p1_all_doc_acts.pt`, `p1_raw_residuals.memmap` + `.meta.json`,
`p1_token_fragments`), tous pointant vers le cache d'extraction partagé
`local_data/activation_cache/<clé>/` — vérifiés résolus (pas cassés) au
moment de ce nettoyage. **Cibles en chemin absolu**
(`/home/h21486/SAE/local_data/activation_cache/...`) : transmettre ce
répertoire de run à quelqu'un sans transmettre aussi `local_data/
activation_cache/` (gros, non mesuré ici, cf. `.gitignore` — jamais suivi
par Git) donnerait des liens cassés, silencieusement, tant que personne ne
tente de rejouer E01-E07 depuis les caches (la simple lecture des JSON
listés ci-dessus n'y touche pas). Non corrigé ici : régénérer ces liens ou
migrer le cache relève d'une opération sur les données, hors mandat de ce
nettoyage.

## Dépannage

Pièges connus (cache/checkpoint, seeds, PyTorch/HuggingFace, diagnostics de
run) : `CLAUDE.md` — toujours le relire à jour plutôt que de s'y fier de
mémoire, il est activement maintenu. Ne pas le dupliquer ici.

## Reprise après ce nettoyage

`docs/CLEANUP_MANIFEST.md` liste précisément ce qui a été changé dans ce
passage (documentation, portabilité, dashboard) et ce qui reste à décider.
