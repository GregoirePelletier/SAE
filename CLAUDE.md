# CLAUDE.md

Prototype de recherche : analyse interprétable de mails clients EDF par SAE (Gemma-3 + GemmaScope-2, extension
`FrozenCoreResidualSAE`). Lire d'abord `docs/HANDOVER.md` (carte, parcours, artefacts) puis `docs/RESULTS_STATUS.md`
(statut des résultats : E01-E07 antérieurs au correctif de filiation parent, rejeu requis). Règles complètes
d'origine : `docs/archive/CLAUDE_long_2026-09.md` ; cluster : `docs/ops.md`.

- Données/caches : ne jamais supprimer données, poids, checkpoints, caches partagés (`local_data/activation_cache/`),
  résultats ni logs sans accord explicite. Ne pas regeler les splits (`configs/post_stage/`) ni réécrire un JSON de résultat.
- Split/parents : `parent_id` vit dans l'espace `load_mails_tsv` ; utiliser `load_and_clean_emails(return_positions=True)`
  pour le repli positionnel. CONFIRM n'entre jamais dans un entraînement, une IDF, une PCA ni un réglage.
- Cache/checkpoint : toute clé encode TOUS les paramètres dont le contenu dépend ; un lien `SAVE_DIR/cache/*` vers une autre
  clé est une erreur voulue, jamais à contourner en supprimant le lien. `SEED` n'entre pas dans la clé d'extraction.
- Précision : bf16 pour les activations Gemma-3 (fp16 déborde) ; branche « extra » de l'extension et SAE du Pipeline 2 en fp32 ;
  ne jamais caster le module entier après construction.
- Juge : `JUDGE_MODEL_ID` (Qwen3.8-27B) toujours découplé de `MODEL_ID` ; aucun auto-jugement.
- Cluster SLURM : aucun calcul sur le frontal (validation bornée <5 s, sans lecture de tenseur/modèle réel, tolérée) ;
  tout le reste par `sbatch`, uniquement sur autorisation. Activations SAE en CSR sparse, jamais denses. `PYTHONUNBUFFERED=1`.
- Boucles > ~1 h GPU : état de progression atomique et reprise (`src/storage/checkpoint.py`) ; structures O(n²) plafonnées.
- Statistiques : `src/analysis/stats.py` (jamais une lecture à l'œil de deux pourcentages).
- Tests : `.venv/bin/python -m pytest tests/ -q` (CPU) ; `scripts/*_test.py` sont des expériences, pas des tests unitaires.
  Un hook `.claude/settings.json` relance la suite après édition d'un `.py` : éviter les éditions inutiles.
- Documentation : présent, sans récit de session ni première personne (`scripts/check_docs.py`) ; `RESULTS_TESTS.md` append-only.
- Git : jamais de trailer `Co-Authored-By: Claude` ; un commit par sujet logique ; pas de push sans demande.
