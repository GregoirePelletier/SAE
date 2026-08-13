# CLAUDE.md

Instructions pour un agent travaillant sur ce dépôt. Ne duplique pas
`RESULTS_TESTS.md` (journal d'expériences numéroté) ni `docs/` (référence
technique) — les complète avec ce qui doit être vu avant de les lire.

## Projet

Analyse interprétable de mails clients EDF via Sparse Autoencoders sur les
hidden states de Gemma-3, labellisation par GemmaScope-2/Neuronpedia + juge
LLM local. Deux pipelines : Pipeline 1 (Gemma-3 → SAE GemmaScope-2 + extension
`FrozenCoreResidualSAE`), Pipeline 2 (F2LLM-v2 → `PhraseLevelSAE` from-scratch).

## Règles de fond

- Ne pas réimplémenter une fonctionnalité déjà présente dans SAELens ou
  interp_embed sans comparaison documentée (`docs/references.md`).
- `FrozenCoreResidualSAE` est spécifique au projet, ne pas la remplacer par un
  usage direct de SAELens.
- bf16 partout, y compris en local (les activations massives de Gemma-3
  débordent en fp16). La branche "extra" de `FrozenCoreResidualSAE`/`ExtendedSAE`
  reste en fp32 ; ne jamais caster le module entier après construction.
- Toute modification doit laisser `pytest tests/ -q` 100% vert.

## Convention de test — deux niveaux, ne pas les mélanger

- `tests/` : assertions unitaires rapides, CPU uniquement (shape, dtype,
  non-régression, "ça ne plante pas"). `pytest tests/ -q` doit rester 100% vert.
- `scripts/*_test.py` / `*_audit.py` : expériences empiriques (ablations,
  audits méthodologiques) qui produisent un JSON de résultats
  (`SAVE_DIR/cache/*.json` ou `local_data/.../*.json`) → une section numérotée
  dans `RESULTS_TESTS.md` (format défini en tête de ce fichier) → un onglet
  dashboard si pertinent (`src/visualization/dashboard.py`).

Un hook post-edit (`.claude/settings.json`) relance `pytest tests/ -q` en
tâche de fond après toute édition d'un fichier `.py` et remonte les échecs.

## Statistiques — utiliser `src/analysis/stats.py`

Module partagé (McNemar apparié, Cochran-Armitage, proportions+IC de Wilson,
h de Cohen, BH/FDR, analyse de puissance) — ne pas réinventer un test par
script.

## Seeds — piège fréquent

`SEED` (entraînement SAE/juge) et `CORPUS_SPLIT_SEED` (split train/test du
corpus) sont **découplés** dans `src/config.py`, tous deux à 42 par défaut —
ne pas supposer qu'ils sont le même paramètre en reconstruisant un split de
référence.

## Cluster SLURM

Conventions de partitions, soumission, logs, disque : `docs/ops.md`.

## Git

Ne jamais ajouter de trailer `Co-Authored-By: Claude` dans les messages de
commit de ce dépôt — GitHub l'affiche comme un contributeur, ce que ce projet
ne veut pas.

## Documentation

- `RESULTS_TESTS.md` est append-only : les identifiants `§N` sont cités
  depuis le rapport et ne doivent jamais être renumérotés. Chaque nouvelle
  section suit le format : Question / Écart à la configuration de référence
  (`docs/evaluation_protocol.md`) / Méthode statistique / n / Résultat /
  Conclusion / Limite connue.
- Rédiger au présent, sans numéro de version interne (`v9`, `v10`...) ni récit
  de session : une contrainte de conception encore active se formule comme
  une règle, pas comme le récit de sa découverte.

## Diagnostics — un run est-il sain avant d'en tirer une conclusion ?

`docs/sae_diagnostics_playbook.md` : checklist ordonnée (convergence →
fidélité de reconstruction → capacité → interprétabilité → significativité →
indépendance du juge) avant de faire confiance à un résultat. Figures
associées : `scripts/generate_diagnostic_plots.py` (agrégation rétroactive,
zéro rerun) + `src/analysis/plotting.py` (fonctions réutilisables).
