# Matériel de rapport de stage

Ce dossier rassemble, au fil du projet, le matériel destiné au rapport de stage —
rédigé pour être directement réutilisable (état des lieux factuel, chiffres sourcés,
pas de tournures conversationnelles). Mis à jour à chaque évolution importante
(cf. `Context.md`, section "Rapport de recherche").

- [`01_etat_de_lart.md`](01_etat_de_lart.md) — Sparse Autoencoders, GemmaScope,
  auto-interprétation par juge LLM, positionnement du projet.
- [`02_architecture.md`](02_architecture.md) — architecture technique du système
  (pipelines, corpus, stockage) — synthèse de `docs/architecture.md` orientée rapport.
- [`03_experiences_et_resultats.md`](03_experiences_et_resultats.md) — **cœur du
  rapport** : démarche expérimentale complète sur le diagnostic et la correction du
  taux d'interprétabilité des features d'extension, avec tables de résultats et
  interprétation.
- [`04_limites_et_perspectives.md`](04_limites_et_perspectives.md) — limites connues,
  comparaisons avec l'état de l'art restant à faire, pistes pour la suite du stage.

Sources primaires (à citer/vérifier avant intégration finale dans le rapport) :
`RESULTS_TESTS.md` (journal chronologique détaillé de toutes les expériences et jobs
SLURM), `Context.md` (historique des décisions et bugs corrigés), `docs/` (référence
technique stable).
