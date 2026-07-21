# Matériel de rapport de stage

Ce dossier contient le rapport de stage de M2 assemblé (**`RAPPORT_DE_STAGE.md`**),
ainsi que ses fichiers sources modulaires, mis à jour à chaque évolution importante
(cf. `Context.md`, section "Rapport de recherche").

## Rapport assemblé

**[`RAPPORT_DE_STAGE.md`](RAPPORT_DE_STAGE.md)** — document complet prêt à relire/
convertir (ex. pandoc → PDF/Word) : page de garde, résumé FR/EN, sommaire,
introduction générale, 5 chapitres numérotés, conclusion générale, bibliographie.
Généré par concaténation/renumérotation des fichiers sources ci-dessous — **ne pas
éditer directement** ; éditer la source puis régénérer (script d'assemblage inline,
cf. historique git de ce fichier pour la commande exacte utilisée).

**Champs restant à compléter par l'auteur** (marqués `[... — à compléter]` dans le
fichier) : nom de l'établissement/spécialité de Master, nom du auteur à confirmer,
noms des tuteurs (entreprise/académique), dates exactes de stage, remerciements. La
§3.10 (ablation de mise à l'échelle v12) est un placeholder à compléter avec les
résultats une fois la chaîne de jobs SLURM terminée.

## Fichiers sources

- [`FRONT_MATTER.md`](FRONT_MATTER.md) — page de garde, résumé/abstract, sommaire.
- [`00_introduction.md`](00_introduction.md) — contexte EDF/SEQUOIA, objectifs,
  démarche, plan du rapport.
- [`01_etat_de_lart.md`](01_etat_de_lart.md) — Sparse Autoencoders, GemmaScope,
  auto-interprétation par juge LLM, positionnement du projet.
- [`02_architecture.md`](02_architecture.md) — architecture technique du système
  (pipelines, corpus, stockage) — synthèse de `docs/architecture.md` orientée rapport.
- [`03_experiences_et_resultats.md`](03_experiences_et_resultats.md) — **cœur du
  rapport** : démarche expérimentale complète, du diagnostic initial à l'ablation de
  mise à l'échelle finale, avec tables de résultats et interprétation.
- [`05_erreurs_et_corrections.md`](05_erreurs_et_corrections.md) — chapitre
  consolidé de tous les bugs/erreurs de conception rencontrés et corrigés, par phase
  chronologique du stage.
- [`04_limites_et_perspectives.md`](04_limites_et_perspectives.md) — limites
  connues, comparaisons avec l'état de l'art, pistes pour la suite du stage.
- [`06_conclusion.md`](06_conclusion.md) — bilan général, compétences acquises,
  perspectives.
- [`07_bibliographie.md`](07_bibliographie.md) — références académiques, dépôts et
  outils réutilisés.

Sources primaires (à citer/vérifier avant intégration finale) : `RESULTS_TESTS.md`
(journal chronologique détaillé de toutes les expériences et jobs SLURM), `Context.md`
(historique des décisions et bugs corrigés), `docs/` (référence technique stable).
