# Matériel de rapport de stage

## Livrables finaux (LaTeX, deux versions)

Le document destiné au jury et à l'entreprise existe en **deux versions LaTeX**,
dérivées d'un même contenu de base mais divergentes dans leur portée et leur niveau
de détail :

- **[`RAPPORT_STAGE_UNIVERSITE.tex`](RAPPORT_STAGE_UNIVERSITE.tex)** — version
  soumise au jury M2 Mathématiques et IA, structurée et calibrée pour respecter les
  consignes officielles (`pdf/RapportSoutenancesDS-2024-M2-MathsetIA.pdf`) :
  Introduction (~3p), Méthodes/état de l'art (~5-7p), Contribution (~5-7p),
  Résultats (~5-7p), Discussion et conclusion (~2p). Coupe le contenu non essentiel
  au jury (ex. panorama exhaustif des modèles d'embeddings français) et évite toute
  narration chronologique systématique des étapes du stage.
- **[`RAPPORT_STAGE_ENTREPRISE.tex`](RAPPORT_STAGE_ENTREPRISE.tex)** — version sans
  contrainte de longueur pour EDF R\&D, qui conserve/ajoute le détail technique et
  les quelques bascules de trajectoire du stage utiles à un lecteur interne
  (panorama complet des embeddings, corrections d'infrastructure notables,
  ex. bug `d_model`/préallocation mémoire).

Les deux versions partagent les mêmes chiffres et conclusions scientifiques ; seules
la portée et la profondeur de détail diffèrent. **Ne pas compiler** (pas de
`pdflatex` disponible sur cette machine au moment de la rédaction) sans vérifier au
préalable le nombre de pages réel de la version université, et le niveau de
confidentialité à valider avec le maître de stage avant remise.

## Rapport assemblé (matériel de travail, pas un livrable final)

**[`RAPPORT_DE_STAGE.md`](RAPPORT_DE_STAGE.md)** — concaténation brute des fichiers
sources ci-dessous, plus proche d'un journal chronologique détaillé que d'un rapport
académique final : utile comme matériel de travail/traçabilité, mais **ce n'est pas
le document à remettre** (voir les deux `.tex` ci-dessus). Généré par
concaténation/renumérotation des fichiers sources ci-dessous — ne pas éditer
directement ; éditer la source puis régénérer (script d'assemblage inline, cf.
historique git de ce fichier pour la commande exacte utilisée).

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
