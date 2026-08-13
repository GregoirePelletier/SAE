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
  contrainte de longueur pour EDF R\&D, qui conserve le panorama technique complet
  (ex. comparaison exhaustive des modèles d'embeddings) coupé dans la version
  université.

Les deux versions partagent les mêmes chiffres et conclusions scientifiques ; seules
la portée et la profondeur de détail diffèrent. **Ne pas compiler** (pas de
`pdflatex` disponible sur cette machine au moment de la rédaction) sans vérifier au
préalable le nombre de pages réel de la version université, et le niveau de
confidentialité à valider avec le maître de stage avant remise.

## Rapport assemblé (matériel de travail, pas un livrable final)

`report/dist/RAPPORT_DE_STAGE.md` — concaténation des fichiers sources
ci-dessous, utile comme matériel de travail/traçabilité, mais **ce n'est pas
le document à remettre** (voir les deux `.tex` ci-dessus). Généré par
`scripts/build_report.py`, jamais versionné (`report/dist/` est gitignoré) :
ne pas éditer directement, éditer la source puis relancer
`.venv/bin/python scripts/build_report.py`.

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
- [`04_limites_et_perspectives.md`](04_limites_et_perspectives.md) — limites
  connues, comparaisons avec l'état de l'art, pistes pour la suite du stage.
- [`06_conclusion.md`](06_conclusion.md) — bilan général, compétences acquises,
  perspectives.
- [`07_bibliographie.md`](07_bibliographie.md) — références académiques, dépôts et
  outils réutilisés.

## Déclaration d'usage de l'IA

**[`DECLARATION_IA.md`](DECLARATION_IA.md)** — à lire avant tout dépôt final ou usage
dans un dossier de candidature : catégorisation honnête de l'usage de l'IA sur
l'ensemble du stage, y compris les cas où l'IA a conçu/exécuté une analyse (pas
seulement rédigé du texte).

Sources primaires (à citer/vérifier avant intégration finale) : `RESULTS_TESTS.md`
(journal d'expériences numéroté, résultats et protocoles), `docs/` (référence
technique stable).
