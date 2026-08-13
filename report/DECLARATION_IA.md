# Déclaration d'usage de l'IA

Ce rapport n'est destiné à aucune revue/conférence dont la base de politiques
IA du plugin `academic-research-skills` couvre le rendu spécifique (ICLR,
NeurIPS, Nature, ACL, EMNLP, revues médicales...) — c'est un mémoire de stage
M2. Cette déclaration adapte donc la substance du protocole (catégorisation
honnête de l'usage réel) sans en suivre le rendu propre à une revue
particulière, et couvre l'ensemble de l'engagement (plusieurs sessions), pas
seulement la dernière session d'édition.

**Outil** : Claude (Anthropic), via Claude Code, en usage interactif continu
tout au long du stage — pas un outil ponctuel de relecture finale.

## Catégories d'usage

| Catégorie | Statut | Détail |
|---|---|---|
| Assistance recherche | Utilisé | Vérification de citations, recherche de littérature manquante (SAEBench, Chanin et al. 2024), vérification croisée d'affirmations contre `RESULTS_TESTS.md`/le code source |
| Vérification de citations | Utilisé | Audit complet de la bibliographie (existence des arXiv ID, exactitude auteurs/venue) |
| Assistance à la rédaction | Utilisé, substantiel | Rédaction de sections entières du rapport au fil du stage (pas seulement des corrections ponctuelles) ; relecture stylistique |
| Assistance à la révision | Utilisé, substantiel | Ajout de réserves méthodologiques suite à un audit externe (voir ci-dessous), correction de descriptions techniques inexactes (ρ_interp), harmonisation terminologique ("réel"→"original") |
| **Assistance à l'analyse** | **Utilisé, substantiel — catégorie la plus significative** | Conception et exécution de tests statistiques (McNemar, Cochran-Armitage, z à deux proportions, analyse de puissance), diagnostic de bugs (dont le bug de conception de corpus qui est le résultat central du rapport), **conception et exécution d'une expérience de confirmation** (job SLURM 42748 : réplique le baseline pré-correctif à n=150 au lieu de n=10, résultat directement cité au chapitre Limites) |
| **Revue par les pairs simulée** | **Utilisé, explicite** | Le rapport a été soumis à un panel de 5 reviewers indépendants (Journal-Fit, Méthodologie, Domaine, Perspective, Avocat du diable — `academic-research-skills:academic-paper-reviewer`) ; les constats retenus sont intégrés directement dans les chapitres concernés (`04_limites_et_perspectives.md` notamment) |
| Autre | Utilisé | Écriture et exécution de code expérimental (`src/sae/saev5.py`, branche `CONFIRMATORY_DOMAIN_BASELINE`), gestion des jobs SLURM, téléchargement de poids de modèles |

## Ce que cela signifie concrètement

Plusieurs affirmations actuellement dans ce rapport reposent directement sur
un travail d'analyse effectué par l'IA, pas seulement rédigé par elle :
- La résolution statistique du constat critique C1 (`04_limites_et_perspectives.md`,
  `RESULTS_TESTS.md` §46) : le calcul de significativité (z=2,74, p≈0,006) et
  l'expérience de confirmation qui le produit ont été conçus et exécutés par
  l'IA, pas seulement transcrits.
- Le résultat de l'audit `academic-paper-reviewer` lui-même constitue une
  critique méthodologique générée par IA du travail produit pendant le
  stage — y compris de travail antérieur également produit avec assistance
  IA. Ce n'est pas une relecture humaine indépendante.

Ce que l'IA n'a pas fait : choisir le sujet de stage, définir les objectifs
initiaux (cadrage EDF R&D), décider quelles pistes poursuivre en priorité,
ni valider en dernier ressort qu'un résultat était suffisamment solide pour
être présenté comme définitif — ces décisions sont restées celles de
l'auteur à chaque étape.

## Limite de cette déclaration

Rédigée par l'IA elle-même à partir de sa propre visibilité sur les sessions
de travail (le contexte de conversation compressé au fil du temps limite la
visibilité exacte sur les toutes premières sessions du stage). L'auteur reste
responsable de vérifier et compléter cette déclaration avant tout usage
externe du rapport (dépôt final, dossier de candidature).
