# Introduction générale

## Contexte

Ce stage s'inscrit au sein d'EDF R&D, dans le cadre du projet **SEQUOIA**, sur le thème
de l'explicabilité automatique de documents texte par **Sparse Autoencoders** (SAE).
Le cas d'usage retenu est l'analyse de mails clients reçus par EDF : un volume
important de correspondance dont le traitement (priorisation, orientation vers le bon
service, détection des réclamations et des urgences) est aujourd'hui coûteux à
automatiser de façon à la fois efficace et **interprétable** — un préalable important
dans un contexte réglementé où une décision automatisée doit pouvoir être justifiée.

Les grands modèles de langage (LLM) offrent une capacité de compréhension fine du
texte, mais leurs représentations internes restent largement opaques : une même
direction de leur espace d'activation encode typiquement plusieurs concepts
sémantiques différents (phénomène de *superposition*), ce qui empêche une lecture
directe de "ce que le modèle a compris du texte". Cette question — dans quelle mesure
un LLM "comprend" réellement le texte qu'il traite plutôt que d'en imiter la surface
statistique — est elle-même débattue ; Beckmann & Queloz (2026) soutiennent que les
avancées récentes de l'interprétabilité mécanique rendent la position purement
sceptique de moins en moins tenable, à condition d'articuler ces résultats à un cadre
théorique de la compréhension. Les Sparse Autoencoders, popularisés récemment par les
travaux d'interprétabilité mécaniste (Anthropic, DeepMind), proposent
de réapprendre une représentation parcimonieuse et de plus haute dimension dans
laquelle chaque direction ("feature") correspond, dans l'idéal, à un concept isolé et
nommable en langage naturel.

## Objectifs du stage

L'énoncé initial du stage (`Context.md`, `pdf/Offre_Stage_EDF_RD_SEQUOIA_E7S_SAE.pdf`)
fixe l'ambition de construire une plateforme d'analyse de mails permettant :

- l'indexation et la recherche par concept,
- le clustering interprétable de documents,
- la détection d'urgence et d'intention,
- la comparaison de corpus (diffing),
- la visualisation des concepts activés,
- le retrieval par propriétés,
- l'explication des décisions prises en aval de ces représentations.

avec la contrainte explicite de réutiliser au maximum l'existant plutôt que de
réimplémenter (SAELens, GemmaScope, *Interpretable Embeddings with Sparse
Autoencoders*, SAE Boost), et de documenter systématiquement les écarts avec ces
références lorsqu'une réimplémentation partielle s'avère nécessaire.

## Démarche

Le stage a suivi une progression en quatre grandes phases, détaillées dans les
chapitres suivants :

1. **Audit et fiabilisation** du pipeline existant (chargement des SAE préentraînés,
   récupération des labels Neuronpedia, précision numérique, robustesse) et validation
   de bout en bout sur un modèle réduit (Gemma-3-270M-it) avant tout passage à
   l'échelle.
2. **Diagnostic et correction** du problème central identifié en début de stage : un
   taux de succès très faible (20%) du protocole d'auto-interprétation des features
   apprises spécifiquement sur le domaine, sur le run à l'échelle complète
   (Gemma-3-12B-it) disponible au démarrage de cette phase. L'investigation a
   déterminé, par une démarche d'ablation contrôlée, que la cause n'était **pas** un
   volume d'entraînement insuffisant mais un défaut de conception du corpus
   d'entraînement.
3. **Relecture critique face à la littérature de référence** (en particulier Jiang,
   Sun et al. 2025, *Interpretable Embeddings with Sparse Autoencoders*) pour
   identifier les écarts méthodologiques restants (retrieval, clustering,
   corrélations, protocole de labellisation) et les corriger ou les documenter comme
   limites assumées.
4. **Mise à l'échelle et consolidation** : mise en place de tests de qualité de
   l'explication (fidélité, plausibilité), d'un protocole d'évaluation couvrant
   l'ensemble des capacités du dépôt sous conditions fixées, d'un dashboard interactif
   de visualisation, puis d'une ablation finale sur le **volume d'entraînement et de
   labellisation** (nombre d'époques, nombre de features jugées, largeur du SAE
   préentraîné) pour vérifier si un passage à l'échelle simple, sans changement de
   méthode, améliore les résultats.

## Plan du rapport

Le chapitre 1 positionne le projet par rapport à l'état de l'art (SAE, GemmaScope,
protocoles d'auto-interprétation). Le chapitre 2 décrit l'architecture technique mise
en œuvre. Le chapitre 3 présente la démarche expérimentale complète et ses résultats,
cœur scientifique du rapport. Le chapitre 4 dresse le bilan consolidé des erreurs
rencontrées et de leurs corrections tout au long du stage. Le chapitre 5 discute les
limites actuelles et les perspectives. Le rapport se conclut par un bilan général du
stage.
