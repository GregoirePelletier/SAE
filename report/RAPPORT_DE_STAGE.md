<div align="center">

**[Nom de l'établissement — Master 2, à compléter]**
**[Intitulé du Master / de la spécialité — à compléter]**

---

# Rapport de stage de Master 2

## Explicabilité automatique de mails clients par Sparse Autoencoders

### Application à l'analyse interprétable de la correspondance client d'EDF

---

**Auteur** : Grégoire Pelletier *(déduit de l'adresse de contact ; à confirmer)*

**Entreprise d'accueil** : EDF R&D — Projet SEQUOIA

**Maître de stage (entreprise)** : [Nom du tuteur EDF — à compléter]

**Tuteur académique** : [Nom du tuteur académique — à compléter]

**Période de stage** : [Dates de début/fin — à compléter]

**Date de rédaction** : 21 juillet 2026

</div>

---

## Remerciements

*[Section à personnaliser par l'auteur — usuellement adressée au maître de stage, à
l'équipe d'accueil, au tuteur académique, et à toute personne ayant contribué au bon
déroulement du stage.]*

---

## Résumé

Ce stage porte sur l'explicabilité automatique de mails clients d'EDF à l'aide de
Sparse Autoencoders (SAE), combinant un SAE préentraîné à grande échelle (GemmaScope-2,
sur les activations de Gemma-3-12B-it) étendu par un second SAE entraîné
spécifiquement sur le domaine, et un second pipeline indépendant fondé sur des
embeddings de phrase (F2LLM-v2). Le pipeline initial, fonctionnel de bout en bout,
présentait un taux de succès faible (20%) au test d'auto-interprétation des features
propres au domaine. Une démarche de diagnostic par ablation contrôlée a établi que ce
taux n'était pas limité par le volume d'entraînement, mais par une erreur de
conception du corpus d'entraînement (uniquement générique, sans emails originaux) —
corrigée, elle porte le taux d'interprétabilité à ~41-45%. Le stage a ensuite mis en
place des tests de qualité de l'explication document-level (fidélité par ablation,
plausibilité par choix forcé, toutes deux positives), un protocole d'évaluation
couvrant l'ensemble des capacités du dépôt sous conditions fixées, un dashboard
interactif, et une ablation finale de mise à l'échelle (largeur du SAE core,
nombre d'époques, nombre de features labellisées) pour vérifier si un passage à
l'échelle simple améliore encore les résultats sans changement de méthode.

**Mots-clés** : Sparse Autoencoders, interprétabilité mécaniste, GemmaScope,
grands modèles de langage, explicabilité, traitement automatique des mails clients,
auto-interprétation par juge LLM.

---

## Abstract

This internship addresses automatic explainability of customer emails at EDF using
Sparse Autoencoders (SAE), combining a large pretrained SAE (GemmaScope-2, on
Gemma-3-12B-it activations) extended by a second SAE trained specifically on the
target domain, alongside an independent sentence-embedding-based pipeline (F2LLM-v2).
The initial end-to-end pipeline showed a low success rate (20%) on the
domain-specific feature auto-interpretation test. A controlled-ablation diagnostic
established that this was not a training-volume limitation but a training-corpus
design flaw (generic text only, no real emails) — once fixed, the interpretability
rate rose to ~41-45%. The internship then implemented document-level explanation
quality tests (ablation-based fidelity, forced-choice plausibility, both positive), a
full-repository evaluation protocol under fixed conditions, an interactive dashboard,
and a final scale-up ablation (core SAE width, number of training epochs, number of
labeled features) to test whether simply scaling up improves results further without
any methodological change.

**Keywords**: Sparse Autoencoders, mechanistic interpretability, GemmaScope, large
language models, explainability, customer email analysis, LLM auto-interpretation.

---

## Sommaire

- Introduction générale
- Chapitre 1 — État de l'art
- Chapitre 2 — Architecture et implémentation
- Chapitre 3 — Démarche expérimentale et résultats
- Chapitre 4 — Inspection des erreurs et corrections
- Chapitre 5 — Limites et perspectives
- Conclusion générale
- Bibliographie

---


\newpage

---

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


\newpage

---

# Chapitre 1 — État de l'art

## Contexte : interprétabilité mécaniste et Sparse Autoencoders

Les réseaux de neurones profonds encodent typiquement plusieurs concepts sémantiques
dans une même direction de leur espace de représentation (*superposition*), ce qui
rend l'interprétation directe des activations difficile. Les **Sparse Autoencoders**
(SAE) répondent à ce problème en apprenant, pour un espace d'activations donné (ici le
residual stream d'un grand modèle de langage), une reprojection vers un espace de plus
haute dimension où chaque direction ("feature") est sensée correspondre à un concept
plus isolé et monosémantique, sous une contrainte de sparsité (peu de features actives
simultanément par exemple).

Formellement, pour une activation $x \in \mathbb{R}^{d}$ :

$$z = \mathrm{encode}(x), \quad \hat{x} = \mathrm{decode}(z), \quad \|z\|_0 \le k \text{ (ou pénalité L1)}$$

avec un objectif de reconstruction ($\|x - \hat{x}\|^2$) sous contrainte de parcimonie.
Ce projet utilise deux variantes :
- **JumpReLU** (SAE GemmaScope-2 préentraîné, seuil par feature appris) pour le
  Pipeline 1 (SAE "core").
- **BatchTopK + AuxK** (les $k$ activations les plus fortes du batch sont conservées,
  un terme auxiliaire ranime les features "mortes") pour les SAE entraînés from-scratch
  du projet (`ExtendedSAE`, `PhraseLevelSAE`).

## GemmaScope-2

GemmaScope ([google-deepmind/gemma-scope](https://github.com/google-deepmind/gemma-scope))
est une collection de SAE préentraînés par DeepMind sur les modèles de la famille
Gemma, à plusieurs couches et plusieurs largeurs (nombre de features). Le projet utilise
GemmaScope-2 (variante pour Gemma-3) sur le residual stream, couche 24 pour le modèle
12B. Le choix de la **largeur** du SAE (parmi 16k/65k/262k/1m disponibles) est un
arbitrage documenté empiriquement dans ce projet, sur le critère de couverture des
labels Neuronpedia (fraction des features disposant d'une explication en langage
naturel) : **65k** offre la meilleure couverture pour ce modèle (87,8%, 57 551/65 536
features labellisées), devant 16k (82,6%, 13 535/16 384), très loin devant 262k (5,3%,
13 851/262 144, confirmant une première estimation manuelle ~10 000/262 144) ; la
largeur 1m n'est pas hébergée par Neuronpedia pour ce modèle (aucune donnée
disponible). 16k a été le choix initial du projet (comparé uniquement à 262k au
démarrage du stage) ; 65k, non vérifié à l'origine, a été adopté après cette
vérification systématique pour le run de mise à l'échelle final (chapitre 3).

## Neuronpedia et l'auto-interprétation des features

[Neuronpedia](https://www.neuronpedia.org) héberge des explications en langage naturel
générées automatiquement pour les features de nombreux SAE publics, dont GemmaScope.
Ces explications servent de "vérité de référence" externe pour les features du SAE
préentraîné ("core") dans ce projet. Elles n'existent en revanche pour aucune feature
d'un SAE entraîné spécifiquement pour ce projet (l'extension `ExtendedSAE`, le
`PhraseLevelSAE`) — d'où le recours à un juge LLM local pour ces features-là.

### Protocole d'auto-interprétation (juge LLM local)

Le projet reprend le protocole *odd-one-out* (utilisé notamment dans SAEBench) combiné
à la mesure ρ_interp de Bills et al. (2023) :

1. Sélectionner les documents/tokens qui activent le plus fortement une feature donnée
   (exemples positifs) et un exemple témoin pris dans un document où la feature est
   quasi-inactive (contrôle négatif).
2. Présenter les exemples mélangés à un LLM et lui demander d'identifier l'intrus (le
   contrôle négatif) — succès binaire (`interp_score`).
3. Si succès : demander au même LLM de nommer/décrire le concept commun aux exemples
   positifs (le label final).
4. Mesurer ρ_interp : corrélation de Spearman entre un score d'intensité attribué par
   le LLM à chaque exemple et l'activation réelle mesurée — une feature bien détectée
   par le juge devrait aussi bien *classer* les exemples par intensité, pas seulement
   trouver l'intrus.

C'est ce protocole, et son taux de succès mesuré sur ce projet, qui fait l'objet du
chapitre expérimental principal (`03_experiences_et_resultats.md`).

## Positionnement du projet

Le projet combine deux approches complémentaires plutôt que de choisir l'une ou
l'autre :
- **Pipeline 1** exploite un SAE préentraîné à l'échelle (GemmaScope, entraîné sur des
  corpus massifs et généralistes) et l'étend avec un second SAE plus petit, entraîné
  spécifiquement sur le domaine (emails EDF), pour capturer les concepts propres au
  domaine que le SAE généraliste ne isole pas nécessairement comme des directions
  dédiées.
- **Pipeline 2** entraîne un SAE de bout en bout sur des embeddings de phrase d'un
  modèle plus petit (F2LLM), offrant un point de comparaison à coût de calcul
  nettement inférieur et une granularité différente (phrase plutôt que token).

Cette architecture à deux niveaux (SAE généraliste + extension spécifique au domaine)
a longtemps été documentée dans ce projet comme un apport spécifique par rapport à un
usage "out of the box" de GemmaScope ou de SAELens (`Context.md`, règle n°3 :
"Conserver `FrozenCoreResidualSAE` — spécifique au projet"). Une relecture tardive de
la littérature de référence a établi qu'il s'agit en réalité d'une implémentation de
**SAE Boost** (Koriagin et al., COLM 2025, *Teach Old SAEs New Domain Tricks with
Boosting*) — une des quatre méthodes explicitement listées dans le cadrage initial du
stage (`Context.md`, objectif n°4), marquée "non fait" dans `docs/references.md`
depuis le début : le projet l'avait en réalité déjà implémentée et validée à
l'échelle, sans jamais l'identifier ni la citer comme telle. `FrozenCoreResidualSAE`/
`ExtendedSAE` (`src/sae/frozen_core.py`) reproduit exactement leur méthode (SAE
résiduel entraîné sur l'erreur de reconstruction `e = x - x̂` d'un SAE core gelé,
sommé à l'inférence) — y compris la taille de dictionnaire résiduel (1024), identique
dans les deux cas sans que ce ne soit délibéré. Détail complet de la comparaison
(écarts de sensibilité `K_EXTRA`, budget de tokens nécessaire, baselines
alternatives jamais testées) : `RESULTS_TESTS.md` §18.

## Perspectives critiques : les SAE apprennent-ils réellement des features signifiantes ?

Une partie de la littérature récente questionne directement la prémisse sur laquelle
repose ce projet. *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random
Baselines?* (Korznikov et al., 2026) montre que des SAE dont des composants clés
(en particulier le décodeur) sont **figés à une initialisation aléatoire, jamais
entraînés**, égalent des SAE réellement entraînés sur les métriques standard du
domaine : interprétabilité automatique, sparse probing, et édition causale (RAVEL).
Sur un cas synthétique à vérité terrain connue, ils montrent également qu'un SAE peut
atteindre une variance expliquée élevée (71%) tout en ne recouvrant que 9% des
véritables features génératrices. Leur conclusion : la reconstruction et
l'interprétabilité mesurées isolément ne suffisent pas à prouver qu'un SAE a appris
une décomposition en features réellement significative plutôt qu'un simple ajustement
de l'encodeur à des directions arbitraires.

Ce résultat interroge directement le protocole d'auto-interprétation odd-one-out
utilisé dans ce projet (ci-dessus) et les sondes de classification en aval
(`clf_acc_sae`, `03_experiences_et_resultats.md` §5.4) : nos taux mesurés
(45,3% d'interprétabilité, >90% de classification) sont-ils réellement dus à un
apprentissage de features significatives, ou un décodeur figé à l'initialisation
obtiendrait-il des scores comparables ? Ce projet reproduit leur protocole de sanity
check sur l'extension du Pipeline 1 (`FrozenDecoderExtendedSAE`,
`src/sae/frozen_core.py`) — résultat **nuancé** : l'interprétabilité odd-one-out
résiste bien (45,3% entraîné vs 29,3% décodeur figé aléatoire, écart significatif)
mais la classification en aval y résiste beaucoup moins (93,5% vs 91,2%),
répliquant partiellement le constat du papier. Méthode et résultats détaillés en
`RESULTS_TESTS.md` §19, `report/03_experiences_et_resultats.md` §11.

## Taxonomie des méthodes d'explication et d'évaluation

*A Survey on Sparse Autoencoders* (Shu, Wu, Zhao et al., EMNLP 2025 Findings)
distingue deux familles de méthodes d'explication des features SAE — **input-based**
(quel exemple d'entrée active la feature — nos protocoles odd-one-out et
labellisation contrastive directe, ci-dessus) et **output-based** (quel changement de
génération produit l'amplification de la feature — le *steering*) — ainsi que deux
familles de métriques d'évaluation — **structurelles** (fidélité de reconstruction :
NMSE, FVE, L0) et **fonctionnelles** (utilité en aval : nos sondes de classification,
tests de fidélité/plausibilité, chapitre 3 §8). Ce projet couvre bien les deux
familles de métriques, mais uniquement le versant *input-based* des méthodes
d'explication : le steering (`steer_activations`/`steer_and_decode`,
`src/sae/sae_shared.py`) existe déjà dans le dépôt mais n'a jamais été évalué comme
méthode d'explication à part entière — piste documentée au chapitre 5.


\newpage

---

# Chapitre 2 — Architecture et implémentation

*(Synthèse orientée rapport — référence technique complète dans `docs/architecture.md`.)*

## Vue d'ensemble

Le système implémente deux pipelines d'analyse interprétable de mails clients EDF,
partageant la même infrastructure de corpus, de stockage et d'évaluation :

```
                    ┌─────────────────────────────────────┐
                    │        Corpus principal              │
                    │  Mails EDF originaux + variantes      │
                    │  augmentées (13 axes de perturbation) │
                    └───────────────┬───────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
     Pipeline 1 (token-level)                     Pipeline 2 (phrase-level)
     Gemma-3-12B-it → SAE GemmaScope-2            F2LLM-v2 → PhraseLevelSAE
     (préentraîné) + extension FrozenCore         (entraîné from-scratch)
     (entraînée from-scratch sur le résidu)
              │                                             │
              ▼                                             ▼
     Max-pool documentaire                        Max-pool documentaire
              │                                             │
              ▼                                             ▼
     Labels : Neuronpedia (core) +                Labels : juge LLM local
     juge LLM local (extension)                   (odd-one-out)
```

## Choix architecturaux clés

### Une extension spécifique au domaine plutôt qu'un SAE unique

Le SAE GemmaScope préentraîné ("core") capture des concepts généraux (appris sur des
corpus massifs, multi-domaines). Pour capturer des concepts spécifiques aux emails EDF
qui ne correspondraient à aucune direction dédiée du SAE core, le Pipeline 1 ajoute une
**extension** (`FrozenCoreResidualSAE`/`ExtendedSAE`) : un second SAE, plus petit
(1024 features, 32 actives simultanément), qui encode le **résidu** — ce que le SAE
core ne reconstruit pas. Le SAE core reste gelé (jamais réentraîné) ; seule l'extension
est entraînée, sur le corpus du projet.

### Nature du corpus "original" (`Mails.tsv`) — précision importante

Le corpus `Mails.tsv`, désigné dans ce rapport par commodité comme les mails
**"originaux"**, n'est **pas** de la correspondance client authentique : il s'agit
d'un jeu de données déjà synthétique, produit par un travail antérieur du
laboratoire EDF R&D (non réalisé pendant ce stage, ni par les mêmes personnes). Les
variantes **augmentées** (`augmented_mails.jsonl`) sont, elles, générées pendant ce
stage (`scripts/run_augmentation.py`, Gemma-3-12B-it) à partir de ces mails
"originaux". **Aucune des deux couches du corpus n'est donc constituée de données
réelles au sens strict** — le terme "original" désigne uniquement leur statut de
donnée d'entrée (antérieure et externe à ce stage) par rapport aux variantes
augmentées qui en dérivent, pas leur authenticité. Ce rapport évite désormais le
terme "réel"/"réels" pour ce corpus, employé par erreur dans les premières phases
de rédaction (cf. chapitre 4 pour la correction et sa justification).

### Corpus d'entraînement : principal vs secondaire

Distinction introduite/formalisée dans cette phase du stage (cf. chapitre suivant) :
- Le corpus qui **entraîne** l'extension et le `PhraseLevelSAE` doit être représentatif
  du domaine cible (emails) pour que les features apprises soient interprétables dans
  ce domaine.
- Un corpus generic (energy/sports/support, substitut public) reste utile comme
  **banc d'essai** pour une capacité générale du système (diffing cross-domaine) mais
  ne doit pas se substituer au corpus principal dans l'entraînement.

### Split train/test group-aware

Les variantes augmentées d'un même mail sont des quasi-duplicata sémantiques de
l'original (même contenu factuel, perturbation contrôlée d'un seul axe à la fois). Un
split aléatoire au niveau des lignes laisserait fuiter des variantes d'un mail de test
dans le train (ou l'inverse), biaisant artificiellement à la hausse les métriques de
classification/silhouette. Le split est donc effectué au niveau du **mail d'origine** :
un mail et toutes ses variantes tombent du même côté.

### Stockage sparse des activations token-level

Une activation SAE dense par token, à la largeur historique 262k, coûterait ~400 Mo
par document. Le projet stocke les activations token-level au format CSR (compressed
sparse row, implémenté en tenseurs torch plutôt qu'avec SciPy pour rester
GPU-compatible), ramenant le coût à quelques centaines de Ko par document — nécessaire
pour traiter des corpus de dizaines de milliers de documents sans épuiser la mémoire.

### Précision numérique (bf16)

Gemma-3 présente des activations de norme très élevée ("massive activations",
documentées dans la littérature) dans son residual stream. Une précision fp16 (plage
dynamique jusqu'à ~65 504) overflow silencieusement sur ces valeurs, corrompant tout
l'entraînement en aval sans erreur explicite (perte `NaN` dès la première epoch, un des
bugs les plus coûteux à diagnostiquer de ce projet — cf. `Context.md`). Le projet
utilise bf16 par défaut partout (même en local), qui partage la plage d'exposant de
fp32.

## Infrastructure de calcul

Cluster SLURM à 3 partitions GPU (a100, h100, h100-bis), sans accès réseau direct
depuis les nœuds de calcul — toutes les dépendances (modèles, données) doivent être
prépositionnées sur disque avant soumission d'un job. Cette contrainte a orienté
plusieurs choix : cache local des labels Neuronpedia (pas d'appel réseau au runtime),
environnement Python déjà provisionné sur disque (`.venv/bin/python` plutôt que `uv
run`, qui tenterait de re-résoudre l'environnement en ligne).


\newpage

---

# Chapitre 3 — Démarche expérimentale et résultats

## 3.1. Problématique

Le pipeline complet (extraction d'activations → SAE → labellisation) fonctionnait de
bout en bout sans erreur, mais produisait un taux de succès faible au test
d'auto-interprétation odd-one-out des features d'extension (celles qui ne sont pas
couvertes par Neuronpedia et dépendent donc entièrement du juge LLM local pour être
labellisées). Sur le dernier run complet disponible avant cette phase du stage
(`results_v9_full`, Gemma-3-12B-it, 10 features jugées) : seules 2 features sur 10
(20%) passaient le test.

Question posée : **ce taux faible est-il dû à un budget d'entraînement (nombre de
tokens) insuffisant pour l'extension SAE, ou à un autre facteur ?**

## 3.2. Diagnostic

### 3.2.1. Élimination de l'hypothèse "features mortes"

Un budget d'entraînement insuffisant se traduirait typiquement par des features
d'extension qui ne s'activent jamais (`dead_feature`), auquel cas le juge ne peut même
pas être interrogé (`odd_one_out_judge` retourne directement `dead_feature` si moins de
3 exemples positifs sont disponibles). Or l'inspection du run `results_v9_full` montre
**0 feature morte sur les 10 jugées** : les features s'activaient bel et bien, avec des
exemples positifs disponibles. L'échec du test n'était donc pas un problème de features
inactives.

### 3.2.2. Inspection qualitative des exemples présentés au juge

L'inspection directe des `pos_examples` stockés pour les features non interprétables a
révélé le problème : les neuf exemples "positifs" présentés au juge pour une même
feature n'avaient, dans plusieurs cas, **aucun concept commun identifiable** — par
exemple un extrait sur un rappel produit iPad, un extrait sur le système carcéral
norvégien, un extrait de recette de cuisine et un extrait sur l'agriculture canadienne,
présentés ensemble comme activant fortement la même feature. Le test odd-one-out ne
peut, par construction, pas réussir dans ce cas : il n'y a pas de concept partagé à
partir duquel identifier l'intrus.

### 3.2.3. Cause racine

La lecture du code d'assemblage du corpus (`src/sae/saev5.py`, bloc `__main__`) a
montré que le corpus utilisé pour échantillonner le réservoir de résidus servant à
entraîner l'extension (`N_TOKENS_EXTRA_TRAIN` tokens tirés de `train_texts`) était
constitué **exclusivement** de textes génériques (FineWeb-2/Wikipedia filtrés par
mots-clés sur trois domaines substituts : énergie, sport, support client). Les mails
réels et leurs variantes augmentées étaient chargés séparément (`email_texts`) et
utilisés **uniquement après l'entraînement**, pour une visualisation UMAP — jamais vus
par le SAE d'extension pendant son entraînement.

Autrement dit : le SAE d'extension apprenait des directions représentant des concepts
Wikipedia génériques et hétérogènes, jamais des concepts liés au contenu réel des
emails EDF. Les "exemples positifs" incohérents observés en 2.2 ne sont pas une
anomalie du juge, mais une conséquence directe et attendue de ce corpus
d'entraînement.

## 3.3. Correction

Deux changements ont été apportés au pipeline (`src/data/preparation.py`,
`src/sae/saev5.py`) :

1. **Nouveau corpus principal** : mails originaux (`Mails.tsv`) + variantes augmentées
   acceptées (`augmented_mails.jsonl`, 39 949 variantes issues de 13 axes de
   perturbation contrôlée — émotion, registre, orthographe, urgence). Ce corpus devient
   celui qui entraîne l'extension SAE et le `PhraseLevelSAE`. Split train/test
   **group-aware** par mail d'origine, pour éviter qu'une variante augmentée d'un mail
   de test ne fuite dans le train.
2. **Corpus secondaire** : le corpus generic (énergie/sport/support) est conservé mais
   réduit et cantonné à un usage post-hoc (démonstration préexistante de diffing
   cross-domaine), sans plus jamais participer à l'entraînement.

## 3.4. Protocole de validation

Trois runs à l'échelle complète sur Gemma-3-12B-it, tous avec le nouveau corpus
principal (emails + augmentés, ~41 200 documents d'entraînement / ~2 200 documents de
test), afin d'isoler l'effet du **volume** de tokens d'entraînement de l'effet du
**domaine** du corpus (déjà corrigé identiquement dans les trois runs) :

| Run | `N_TOKENS_EXTRA_TRAIN` | Durée d'exécution |
|---|---|---|
| Ablation volume bas | 100 000 | 3h11min37s |
| **Run principal** (budget par défaut) | 500 000 | 3h01min53s |
| Ablation volume haut | 2 000 000 | 2h21min01s |

Le nombre de features jugées par run a été porté de 10 à **150** (`N_FEATURES_TO_LABEL`)
pour disposer d'une puissance statistique correcte sur le taux d'interprétabilité
observé (avec n=10, l'incertitude sur un taux observé est trop large pour conclure).

## 3.5. Résultats

### 3.5.1. Taux d'interprétabilité (test odd-one-out)

| Corpus | `N_TOKENS_EXTRA_TRAIN` | n features jugées | Features mortes | Taux d'interprétabilité |
|---|---|---|---|---|
| Generic (avant correction) | 500 000 | 10 | 0 | **20,0%** (2/10) |
| Emails + augmentés | 100 000 | 150 | 0 | **40,7%** (61/150) |
| Emails + augmentés | 500 000 | 150 | 0 | **45,3%** (68/150) |
| Emails + augmentés | 2 000 000 | 150 | 0 | **44,7%** (67/150) |

L'écart-type binomial attendu sur un taux observé avec n=150 est d'environ 4,1 points
de pourcentage (approximation de Wald à 95%, soit un intervalle de confiance
d'environ ±8 points). Les trois taux mesurés à corpus identique (40,7% / 45,3% / 44,7%)
sont statistiquement indistinguables les uns des autres au regard de cette incertitude,
malgré un facteur 20 entre le budget de tokens le plus faible et le plus élevé testés.

### 3.5.2. Interprétation

- **Corriger le domaine du corpus (generic → emails), à volume comparable (500 000
  tokens), plus que double le taux d'interprétabilité (20,0% → 45,3%).** C'est le
  facteur dominant identifié dans cette investigation.
- **Faire varier le volume de tokens d'un facteur 20, à domaine fixé (emails), ne
  produit aucun effet mesurable.** Le SAE d'extension n'est pas limité par le volume de
  tokens dès 100 000 tokens, une fois le corpus correctement ciblé sur le domaine ; le
  porter à 2 000 000 n'apporte aucun gain supplémentaire observable.
- **Réponse à la question posée** : le taux de détection de l'intrus faible observé
  initialement n'était **pas** un problème de volume d'entraînement, mais un problème
  de **contenu/domaine du corpus** d'entraînement — les emails, cible réelle du
  projet, n'entraient jamais dans les données servant à entraîner l'extension SAE.

### 3.5.3. Qualité des labels obtenus

Contraste direct entre les labels obtenus avant et après correction, pour les features
qui passent le test :

- **Avant** (corpus generic) : labels générés à partir d'extraits Wikipedia sans lien
  avec le domaine (ex. artefacts de formatage — listes, ponctuation).
- **Après** (corpus emails) : `Réclamations Clients`, `Litiges Factures`, `Résiliation
  Énergie`, `Menace Résiliation`, `Demande Urgente`, `Problèmes énergie`, `Insatisfaction
  client` — des concepts directement alignés avec les objectifs métier du projet
  (détection d'urgence, détection d'intention, réclamations).

### 3.5.4. Résultat additionnel : séparabilité linéaire des axes de perturbation

Une sonde de classification logistique a été ajoutée pour mesurer si les codes latents
du SAE permettent de séparer linéairement les 14 classes du corpus principal (13
combinaisons axe/niveau de perturbation + "original"). Résultats (run à 100 000
tokens, premier disposant du correctif nécessaire pour la classification multi-classe,
cf. §6) :

| Pipeline | Précision de classification (5-fold) |
|---|---|
| Pipeline 1 (Gemma-3 + GemmaScope) | 93,5% |
| Pipeline 2 (F2LLM + PhraseLevelSAE) | 79,3% |

Ce résultat, obtenu comme sous-produit de la validation du corpus, est directement
pertinent pour les objectifs du projet (détection d'urgence, détection d'intention) :
il indique que les représentations latentes du SAE encodent l'information nécessaire à
ces tâches de façon linéairement séparable, sur un corpus qui simule des variations
réalistes de ton et d'urgence dans les emails clients.

## 3.6. Bug corrigé pendant la validation

La sonde de classification multi-classe (§5.4) a d'abord échoué systématiquement
(exception silencieusement attrapée, métrique retournée à `NaN`) : la bibliothèque
scikit-learn utilisée (`LogisticRegression(solver="liblinear")`) ne supporte, dans ses
versions récentes, que la classification binaire. Corrigé en sélectionnant
dynamiquement le solveur (`lbfgs`, qui supporte nativement le cas multinomial) au-delà
de deux classes, sans changer le comportement du probe binaire préexistant
(énergie/sport) qui continue d'utiliser `liblinear`. Deux des trois runs de l'ablation
volume ont démarré leur exécution avant que ce correctif ne soit disponible ; le
troisième (démarré après, une fois sorti de la file d'attente SLURM) a permis d'obtenir
les valeurs du §5.4.

## 3.7. Suites données au diagnostic

Deux analyses complémentaires ont été menées après la validation du §5, pour répondre
plus directement aux objectifs métier du stage et à la limite identifiée au §5.4/§7
(taux résiduel non expliqué) — toutes deux réutilisent des activations déjà en cache,
sans calcul GPU supplémentaire pour la seconde.

### 3.7.1. Le résidu non-interprété est-il dû au protocole de jugement lui-même ?

Le protocole odd-one-out (`odd_one_out_judge`) ne prend qu'une seule décision greedy
par feature. Pour tester sa robustesse à l'ordre de présentation des exemples (un
biais de position bien documenté chez les LLM en contexte de choix multiple), la même
question a été répétée 5 fois par feature, avec un ordre de mélange différent à
chaque fois, pour les 150 features déjà jugées du run principal
(`scripts/judge_robustness_check.py`, job SLURM 40672 — recharge uniquement le modèle
comme juge, aucune réextraction d'activations).

| Métrique | Valeur |
|---|---|
| Taux d'interprétabilité, décision unique (référence) | 45,3% |
| Taux d'interprétabilité, vote majoritaire sur 5 répétitions | 48,7% |
| Features dont la décision change selon l'ordre | 31,3% (47/150) |
| Taux d'accord moyen entre les 5 répétitions | 80,3% |
| Features avec décision unanime (0/5 ou 5/5) | 30,7% (46/150) |

Le taux agrégé ne change que marginalement, mais la fiabilité de la décision prise
pour une feature individuelle est faible : moins d'un tiers des features obtiennent
une décision unanime sur 5 répétitions identiques (mêmes exemples, ordre différent).
**Une partie substantielle du taux d'échec observé au §5 reflète donc le bruit du
protocole de jugement plutôt qu'un défaut réel des features testées.**

### 3.7.2. Le SAE prédit-il l'urgence et l'intention sur des mails originaux ?

Le résultat du §5.4 (séparabilité des axes d'augmentation synthétiques) a été
complété par un test sur des labels **indépendants du corpus augmenté** : des labels
faibles par expression régulière, déjà calculés sur le texte brut des mails originaux
(`src/data/dataset.py::INTENT_KEYWORDS_FR` — réclamation, résiliation, remboursement,
information, urgence), appliqués aux 3 300 mails originaux du split d'entraînement
(`scripts/intent_urgency_probe.py`, zéro calcul GPU — réutilise
`p1_all_doc_acts_ext_d1024.pt` déjà en cache).

| Intention | Prévalence | Précision sonde | Baseline (classe majoritaire) | Écart |
|---|---|---|---|---|
| Urgence | 29,3% | 97,7% | 70,7% | **+27,0 pts** |
| Réclamation | 55,1% | 97,7% | 55,1% | **+42,6 pts** |
| Information | 18,2% | 87,8% | 81,8% | +6,0 pts |
| Remboursement | 14,5% | 84,5% | 85,5% | −1,0 pt |

Ce résultat répond directement aux deux objectifs "détection d'urgence" et
"détection d'intentions" énoncés dans le cadrage initial du projet
(`Context.md`, section "Objectif") : les codes latents du SAE séparent très
nettement l'urgence et la réclamation, sur des mails originaux non augmentés, avec un
gain net important sur la baseline naïve. Le remboursement ne bat pas sa baseline
(déjà forte du fait du déséquilibre de classe, 85,5% de négatifs) — à interpréter
comme une limite du label faible par regex pour cette catégorie plutôt que comme un
échec du SAE, sans donnée annotée manuellement pour trancher entre les deux
hypothèses.

## 3.8. Qualité de l'explication document-level : fidélité et plausibilité

Question distincte des sections précédentes (qui évaluent une feature isolée ou une
capacité globale du corpus) : pour UN document donné, l'explication produite par le
pipeline (les features les plus actives et leurs labels) est-elle une bonne
explication ? Deux propriétés indépendantes ont été testées.

### 3.8.1. Fidélité (l'explication reflète-t-elle ce qui pilote réellement la décision ?)

Test par ablation (`scripts/explanation_fidelity_test.py`) : sur 200 mails originaux par
intention (urgence, réclamation, information, remboursement), correctement classés
par une sonde logistique, ablation des 10 features les plus contributives à la
décision vs 10 features actives aléatoires vs les 10 moins contributives.

| Intention | Chute top-10 | Chute random-10 | Ratio top/random |
|---|---|---|---|
| Réclamation | 0,576 | ~0,000 | 576 225× |
| Remboursement | 0,9997 | 0,0009 | 1 058× |
| Information | 0,9998 | 0,0040 | 251× |
| Urgence | 0,612 | ~0,000 | 42 837× |

Résultat sans ambiguïté : les features désignées comme explication portent
réellement la décision (leur ablation fait s'effondrer la prédiction), contrairement
à des features actives choisies au hasard (effet quasi nul). L'explication n'est pas
une justification a posteriori déconnectée du mécanisme réel.

### 3.8.2. Plausibilité (un lecteur trouve-t-il l'explication convaincante ?)

Test par choix forcé au niveau document (`scripts/explanation_plausibility_test.py`,
juge Gemma-3-12B-it — un jugement comparatif comme l'odd-one-out plutôt qu'une
auto-évaluation de confiance isolée, dont la fiabilité limitée a été confirmée par
ailleurs, cf. §7.1 et `RESULTS_TESTS.md` §15.4) : sur 60 mails
réels, le juge choisit entre l'ensemble réel des concepts les plus actifs et un
ensemble de concepts tirés au hasard, sans savoir lequel est réel.

**Résultat : 71,7% de choix corrects (43/60) contre 50% attendu au hasard**
(z ≈ 3,4, p < 0,001) — l'explication a une valeur perçue réelle, significativement
au-dessus du hasard, mais loin d'être parfaite : dans 28,3% des cas le juge préfère le
décoy aléatoire, cohérent avec le taux d'interprétabilité résiduel (~45-55%) mesuré
par ailleurs.

### 3.8.3. Protocole d'évaluation complet du dépôt

Ces deux tests s'inscrivent dans un protocole plus large couvrant l'ensemble des
méthodes du dépôt sous conditions fixées (`docs/evaluation_protocol.md`) : 16
capacités recensées (reconstruction, labellisation par les deux protocoles,
robustesse du jugement, séparabilité synthétique et réelle, fidélité/plausibilité de
l'explication, retrieval, corrélations, diffing, choix du backbone d'embedding), avec
pour chacune sa commande de reproduction et la méthode alternative à laquelle elle
est comparée. Un script de consolidation (`scripts/consolidate_evaluation_report.py`)
assemble automatiquement tous les résultats disponibles d'un run en un rapport
unique, également exposé dans le dashboard Streamlit.

Un résultat notable de ce passage en revue systématique : la comparaison du backbone
d'embedding du Pipeline 2 (F2LLM-v2-80M vs -330M, "assez grand") donne un résultat
**mixte** — le modèle plus grand reconstruit légèrement mieux (NMSE −7,5%) et sépare
un peu mieux le corpus de diffing générique (+2 points), mais sépare légèrement MOINS
bien les axes email (−2,2 points), la métrique la plus proche des objectifs
métier. Aucun écart n'est de l'ordre d'un problème majeur ; pas de justification
claire pour préférer l'un à l'autre sur ce projet.

## 3.9. Limites de cette investigation

- Le taux d'interprétabilité obtenu après correction (~41-45%) reste loin de 100% :
  environ 55 à 59% des features d'extension restent non interprétables par le juge même
  sur le corpus corrigé. Cette investigation a établi que ce résidu n'est **pas**
  expliqué par le volume de tokens (testé jusqu'à 2 000 000, sans effet) ; les causes
  possibles restantes (robustesse du protocole de jugement lui-même, qualité du
  contrôle négatif, capacité architecturale de l'extension) n'ont pas été testées dans
  le temps disponible et sont proposées comme pistes de poursuite
  (cf. `04_limites_et_perspectives.md`).
- Les trois runs de l'ablation volume n'ont pas partagé de cache d'extraction
  d'activations (chaque run réextrait les activations du modèle 12B depuis zéro) : un
  choix délibéré pour garantir qu'aucune contamination de cache entre configurations
  différentes ne puisse biaiser la comparaison, au prix d'un coût de calcul plus élevé
  (~8h30 GPU cumulées pour les trois runs plutôt qu'un partage possible de l'étape
  d'extraction, identique entre les trois configurations).

## 3.10. Ablation de mise à l'échelle du volume d'entraînement et de labellisation (v12)

Question posée en fin de stage, dans le même esprit que l'ablation du §4/§5 mais sur
un axe différent : une fois le domaine du corpus corrigé (§3) et le résidu
non-interprété partiellement expliqué par le bruit du protocole de jugement (§7.1),
un passage à l'échelle du **volume d'entraînement et de labellisation**, sans aucun
autre changement de méthode, améliore-t-il la proportion de features labellisées
avec succès et les différents scores de reconstruction/séparabilité ?

Trois leviers ont été augmentés simultanément par rapport au run principal du §4
(`results_v10_emails_main`), dans un nouveau run unique (`results_v12_scaled_65k`)
regroupant l'ensemble des analyses du dépôt (toutes les capacités du protocole
d'évaluation, §8.3) :

| Paramètre | Run principal (§4) | Run v12 (échelle) | Facteur |
|---|---|---|---|
| Largeur du SAE core (couverture Neuronpedia) | 16k (82,6%, 13 535 labels) | **65k (87,8%, 57 551 labels)** | — (meilleure couverture, cf. Chapitre 1) |
| `EPOCHS_EXTRA` (SAE d'extension, Pipeline 1) | 10 | **40** | ×4 |
| `EPOCHS` (Phrase-Level SAE, Pipeline 2) | 30 | **100** | ×3,3 |
| `N_FEATURES_TO_LABEL` (features jugées) | 150 | **600** | ×4 |
| `N_TOKENS_EXTRA_TRAIN` | 500 000 | 500 000 (inchangé) | — (déjà démontré non limitant, §5.2) |
| Backbone Pipeline 2 | F2LLM-v2-80M | F2LLM-v2-330M | (condition fixée du protocole d'évaluation, §8.3) |

### 3.10.1. Résultats du run combiné

| Métrique | Run principal (16k) | Run v12 (65k, échelle) |
|---|---|---|
| Taux d'interprétabilité (odd-one-out) | 45,3% (68/150) | **53,7% (322/600)** |
| Taux d'interprétabilité, vote majoritaire (5 répétitions) | 48,7% | 55,5% |
| Accord moyen entre 5 répétitions du juge | 80,3% | 79,3% (stable) |
| `clf_acc_email_axes` (Pipeline 1, 14 classes) | 93,5% | 91,9% |
| `clf_acc_email_axes` (Pipeline 2, 14 classes) | 79,3%/77,2% (80M/330M) | 76,7% |
| NMSE Pipeline 2 | 0,0745 (80M) / 0,0689 (330M) | 0,0667 |

Le taux d'interprétabilité global progresse nettement (45,3%→53,7%), mais ce run
combine trois leviers à la fois (largeur, époques, nombre de features jugées) — cf.
§10.3 pour leur décomposition. Les scores de classification/séparabilité (déjà très
élevés, >90% pour Pipeline 1) ne progressent pas et reculent même très légèrement,
cohérent avec un plafond déjà proche pour cette tâche plutôt qu'un signal de
dégradation.

### 3.10.2. Le rang par magnitude n'est pas un bon proxy de l'interprétabilité

Analyse à coût nul (aucun calcul GPU, relecture de l'ordre de sélection déjà en
cache) pour savoir si les 450 features supplémentaires labellisées par le scale-up
diluent la statistique ou apportent un signal réel :

| Sous-ensemble (rang par magnitude d'activation) | n | Taux d'interprétabilité |
|---|---|---|
| Rang 1-150 (budget du run principal) | 150 | 44,0% |
| Rang 151-600 (apport du scale-up) | 450 | **56,9%** |

Résultat notable : le sous-ensemble de tête (rang 1-150) donne un taux
statistiquement indiscernable du run principal à 16k (44,0% vs 45,3%, écart bien en
deçà de l'incertitude binomiale à n=150) — un bon signe de cohérence entre les deux
runs. Mais les features de rang inférieur (151-600), moins actives en moyenne, sont
en réalité **plus** interprétables que celles de tête. La magnitude d'activation
moyenne n'est donc pas un proxy fiable de l'interprétabilité potentielle d'une
feature : restreindre la labellisation aux features de plus forte magnitude exclut
systématiquement des candidates au moins aussi bonnes, voire meilleures.

### 3.10.3. Bug trouvé pendant l'analyse : chemin de labels figé sur 16k

Le test de plausibilité (§8.2) donnait un résultat fortement dégradé sur ce run
(56,7% contre 71,7% sur le run principal), en contradiction apparente avec la hausse
du taux d'interprétabilité (§10.1). Diagnostic : `explanation_plausibility_test.py`
(et 3 scripts analogues) chargeait un chemin de labels Neuronpedia **figé sur la
largeur 16k**, indépendamment de la largeur réellement utilisée par le run (ici
65k) — pour toute feature d'index < 16 384, le label affiché au juge comme "réel"
appartenait en fait à une feature totalement différente du dictionnaire 16k
(certaines de ces entrées 16k étant elles-mêmes des transcriptions de raisonnement
non nettoyées côté Neuronpedia, 0,35% du fichier). Corrigé (chemin dérivé de
`SAE_ID` comme partout ailleurs dans le dépôt) et test recalculé : **88,3% (53/60)**
une fois corrigé, contre 71,7% sur le run principal — une réelle et forte
amélioration (+16,6 points), cohérente avec un catalogue de features labellisées
bien plus riche (65k core à 87,8% de couverture + 600 features d'extension jugées,
contre 16k à 82,6% + 150 seulement) pour construire l'ensemble "réel" présenté au
juge. Ce bug n'affectait que ce test (qui juge directement le texte du label) ; le
test de fidélité, qui ablate par index et n'utilise le label que pour l'annotation
cosmétique des exemples exportés, restait numériquement valide (recalculé par
prudence, résultat dans le même ordre de grandeur). Détail complet dans
`RESULTS_TESTS.md` §17.3/17.6.

### 3.10.4. Décomposition largeur / époques / capacité (ablations isolées)

Trois runs à facteur unique, isolant respectivement la largeur du SAE core, le
nombre d'époques et la capacité de l'extension (`D_EXTRA`/`K_EXTRA`, toutes choses
égales par ailleurs, `N_FEATURES_TO_LABEL=150` dans les trois cas), permettent
d'attribuer la hausse du §10.1 à sa cause plutôt qu'à l'effet combiné des quatre
leviers — même démarche que l'ablation de volume de tokens du §5 :

| Run | Largeur | Époques | Capacité extension | Interprétabilité |
|---|---|---|---|---|
| Run principal (référence) | 16k | 10/30 | 1024/32 | 45,3% (68/150) |
| Largeur seule | 65k | 10/30 | 1024/32 | 43,3% (65/150) |
| Époques seules | 16k | 40/100 | 1024/32 | 41,3% (62/150) |
| Capacité seule | 16k | 10 | 2048/64 | 40,0% (60/150) |
| Combiné (tranche rang 1-150, §10.2) | 65k | 40/100 | 1024/32 | 44,0% (66/150) |

Les trois ablations isolées donnent des taux statistiquement indistinguables du run
principal (écart-type binomial attendu ≈4,1 points à n=150) : **aucun des trois
leviers pris isolément n'améliore le taux d'interprétabilité odd-one-out**, un
résultat qui rejoint celui déjà établi pour le volume de tokens (§5.2). La hausse
observée sur le run combiné (45,3%→53,7%, §10.1) s'explique donc presque entièrement
par l'effet de composition du §10.2 (les features de rang inférieur sont mieux
interprétées), pas par une meilleure qualité d'entraînement due à la largeur, aux
époques ou à la capacité. **Conclusion sur la question posée en tête de ce
chapitre** : scaler la pipeline améliore réellement la plausibilité de l'explication
(§10.3) et la richesse du catalogue de features labellisées disponibles, mais pas le
taux brut d'interprétabilité odd-one-out — celui-ci reste gouverné par le domaine du
corpus (§3-5) et par le protocole de jugement lui-même (§7.1), pas par le volume
d'aucun des quatre paramètres d'échelle testés à ce jour (tokens, largeur, époques,
capacité).

## 3.11. Sanity check : le protocole d'évaluation distingue-t-il un SAE entraîné d'un décodeur aléatoire ?

Question posée par la lecture critique de Korznikov et al. (2026, *Sanity Checks
for Sparse Autoencoders*, chapitre 1) : leurs résultats montrent qu'un SAE dont le
décodeur est figé à une initialisation aléatoire (jamais entraîné) égale un SAE
réellement entraîné sur interprétabilité automatique, sparse probing et édition
causale — remettant en cause la validité de ces métriques comme preuve d'un
apprentissage de features significatif. Reproduit sur ce projet
(`FrozenDecoderExtendedSAE`, `src/sae/frozen_core.py`) : même conditions que le run
principal (16k, 150 features jugées, `EPOCHS_EXTRA=10`, `D_EXTRA=1024`/`K_EXTRA=32`),
seul le décodeur de l'extension reste figé aléatoire (`requires_grad_(False)`),
l'encodeur s'entraîne normalement.

| Métrique | Run principal (décodeur entraîné) | Frozen Decoder (décodeur figé) | Écart (significativité) |
|---|---|---|---|
| Interprétabilité odd-one-out | 45,3% (68/150) | 29,3% (44/150) | −16,0 points (z=2,91, p<0,01) |
| Classification en aval (14 classes) | 93,5% | 91,2% | −2,3 points (z=2,86, p<0,01) |

**Résultat nuancé** : sur l'interprétabilité odd-one-out, l'écart est net et
statistiquement solide — contrairement au résultat du papier, notre protocole
distingue clairement un décodeur entraîné d'un décodeur aléatoire. Mais sur la
classification en aval, l'écart est réel tout en restant faible en proportion du
signal total : un décodeur purement aléatoire capture déjà 91,2% de précision, la
quasi-totalité de ce qu'atteint le décodeur entraîné (93,5%) — ce volet réplique
largement le constat du papier sur le sparse probing. **Conséquence pour la lecture
des résultats de ce rapport** : le taux d'interprétabilité odd-one-out (§3-10) reste
une preuve solide d'apprentissage de features significatives ; la sonde de
classification en aval (§5.4), en revanche, doit être lue avec prudence — un score
élevé ne suffit pas à lui seul à prouver un apprentissage réel, une fraction
substantielle du signal étant déjà disponible avec un décodeur aléatoire de taille
comparable. Les tests de fidélité et de plausibilité (§8), de nature causale
différente (ablation directe des features, jugement humain-like sur le document
entier), ne sont pas concernés par cette réserve. Détail complet et calculs de
significativité : `RESULTS_TESTS.md` §19.

## 3.12. Ablation de variance de seed

Question posée par la littérature (*Unstable Features, Reproducible Subspaces*,
arXiv:2606.12138 ; *Toward Identifiable Sparse Autoencoders*, arXiv:2605.31245) :
les features individuelles d'un SAE varient-elles substantiellement d'un seed
d'entraînement à l'autre, à corpus et hyperparamètres identiques ? `SEED` a été
découplé du split train/test (`CORPUS_SPLIT_SEED`, nouveau, défaut 42 inchangé)
pour isoler la variance d'entraînement du SAE de toute variance de corpus. Run à
`SEED=123` (sinon identique au run principal) :

| Métrique | SEED=42 | SEED=123 | Écart |
|---|---|---|---|
| Interprétabilité odd-one-out | 45,3% (68/150) | 47,3% (71/150) | +2,0 points (non significatif) |
| `clf_acc_email_axes` | 93,5% | 91,3% | −2,2 points |
| Recouvrement exact des labels interprétables | — | 22/78 = 28,2% | |

Le taux agrégé d'interprétabilité est stable entre seeds, mais seulement 28,2% des
labels sont des chaînes identiques entre les deux runs — les deux seeds recouvrent
des thèmes similaires (adresses, contrats énergie, réclamations, coupures,
urgence) mais rarement la même feature exacte. **Conséquence pour la lecture de ce
rapport** : les features individuelles citées en exemple sont représentatives d'une
catégorie de concepts récurrente, pas des atomes stables et reproductibles du
dictionnaire ; seul le taux agrégé d'interprétabilité doit être lu comme une mesure
fiable de la qualité du SAE. Détail : `RESULTS_TESTS.md` §21.

## 3.13. Biais multilingue du juge (français vs anglais traduit)

Question posée par la littérature sur l'interprétabilité multilingue (Resck et al.
2025 ; *Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse
Languages*, arXiv:2507.11230) : le juge interprète-t-il mieux les mêmes features
quand les exemples sont présentés en anglais plutôt qu'en français ? Les 150
features déjà jugées sont retraduites (un appel Gemma-3 par feature) et rejugées
intégralement en anglais (`scripts/multilingual_judge_bias_test.py`), sans
réextraction ni réentraînement.

| Métrique | Français original | Anglais traduit | Écart |
|---|---|---|---|
| Interprétabilité odd-one-out (n=145) | 46,9% | 45,5% | −1,4 point (non significatif) |
| Features changeant de statut (FR↔EN) | — | 56/145 = 38,6% | |

**Résultat nul sur l'hypothèse testée** : pas de différence significative entre
français et anglais traduit — aucun biais systématique détecté envers l'anglais sur
ce protocole et ce corpus. Le taux de retournement individuel (38,6%), supérieur à
celui déjà mesuré pour le réordonnancement des exemples (§7.1, 31,3%), est
symétrique (27 flips dans un sens, 29 dans l'autre) plutôt qu'orienté vers
l'anglais — cohérent avec un bruit de traduction générique plutôt qu'un déficit
structurel de l'auto-interprétation en français. Renforce le constat du §7.1 :
le protocole odd-one-out à décision unique reste sensible à toute perturbation de
surface (ordre ou langue), justifiant le vote majoritaire comme protocole par
défaut. Détail et limites assumées (traduction par le même modèle juge, pas de
réentraînement sur corpus anglais natif) : `RESULTS_TESTS.md` §22.


\newpage

---

# Chapitre 4 — Inspection des erreurs et corrections

Ce chapitre consolide, dans l'ordre chronologique des quatre phases du stage
(cf. introduction), l'ensemble des dysfonctionnements identifiés et corrigés. Il
regroupe des informations autrement dispersées entre `Context.md` (journal d'audit) et
`RESULTS_TESTS.md` (journal d'expériences), pour répondre spécifiquement à l'exigence
d'un rapport de stage de documenter non seulement ce qui a fonctionné, mais aussi les
erreurs rencontrées et la façon dont elles ont été diagnostiquées.

## Phase 1 — Audit et fiabilisation initiale

Le pipeline hérité présentait plusieurs bugs bloquants ou silencieux, découverts par
exécution réelle (aucun n'était détecté par une simple lecture de code) :

| Bug | Symptôme | Cause | Correction |
|---|---|---|---|
| Overflow fp16 sur activations massives | `Loss=nan` dès la 1ère époque d'entraînement de `ExtendedSAE` | Gemma-3 présente des activations de norme très élevée (~1e5) dans son residual stream, au-delà du maximum représentable en fp16 (~65 504) | Précision par défaut passée à **bf16** partout, y compris en local (même plage d'exposant que fp32) |
| Cast dtype global cassant le backward | `RuntimeError: Found dtype BFloat16 but expected Float` après le fix bf16 | `FrozenCoreResidualSAE` castée en bloc via `.to(TORCH_DTYPE)`, alors que sa branche "extra" est conçue pour rester fp32 | Cast restreint à `core_sae` uniquement ; `residual` explicitement recasté en fp32 dans `forward()` |
| `hook_layer` figé en dur | Mauvaise couche extraite sur les modèles autres que 12b/layer-24 | Valeur `24` codée en dur, biais historique | Dérivée dynamiquement du `hook_name` résolu |
| `d_in` incorrect pour 270m | Dimensions incohérentes au chargement du SAE | Valeur supposée (1024) tirée de spécifications génériques publiques | Corrigée à **640**, confirmée empiriquement sur les poids réellement téléchargés |
| Route REST Neuronpedia cassée | Récupération de labels systématiquement vide | `/api/explanation/export` non fiable | Téléchargement direct des lots `.jsonl.gz` du bucket S3 public `neuronpedia-datasets` |
| Repos HuggingFace incorrects | Téléchargements en échec (404) | `gemma-scope-2-4b-it-res` figé en dur ; org F2LLM `Alibaba-NLP` inexistante | `download_sae.py` réécrit pour lire `src/config.py` ; org corrigée en `codefuse-ai` |
| Crashs sur petits corpus | `ValueError`/`TypeError` (UMAP/HDBSCAN, `corpus_diff_stats`) | Aucune garde pour 0 feature active ou `N_DOCS` trop petit pour l'initialisation spectrale | Dégradation propre ajoutée (résultat vide bien formé plutôt qu'exception) |
| Imports cassés | `ModuleNotFoundError` / échec de collecte pytest | Chemin `sys.path` erroné, import relatif incorrect, symbole inexistant importé | Chemins et imports corrigés ; test réécrit sur la fonction réellement maintenue |

Validation de cette phase : suite de tests (8/8 passants) et run complet de bout en
bout sur **Gemma-3-270M-it** (profil réduit, 6 Go VRAM), qui a permis d'itérer
rapidement avant tout passage à l'échelle coûteux sur 12B. Cette validation locale a
aussi révélé une **limite** qui deviendra le sujet central de la phase 2 : les
features de l'extension `ExtendedSAE` restaient `dead_feature` (jamais activées) avec
un budget d'entraînement modeste.

## Phase 2 — Diagnostic du taux d'interprétabilité (cœur du stage)

Sur le premier run à l'échelle complète disponible (Gemma-3-12B-it, `results_v9_full`),
la limite ci-dessus se manifestait différemment : les features n'étaient plus mortes
(0/10), mais seules 2/10 (20%) passaient le test d'auto-interprétation odd-one-out.
La démarche de diagnostic complète (élimination de l'hypothèse "features mortes",
inspection qualitative des exemples présentés au juge, lecture du code d'assemblage
du corpus) est détaillée au chapitre 3. Elle a révélé une **erreur de conception**,
plus qu'un bug au sens strict : le corpus utilisé pour entraîner le SAE d'extension
était construit **exclusivement** à partir de textes génériques (FineWeb-2/Wikipedia
filtrés par mots-clés), les emails originaux n'étant chargés que pour une visualisation
post-hoc, jamais vus pendant l'entraînement. Corrigée par
`build_email_train_test_corpus()` (corpus principal = emails + augmentés, split
group-aware par mail d'origine), cette correction a fait passer le taux
d'interprétabilité de 20% à ~41-45%, l'effet dominant identifié dans tout le stage.

Un bug secondaire a été découvert en instrumentant la validation de cette correction :
la sonde de classification multi-classe sur les 14 axes de perturbation échouait
silencieusement (exception attrapée, métrique `NaN`) car `LogisticRegression
(solver="liblinear")` ne supporte, dans les versions récentes de scikit-learn, que la
classification binaire. Corrigé par sélection dynamique du solveur (`lbfgs` au-delà de
deux classes), sans régresser le probe binaire préexistant.

## Phase 3 — Relecture critique face à la littérature de référence

Une relecture ligne à ligne du code du projet face au papier de référence
(*Interpretable Embeddings with Sparse Autoencoders*, Jiang, Sun et al. 2025,
Appendices C/E/F) a mis au jour quatre écarts, tous corrigés ou explicitement
documentés comme piste non intégrée :

- **Retrieval et clustering ciblé par sous-chaîne littérale** (`word in label`) plutôt
  que par similarité d'embedding sémantique (méthode de la référence) — ratait des
  labels sémantiquement liés mais formulés différemment, et retournait des faux
  positifs sur un mot partagé sans rapport de sens. Corrigé par une nouvelle fonction
  `select_latents_by_similarity` (embeddings **bge-m3**, choisi après comparaison
  empirique face à F2LLM : ce dernier donnait de bons résultats sur une requête mais
  des résultats sans rapport sur une autre, pooling dernier-token mal adapté aux
  labels courts en contexte multilingue).
- **Détection de corrélations "intéressantes" jamais câblée** : la fonction NPMI +
  communautés Louvain existait mais n'était appelée nulle part dans le pipeline
  principal, aucun filtre "NPMI élevé + similarité sémantique faible" (méthode de la
  référence) n'était implémenté. Nouvelle fonction `find_interesting_pairs` ajoutée et
  intégrée au pipeline principal.
- **Marqueurs erronés sur les exemples négatifs** dans le test de labellisation
  contrastive directe (protocole alternatif de la référence, Appendix C) : bug trouvé
  en écrivant le test de comparaison lui-même.
- **Prompt contenant un exemple de valeur JSON littéral** que le modèle recopiait
  verbatim pour ~59% des features testées — un "succès" en apparence (champ `confident`
  à `true`) qui masquait un défaut de prompt. Corrigé en remplaçant l'exemple par une
  notation `<placeholder>` explicite et une instruction de ne pas la recopier ; la
  proportion de copies a diminué mais le champ `confident` auto-rapporté reste peu
  fiable (`true` pour 150/150 features dans les deux runs, avant et après le fix), un
  résultat en soi (l'auto-évaluation de confiance d'un LLM ne peut pas se substituer à
  une mesure indépendante — cf. chapitre 5).
- **Biais résiduel "Objet:"/"Subject:"** dans le corpus augmenté : 20,6% des mails
  générés conservaient une ligne d'objet que les mails originaux n'ont pas, un artefact
  de formatage risquant d'être appris comme signal par le SAE plutôt que le contenu
  réel. Corrigé au chargement (`load_augmented`, 0,0% après fix) ; effet mesuré sur le
  diffing complet (réduction de 65% et 49% du nombre de features "significatives" sur
  les deux axes orthographiques les plus confondables avec l'artefact) — plus modeste
  qu'attendu à l'échelle du corpus complet, mais réel et désormais éliminé.

## Phase 4 — Mise à l'échelle et consolidation

- **Bug de configuration réseau récurrent (3 occurrences séparées)** : les nœuds de
  calcul du cluster SLURM sont isolés du réseau (`HF_HUB_OFFLINE=1`) ; le `MODEL_ID`
  par défaut (`src/config.py`) résout vers un identifiant de dépôt HuggingFace distant
  plutôt qu'un chemin disque local. Trois jobs différents (test de plausibilité
  d'explication, rerun Pipeline 2 avec F2LLM-330M) ont échoué pour cette raison avant
  d'être identifiés et corrigés par surcharge explicite de la variable d'environnement
  — un rappel que ce risque doit être vérifié systématiquement à chaque nouveau script
  SLURM plutôt que découvert à l'exécution.
- **Mélange de types dans l'affichage du dashboard** : une colonne de métriques
  mêlant valeurs numériques et texte libre provoquait un avertissement de conversion
  silencieuse côté pyarrow — corrigé en séparant l'affichage du texte libre.
- **Choix de largeur du SAE core reposant sur une donnée non vérifiée** : la
  documentation initiale du projet ne comparait la couverture Neuronpedia qu'entre 16k
  et 262k pour le modèle 12B (16k retenu, 262k écarté). Une largeur intermédiaire,
  65k, n'avait jamais été vérifiée spécifiquement pour ce modèle (seule une couverture
  ~98% pour 65k était documentée, mais mesurée sur un modèle différent,
  gemma-3-270m-it). Une vérification empirique systématique des quatre largeurs
  disponibles (16k/65k/262k/1m) a montré que 65k est en réalité la meilleure
  couverture pour 12B (87,8%, 57 551 features labellisées), meilleure que 16k (82,6%,
  13 535) — largeur adoptée pour le run de mise à l'échelle final (chapitre 3).
- **Suppression accidentelle d'un lien symbolique confondu avec un doublon** : lors
  d'un nettoyage disque, un dossier racine de 30 Go (`saes/`) a été identifié comme un
  doublon legacy de `local_data/saes/` (ancienne convention de nommage) et supprimé
  après confirmation. En réalité, `local_data/saes/gemma-scope-2-12b-it` était un
  **lien symbolique** vers ce même dossier, pas une copie indépendante — sa
  suppression a donc effacé la seule copie physique des poids SAE, laissant un lien
  cassé, détecté seulement à l'échec du job suivant (`ValueError` au chargement du
  SAE). Impact nul sur les résultats déjà produits (artefacts déjà écrits sur disque
  indépendamment des poids sources) ; poids retéléchargés et lien symbolique remplacé
  par un dossier réel pour éliminer la source de confusion. Leçon retenue : vérifier
  `ls -la`/`readlink` avant de supprimer un chemin présenté comme "doublon", pas
  seulement sa taille ou son nom.
- **Erreur de terminologie sur la nature du corpus "original"** : tout le rapport,
  jusqu'à cette phase, employait "mails réels"/"emails réels"/"corpus réel EDF"
  pour désigner `Mails.tsv`. Cette formulation est incorrecte : `Mails.tsv` est
  lui-même un jeu de données synthétique, produit par un travail antérieur du
  laboratoire EDF R&D (indépendant de ce stage), pas de la correspondance client
  authentique. Ni le corpus "original" (`Mails.tsv`) ni les variantes augmentées
  générées pendant ce stage ne sont donc des données réelles au sens strict.
  Corrigé par une relecture terminologique complète du rapport et des documents
  techniques (`docs/`, `RESULTS_TESTS.md`, `README.md`) : le terme "réel(s)" est
  remplacé par "original(aux)" partout où il désignait ce corpus, avec une
  clarification explicite ajoutée au chapitre 2. Erreur sans impact sur la
  validité des résultats eux-mêmes (aucune métrique ne dépend de l'authenticité du
  corpus), mais une imprécision qu'un rapport de stage se doit de corriger.

## Constat transversal

Sur l'ensemble du stage, la quasi-totalité des bugs significatifs ont été détectés par
**exécution réelle et inspection directe des résultats intermédiaires** (valeurs de
perte, exemples présentés au juge, couverture mesurée empiriquement), jamais par
relecture de code seule. Ceci a orienté une pratique systématique : ne jamais publier
un résultat sans avoir inspecté au moins un échantillon qualitatif des données qui
l'ont produit — pratique qui a directement permis de découvrir l'erreur de conception
du corpus (phase 2, le résultat le plus important du stage) et le biais "Objet:"
(phase 3), tous deux invisibles à la seule lecture des métriques agrégées.


\newpage

---

# Chapitre 5 — Limites et perspectives

## Limites actuelles

### Taux d'interprétabilité résiduel (~55-59% de features non interprétées)

Établi comme n'étant pas dû au volume de tokens (cf. `03_experiences_et_resultats.md`).

**Mise à jour (testé)** : la piste "robustesse du protocole de jugement" a été
vérifiée (`scripts/judge_robustness_check.py`, `RESULTS_TESTS.md` §13.1). En
répétant la question odd-one-out 5 fois par feature avec un ordre de mélange
différent à chaque fois : seulement 30,7% des features obtiennent une décision
unanime sur les 5 répétitions ; le taux agrégé d'interprétabilité bouge peu (45,3%→
48,7%) mais 31,3% des features changent individuellement de statut selon l'ordre de
présentation. **Confirmé : une partie substantielle du résidu non-interprété est due
au bruit du protocole de jugement (décision greedy unique, sensible à l'ordre), pas
nécessairement à un défaut réel des features.** Un vote majoritaire sur plusieurs
répétitions devrait être adopté comme protocole par défaut plutôt qu'une seule
décision greedy.

Pistes encore non testées par manque de temps, par ordre de coût croissant :

1. ~~**Robustesse du protocole de jugement**~~ **FAIT**, cf. ci-dessus.
2. **Qualité du contrôle négatif** : le contrôle négatif (`build_feature_examples_with_control`)
   est actuellement un document sous un quantile bas d'activation pour la feature
   testée, pas nécessairement un contre-exemple "propre" conceptuellement. Une
   feature réellement monosémantique pourrait échouer au test si le contrôle négatif
   choisi partage accidentellement une propriété de surface avec les exemples positifs.
3. **Capacité architecturale de l'extension** (`D_EXTRA=1024`, `K_EXTRA=32`) : non
   testée dans cette investigation (fixée dans les trois runs de validation). Une
   extension plus large ou plus/moins parcimonieuse pourrait changer le taux de
   features réellement monosémantiques indépendamment du corpus ou du volume.
4. **Fiabilité du juge selon la taille du modèle** : observée comme dégradée sur
   `gemma-3-270m-it` par rapport à un modèle plus grand lors de la validation locale
   initiale (`Context.md`). Le juge utilisé pour la validation à l'échelle (12B) est
   déjà le plus grand modèle disponible dans ce projet ; tester un modèle encore plus
   grand comme juge (sans nécessairement l'utiliser pour l'extraction d'activations)
   est une piste possible mais coûteuse.

**Mise à jour (testé, session interp_embed)** : une piste supplémentaire, plus
fondamentale, a été testée (`scripts/contrastive_labeling_test.py`,
`RESULTS_TESTS.md` §15.4) — le protocole de la référence (interp_embed, Appendix C)
ne gate JAMAIS la labellisation derrière un test odd-one-out : il génère toujours un
label par contraste direct (10 positifs + 10 négatifs). Sur les 82 features
originellement rejetées par notre gate, la génération contrastive directe produit un
label spécifique et qualitativement plausible pour la totalité d'entre elles après
correction de deux bugs trouvés en écrivant le test (marqueurs `<<>>` erronés sur les
négatifs ; un exemple de valeur JSON dans le prompt que le modèle recopiait
littéralement pour ~59% des features au premier essai). Exemples de labels récupérés :
`Mise en service énergie`, `Numéro de contrat`, `Demande de résiliation`,
`Informations bancaires`, `Sentiment d'urgence`. **Limite** : le champ `confident`
auto-rapporté par le LLM reste à `true` pour 150/150 features dans les deux runs —
pas un signal de qualité fiable en l'état, il faudrait le remplacer par une
validation croisée indépendante (ρ_interp déjà implémenté, ou vote majoritaire
odd-one-out en aval plutôt qu'en amont de la labellisation). **Non intégré au
pipeline de production dans cette session** — changerait le chiffre central du
rapport (45,3%), nécessite de refaire tourner une validation à l'échelle comparable
avant adoption.

### Comparaisons avec l'état de l'art

`Context.md` (règle n°2) demande une comparaison documentée et systématique avec
SAELens. **Fait** (`scripts/saelens_numeric_comparison.py`, `docs/references.md`) :
comparaison chiffrée, sur le même SAE natif sae-lens et les mêmes activations réelles,
entre notre formule de variance expliquée et les deux formules maintenues par
`sae_lens.evals` elle-même. Résultat notable : désaccord numérique important entre
les trois (0,41 / 0,83 / 1,00) causé par les activations massives de Gemma-3 — la
formule qui somme sur les dimensions avant de normaliser est mécaniquement dominée
par une seule dimension outlier et rapporte une variance quasi-totalement expliquée,
sans rapport avec la qualité de reconstruction réelle. Recommandation retenue : ne
jamais publier un score de variance expliquée sans préciser la formule exacte utilisée
sur ce modèle. La comparaison avec `interp_embed` reste partielle (test optionnel
dépendant d'une installation non faite par défaut).

**Mise à jour (identifié, session pdf/)** : "SAE Boost" (Koriagin et al., COLM 2025)
n'était pas "non fait" mais déjà implémenté sans le savoir --
`FrozenCoreResidualSAE`/`ExtendedSAE` EST une implémentation de SAE Boost (même
architecture : SAE résiduel sur l'erreur de reconstruction d'un core gelé, sommé à
l'inférence). Deux écarts identifiés par la relecture du papier restent à tester :
(1) leur étude de sensibilité montre qu'un `K_EXTRA` plus faible (k=5 optimal chez
eux, contre 32 dans ce projet) améliore l'interprétabilité au prix d'un peu d'EV
domaine -- non testé ; (2) leur étude montre qu'un budget de 100-200M tokens est
nécessaire pour que le SAE résiduel converge sans dégrader la performance générale
(jusqu'à -31% d'EV en dessous de 100M) -- notre ablation volume (100k-2M tokens)
reste 50-100x en dessous de ce seuil, donc **notre conclusion "le volume ne change
rien" n'est établie que dans un régime que leur étude qualifie d'insuffisant** ;
elle ne peut pas être extrapolée sans un run à cette échelle, non lancé dans ce
stage (coût GPU substantiel). Aucune comparaison chiffrée avec leurs baselines
alternatives (Extended SAE random/most-active init, SAE Stitching, full
fine-tuning) n'a été menée sur ce projet. Détail complet : `RESULTS_TESTS.md` §18.

**Mise à jour (nouveau, testé, session pdf/)** : une question plus fondamentale a été
posée par *Sanity Checks for Sparse Autoencoders* (Korznikov et al., 2026) --
un SAE dont le décodeur est figé à une initialisation aléatoire (jamais entraîné)
égale, dans leur étude, un SAE réellement entraîné sur interprétabilité automatique,
sparse probing et édition causale. Reproduit sur ce projet
(`FrozenDecoderExtendedSAE`, `SANITY_CHECK_FROZEN_DECODER=1`) : cf. `RESULTS_TESTS.md`
§19 pour le protocole et le résultat (en cours au moment de la rédaction de cette
version du rapport).

### Biais de génération résiduel dans le corpus augmenté

**Corrigé et mesuré** (`RESULTS_TESTS.md` §14.1) : 20,6% des mails augmentés
contenaient encore une ligne "Objet :"/"Subject :" que les mails originaux n'ont pas.
Fix appliqué au chargement (`load_augmented`, pas de régénération nécessaire — 0,0%
après fix) et effet mesuré sur le diffing complet : réduction de 65% et 49% du nombre
de features "significatives" sur les deux axes orthographiques (les plus confondables
avec l'artefact), effet modéré sur l'urgence (−7,5%/−3,1%), négligeable ailleurs.
Contrairement à l'hypothèse initiale, l'artefact ne dominait déjà pas la majorité des
features significatives à l'échelle du corpus complet (0,21% des features
significatives portaient un label "Subject:"/"Objet:", avant comme après) — son effet
mesuré est réel mais plus circonscrit que ce que suggérait l'observation initiale sur
l'échantillon test à 60 mails (§6, où l'artefact dominait 8/13 classements).

### Retrieval par propriétés et clustering ciblé (bug corrigé)

**Corrigé** (`RESULTS_TESTS.md` §15.1-15.2) : `property_based_retrieval` et
`targeted_clustering_by_axis` sélectionnaient les latents pertinents pour une
requête par matching de sous-chaîne littérale (`word in label`) plutôt que par
similarité d'embedding (méthode de la référence, interp_embed §4.4/Appendix F.1) —
vérifié empiriquement que ça ratait des labels sémantiquement liés mais formulés
différemment, et retournait des faux positifs (mot partagé sans rapport de sens).
Bug additionnel dans `property_based_retrieval` : la pondération "température"
utilisait l'ordre d'itération du dict de labels comme proxy de pertinence, pas un
rang réel. Fix : nouvelle fonction `select_latents_by_similarity`
(`src/sae/saev5.py`), embeddings **bge-m3** (pas F2LLM, testé et rejeté : bons
résultats sur une requête, résultats sans rapport sur une autre — pooling
dernier-token mal adapté à des labels courts en contexte cross-lingue). Validé
bout-en-bout sur les activations déjà en cache, non revalidé par un run complet
(ne change pas la reconstruction des activations elles-mêmes, seulement la
sélection de latents en aval).

### Corrélations "intéressantes" (gap comblé, résultat peu concluant)

**Corrigé** (`RESULTS_TESTS.md` §15.3) : `cooccurrence_graph` (NPMI + communautés
Louvain) n'était jamais appelée dans le pipeline principal — seule la matrice NPMI
brute était calculée et cachée, sans analyse en sortie. Nouvelle fonction
`find_interesting_pairs` (`src/analysis/cooccurrence.py`), filtre NPMI élevé +
similarité sémantique des labels faible (méthode interp_embed §4.2/Appendix E.1).
**Calculée rétroactivement** (`scripts/compute_interesting_correlations_retro.py`,
`RESULTS_TESTS.md` §16.3) sur `results_v10_emails_main` sans réextraction Gemma-3 :
seulement 3 paires retenues sur 26 579 arêtes du graphe (3 395 nœuds), et 2 des 3
impliquent une feature non labellisée — résultat honnête mais peu exploitable en
l'état (impossible de juger la pertinence d'une corrélation quand un des deux côtés
n'a pas de label). Piste retenue : élargir la plage de fréquence ou prioriser les
paires où les deux features sont labellisées.

### Qualité de l'explication document-level (nouveau, testé)

Question distincte de tout ce qui précède (qui évalue une feature isolée ou une
capacité globale), directement issue d'une demande utilisateur : pour UN document
donné, l'explication produite (features actives + labels) est-elle bonne ?
- **Fidélité** (`scripts/explanation_fidelity_test.py`, ablation) : chute de 58 à 100
  points de probabilité en ablatant les 10 features "explicatives", chute quasi nulle
  (<0,4 point) en ablatant des features aléatoires ou peu contributives (ratios de
  250× à 576 000× selon l'intention). Résultat sans ambiguïté : l'explication porte
  réellement la décision.
- **Plausibilité** (`scripts/explanation_plausibility_test.py`, choix forcé, juge
  Gemma-3-12B-it) : 71,7% (43/60) de choix corrects contre 50% au hasard (p < 0,001) —
  significativement au-dessus du hasard, mais loin d'être parfait (cohérent avec le
  taux d'interprétabilité résiduel ~45-55%).

Détail complet : `RESULTS_TESTS.md` §16.1-16.2, `report/03_experiences_et_resultats.md`
§8, dashboard (onglet "Explication (fidélité/plausibilité)").

### Comparaison du backbone d'embedding Pipeline 2 : F2LLM-80M vs -330M (nouveau, testé)

Résultat **mixte** (`RESULTS_TESTS.md` §16.5) : -330M reconstruit légèrement mieux
(NMSE −7,5%) et sépare un peu mieux le corpus de diffing générique (+2 points), mais
sépare légèrement MOINS bien les axes email (−2,2 points, la métrique la plus
proche des objectifs métier). Aucun écart n'est de l'ordre d'un problème majeur ; pas
de justification claire pour préférer l'un à l'autre sur ce projet à ce stade.

### Facteurs non contrôlés dans le corpus augmenté

Les variantes augmentées sont générées par le même modèle (Gemma-3-12B-it) qui sert
aussi de juge d'interprétation et d'extracteur d'activations. Un style de génération
propre au modèle (tournures récurrentes, longueur, structure) pourrait constituer un
facteur de confusion partagé entre "ce qui rend une variante reconnaissable comme
augmentée" et "ce que le SAE apprend à détecter" — non quantifié dans cette
investigation.

### Pistes issues d'une relecture élargie de la littérature (nouveau)

Une relecture de l'ensemble des PDF de référence disponibles (`pdf/`, au-delà des
seuls SAE Boost et sanity checks déjà traités ci-dessus) fait ressortir trois pistes
non testées, chacune directement actionnable mais représentant un effort
d'implémentation plus substantiel que les corrections déjà apportées :

- **Feature splitting/absorption comme cause possible du résidu non-interprété**
  (*Matryoshka SAEs*, Bussmann et al. 2025) : leur travail montre qu'agrandir
  simplement un dictionnaire SAE (notre ablation capacité, `D_EXTRA` 1024→2048,
  chapitre 3 §10.4) peut dégrader la qualité des features de haut niveau par
  fragmentation/absorption plutôt que de mieux couvrir le domaine — cohérent avec le
  fait que notre ablation capacité n'a montré aucun gain d'interprétabilité. Leur
  solution (dictionnaires SAE emboîtés, entraînés simultanément à plusieurs tailles)
  n'a pas été implémentée : changement de la boucle d'entraînement plus substantiel
  que le sanity check Frozen Decoder déjà réalisé. **Ne pas confondre** avec
  `MATRYOSHKA_DIM` (`src/config.py`), qui ne concerne que la troncature des
  embeddings F2LLM, un mécanisme complètement différent (cf. `docs/references.md`).
- **Entraînement supervisé conjoint SAE+classifieur pour la classification**
  (*ClassifSAE*, Le Bail et al. 2025) : ce projet extrait des concepts de façon
  totalement non supervisée puis les relie à la classification par une sonde
  post-hoc (`downstream_classification`). ClassifSAE propose d'entraîner le SAE
  conjointement avec un classifieur (avec une pénalité de parcimonie sur le taux
  d'activation), spécifiquement pour concentrer les concepts pertinents à la tâche —
  directement aligné avec les objectifs "détection d'urgence"/"détection d'intention"
  du cadrage initial. Non implémenté : nécessiterait une nouvelle boucle
  d'entraînement (SAE + tête de classification jointe), distincte de l'architecture
  actuelle des deux pipelines.
- **Steering comme méthode d'explication "output-based" jamais évaluée** (taxonomie
  de *A Survey on Sparse Autoencoders*, Shu et al. 2025) : `steer_activations`/
  `steer_and_decode` (`src/sae/sae_shared.py`) et `p1_steering_demo.json` existent
  déjà dans le dépôt, mais n'ont jamais été évalués comme méthode d'explication à
  part entière (contrairement aux protocoles "input-based" — odd-one-out,
  labellisation contrastive — qui sont, eux, au cœur du chapitre 3). Piste peu
  coûteuse : mesurer si l'amplification d'une feature jugée interprétable produit un
  changement de génération cohérent avec son label, sur un échantillon de documents.
- **Biais multilingue** (*survey* sur l'explicabilité des LLM multilingues, Resck et
  al. 2025) : **mesuré** (`RESULTS_TESTS.md` §22) — pas de différence significative
  d'interprétabilité entre français et anglais traduit (46,9% vs 45,5%), mais 38,6%
  des features changent individuellement de statut selon la langue, un taux de
  bruit supérieur à celui déjà mesuré pour le réordonnancement des exemples (§13.1,
  31,3%). Pas de biais systématique détecté envers l'anglais sur ce test précis.
- **Variance de seed d'entraînement du SAE** (*Unstable Features, Reproducible
  Subspaces*, arXiv:2606.12138) : **mesurée** (`RESULTS_TESTS.md` §21) — taux
  d'interprétabilité agrégé stable entre deux seeds (45,3% vs 47,3%), mais
  seulement 28,2% de recouvrement exact des labels individuels obtenus. Les
  features prises individuellement ne sont pas reproductibles à l'identique d'un
  seed à l'autre, contrairement au taux agrégé.

## Perspectives pour la suite du stage

1. ~~Tester en priorité la robustesse du protocole de jugement (vote majoritaire)~~
   **FAIT** (cf. section "Limites actuelles" ci-dessus) — passer ce vote majoritaire
   en protocole par défaut de `odd_one_out_judge` (actuellement une fonction séparée,
   `scripts/judge_robustness_check.py`, à fusionner dans `src/sae/judge.py` si adopté).
2. ~~Formaliser la comparaison **chiffrée** avec SAELens~~ **FAIT** (cf. section
   "Comparaisons avec l'état de l'art" ci-dessus) — a révélé un problème plus large
   (désaccord entre formules de variance expliquée sur activations à magnitude
   hétérogène) qu'une simple validation d'implémentation. Reste à faire : implémenter
   la métrique robuste aux outliers proposée (médiane des ratios par token).
3. Poursuivre la factorisation de `src/sae/saev5.py` vers l'architecture cible décrite
   dans `Context.md` (`src/models/`, séparation training/extraction) — dette technique
   qui n'affecte pas la validité des résultats mais complique la maintenance.
4. ~~Dashboard interactif (Streamlit)~~ **FAIT** : `src/visualization/dashboard.py`
   (`RESULTS_TESTS.md` §14.2) — vue d'ensemble, UMAP interactif, features (avec
   exemples positifs/négatifs), diffing, recherche par mot-clé, urgence/robustesse.
   Limite : recherche par mot-clé sur les labels déjà attribués, pas une ré-inférence
   BM25 live sur le vocabulaire latent complet (cf. `scripts/retrieval_demo.py` pour
   cette dernière) ; pas de déploiement serveur persistant, lancement manuel.
5. ~~Exploiter le résultat de séparabilité linéaire des axes de perturbation... pour
   un cas d'usage concret de détection d'urgence/d'intention sur mails originaux~~
   **FAIT** : `scripts/intent_urgency_probe.py`, `RESULTS_TESTS.md` §13.2 — sonde sur
   les labels faibles réels (regex, indépendants du corpus augmenté) : +27,0 points
   sur l'urgence, +42,6 points sur la réclamation par rapport à la baseline classe
   majoritaire. Reste à faire : évaluer sur un jeu de labels d'urgence/intention
   annotés manuellement plutôt que des labels faibles par regex (limite ci-dessous),
   et sur le Pipeline 2 (F2LLM) en plus du Pipeline 1 déjà testé.
6. ~~Corriger le retrieval/clustering ciblé (matching par sous-chaîne) et brancher
   les corrélations "intéressantes"~~ **FAIT** (cf. sections ci-dessus,
   `RESULTS_TESTS.md` §15.1-15.3) — validés sur les activations déjà en cache, pas
   encore par un run complet à l'échelle (aucun changement des activations
   elles-mêmes, seulement de la sélection de latents en aval).
7. Adopter le protocole de labellisation contrastive directe (§15.4) comme
   alternative/complément au gate odd-one-out — well-evidenced (labels qualitativement
   plausibles récupérés sur 100% d'un échantillon de features rejetées) mais nécessite
   (a) une validation croisée de la qualité des labels (le champ `confident`
   auto-rapporté n'est pas fiable), (b) un run de validation à l'échelle comparable
   aux 3 runs de `RESULTS_TESTS.md` §12 avant de remplacer le chiffre 45,3% publié.
8. ~~Calculer `find_interesting_pairs` (corrélations)~~ **FAIT** (rétroactivement,
   sans réextraction, `RESULTS_TESTS.md` §16.3) — résultat peu concluant (3 paires
   seulement, 2/3 avec une feature non labellisée). Reste à faire : comparer à des
   biais/artefacts réels connus du corpus (ex. le biais "Objet :" avant correction,
   §14.1) pour valider empiriquement la méthode sur ce projet, à la manière de la
   validation par injection synthétique du papier de référence (§4.2, Appendix E.2) ;
   élargir la plage de fréquence de `cooccurrence_graph` pour augmenter le rappel.
9. ~~Mettre en place un test de qualité de l'explication document-level (fidélité +
   plausibilité)~~ **FAIT** (`RESULTS_TESTS.md` §16.1-16.2) — résultats très positifs
   sur la fidélité, positifs mais imparfaits sur la plausibilité. Reste à faire :
   étendre le test de plausibilité au Pipeline 2, et à un échantillon plus large que
   60 documents pour resserrer l'intervalle de confiance.
10. ~~Comparer le backbone d'embedding Pipeline 2 (F2LLM-80M vs -330M)~~ **FAIT**
    (`RESULTS_TESTS.md` §16.5) — résultat mixte, pas de gain net. Reste à faire :
    tester bge-m3 comme backbone Pipeline 2 (actuellement utilisé seulement pour la
    similarité de labels), qui a montré une meilleure fiabilité sur des textes courts
    dans un contexte différent (§15.2) mais n'a jamais été testé comme backbone
    d'entraînement du `PhraseLevelSAE` lui-même.
11. ~~Concevoir un protocole de test complet du dépôt sous conditions fixées~~ **FAIT** :
    `docs/evaluation_protocol.md` + `scripts/consolidate_evaluation_report.py` +
    onglet dashboard "Rapport consolidé". Aucun problème majeur rencontré sur cette
    passe (cf. les critères de décision du protocole) — la comparaison multi-modèles/
    conditions envisagée par l'utilisateur peut être considérée en suite de stage.
12. ~~Identifier et documenter la correspondance avec SAE Boost~~ **FAIT**
    (`RESULTS_TESTS.md` §18). Reste à faire : tester un `K_EXTRA` plus faible (proche
    de leur k=5 optimal) et, si le budget GPU le permet, un run à 100-200M tokens
    d'entraînement de l'extension pour vérifier si leur seuil de convergence
    s'applique à ce projet (notre ablation actuelle reste 50-100x en dessous) ;
    comparer chiffré à leurs baselines alternatives (Extended SAE, SAE Stitching,
    full fine-tuning) sur le corpus emails.
13. ~~Reproduire le sanity check "Frozen Decoder" (Korznikov et al. 2026)~~ **FAIT**
    (`RESULTS_TESTS.md` §19, `FrozenDecoderExtendedSAE`) — résultat **nuancé** :
    l'interprétabilité odd-one-out résiste bien (45,3% entraîné vs 29,3% figé
    aléatoire, écart significatif) mais la classification en aval y résiste beaucoup
    moins (93,5% vs 91,2% — un décodeur aléatoire capture déjà la quasi-totalité du
    signal), répliquant partiellement le constat du papier sur le sparse probing.
    Reste à faire : étendre le sanity check au Pipeline 2 (`PhraseLevelSAE`,
    entraîné from-scratch, jamais testé contre un décodeur figé) ; envisager les
    métriques plus exigeantes du papier (AutoInterp par description+détection sur
    échantillon non vu, sparse probing SAEBench) en remplacement de la sonde de
    classification actuelle, dont ce sanity check a montré la faible sensibilité.
14. Tester des dictionnaires SAE emboîtés (*Matryoshka SAEs*, Bussmann et al. 2025)
    pour l'extension P1, comme piste alternative à l'ablation capacité simple
    (`D_EXTRA`/`K_EXTRA`, déjà testée sans effet) pour expliquer/réduire le résidu
    non-interprété — nécessite une nouvelle boucle d'entraînement multi-échelle.
15. Entraîner un SAE supervisé conjointement avec un classifieur (*ClassifSAE*,
    Le Bail et al. 2025) pour la détection d'urgence/intention, en alternative à la
    sonde post-hoc actuelle (`downstream_classification`) — permettrait de comparer
    directement la précision et l'interprétabilité des concepts obtenus.
16. Évaluer le steering (`steer_activations`/`steer_and_decode`, déjà implémenté mais
    jamais utilisé comme méthode d'explication à part entière) comme complément
    "output-based" aux protocoles "input-based" déjà validés (chapitre 3) — piste peu
    coûteuse (pas de nouvel entraînement, juste une nouvelle évaluation).
17. ~~Quantifier le biais multilingue potentiel du juge d'auto-interprétation~~
    **FAIT** (`RESULTS_TESTS.md` §22) — résultat **nul sur l'hypothèse testée** :
    aucune différence significative entre le taux d'interprétabilité en français et
    en anglais traduit (46,9% vs 45,5%, z=0,24). Renforce en revanche le constat du
    §13.1 : 38,6% des features changent de statut selon la langue de présentation
    (contre 31,3% pour un simple réordonnancement), confirmant que le protocole
    odd-one-out à décision unique reste globalement bruyant face à toute
    perturbation de surface, pas spécifiquement biaisé envers l'anglais sur ce
    corpus. Reste à faire : tester l'hypothèse alternative (entraîner le SAE sur un
    corpus anglais natif équivalent, pas seulement traduire la vue du juge).
18. ~~Tester la variance de seed d'entraînement du SAE~~ **FAIT**
    (`RESULTS_TESTS.md` §21, *Unstable Features, Reproducible Subspaces*,
    arXiv:2606.12138) — taux d'interprétabilité agrégé stable entre seeds (45,3% vs
    47,3%, non significatif) mais seulement 28,2% de recouvrement exact des labels
    individuels obtenus. Confirme que les features individuelles ne sont pas
    reproductibles à l'identique (seule la performance agrégée et la thématique
    générale le sont) — nuance importante pour la lecture des exemples de features
    cités dans ce rapport (chapitre 3) : à comprendre comme représentatifs d'une
    catégorie récurrente de concepts, pas comme des atomes stables du dictionnaire.


\newpage

---

# Conclusion générale

## Bilan par rapport aux objectifs initiaux

Le stage visait à rendre fonctionnelle et exploitable une plateforme d'analyse
interprétable de mails clients EDF fondée sur des Sparse Autoencoders. Au terme des
quatre phases décrites dans ce rapport, les deux pipelines (Gemma-3 + GemmaScope
étendu ; F2LLM + SAE dédié) fonctionnent de bout en bout sur le corpus original, avec des
résultats quantifiés et reproductibles sur l'ensemble des capacités visées par
l'énoncé initial :

- **Détection d'urgence et d'intention** : séparabilité linéaire forte sur les axes
  synthétiques (93,5%/79,3% selon le pipeline) et gain net mesuré sur des labels
  faibles indépendants tirés de mails originaux non augmentés (+27,0 points sur l'urgence,
  +42,6 points sur la réclamation par rapport à la baseline naïve).
- **Explication des décisions** : deux tests indépendants (fidélité par ablation,
  plausibilité par choix forcé) confirment que les features désignées comme
  explication d'un document portent réellement la décision, et sont perçues comme
  significativement plus convaincantes qu'un ensemble de concepts tiré au hasard.
- **Recherche par concept, clustering, diffing, corrélations** : implémentés,
  corrigés après relecture face à la référence méthodologique, et exposés dans un
  dashboard interactif unique.
- **Comparaison documentée avec l'état de l'art** (SAELens, interp_embed) :
  effectuée de façon chiffrée, avec des résultats parfois inattendus (désaccord entre
  formules de variance expliquée à magnitude d'activation hétérogène) qui ont une
  valeur méthodologique au-delà du seul projet.

Le résultat le plus significatif du stage reste le diagnostic du chapitre 3 : le taux
d'auto-interprétation des features, initialement très faible (20%), n'était pas
limité par le volume d'entraînement mais par une erreur de conception du corpus
d'entraînement de l'extension — un exemple concret de la valeur d'une démarche
d'ablation contrôlée plutôt que d'une intuition non testée ("il faut probablement plus
de données").

## Compétences mobilisées et acquises

Ce stage a mobilisé des compétences de recherche appliquée en apprentissage profond
(SAE, précision numérique, entraînement à l'échelle sur cluster GPU), de conception
expérimentale (isolation de variables confondues, ablations contrôlées, tests
statistiques sur des proportions observées), de lecture critique de la littérature
(relecture ligne à ligne d'une implémentation de référence pour en extraire des écarts
méthodologiques actionnables), ainsi que de rigueur en ingénierie logicielle
(diagnostic de bugs silencieux par inspection des résultats intermédiaires plutôt que
par la seule lecture de code, documentation continue des décisions et de leurs
justifications).

## Perspectives

Le chapitre 5 détaille les limites et pistes de poursuite technique. Deux directions
plus larges se dégagent pour la suite :

1. **Comparaison inter-modèles et inter-conditions**, envisagée dès le cadrage de
   cette phase de mise à l'échelle mais volontairement différée jusqu'à validation
   d'une condition unique et maîtrisée (ce rapport) — à mener maintenant que le
   protocole d'évaluation complet du dépôt (`docs/evaluation_protocol.md`) est en
   place et a démontré son applicabilité sur une première condition.
2. **Adoption en production** des protocoles alternatifs identifiés comme prometteurs
   mais non intégrés faute de validation croisée suffisante dans le temps du stage
   (labellisation contrastive directe, vote majoritaire du juge odd-one-out comme
   protocole par défaut) — les deux nécessitent une repasse de validation à l'échelle
   avant de remplacer les chiffres actuellement publiés dans ce rapport.


\newpage

---

# Bibliographie

*Note de rédaction* : les références ci-dessous listent les travaux effectivement
mobilisés pendant le stage (méthode, comparaison chiffrée, ou lecture critique). Les
métadonnées bibliographiques complètes (auteurs exacts, venue, année) des articles
disponibles uniquement sous forme de PDF local (`pdf/`) sont à vérifier/compléter
avant intégration dans la version finale déposée, conformément à la remarque déjà
présente dans `report/README.md`.

## Références académiques

- Jiang, N., Sun, R. et al. (2025). *Interpretable Embeddings with Sparse
  Autoencoders: A Data Analysis Toolkit* (`pdf/InterpretableSAE_Embeddings.pdf`,
  [github.com/nickjiang2378/interp_embed](https://github.com/nickjiang2378/interp_embed)).
  Référence méthodologique principale du stage : protocole de labellisation
  contrastive (Appendix C), détection de corrélations "intéressantes" (§4.2,
  Appendix E), retrieval par propriétés et clustering ciblé par similarité
  d'embedding (§4.3/4.4, Appendix F.1). Une relecture ligne à ligne de cette
  référence face au code du projet a permis d'identifier quatre écarts
  méthodologiques (cf. chapitre 4).
- Bills, S. et al. (2023). *Language models can explain neurons in language models*
  (OpenAI). Origine de la mesure ρ_interp (corrélation de Spearman entre intensité
  jugée par un LLM et activation réelle) utilisée dans le protocole
  d'auto-interprétation local (`src/sae/judge.py`).
- Koriagin, N., Aksenov, Y., Laptev, D., Gerasimov, G., Balagansky, N., Gavrilov, D.
  (2025). *Teach Old SAEs New Domain Tricks with Boosting*
  ([arXiv:2507.12990](https://arxiv.org/abs/2507.12990), COLM 2025,
  `pdf/teacholdsaes.pdf`). Introduit "SAE Boost" — identifié a posteriori comme
  l'architecture déjà implémentée par `FrozenCoreResidualSAE`/`ExtendedSAE` de ce
  projet (cf. chapitre 1 "Perspectives critiques", `RESULTS_TESTS.md` §18).
- Korznikov, A., Galichin, A., Dontsov, A., Rogov, O. Y., Oseledets, I., Tutubalina, E.
  (2026). *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?*
  ([arXiv:2602.14111](https://arxiv.org/abs/2602.14111), `pdf/sanitychecks.pdf`).
  Introduit les baselines à composants gelés/aléatoires (Frozen Decoder, Frozen
  Encoder, Soft-Frozen Decoder) comme test de validité des métriques SAE standard.
  Protocole "Frozen Decoder" reproduit sur ce projet
  (`FrozenDecoderExtendedSAE`, `RESULTS_TESTS.md` §19).
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L. (2023). *Sparse
  Autoencoders Find Highly Interpretable Features in Language Models*
  ([arXiv:2309.08600](https://arxiv.org/abs/2309.08600), `pdf/2309.08600v3.pdf`).
  Un des deux papiers fondateurs de l'usage des SAE pour l'interprétabilité des LLM.
- Bussmann, B., Leask, P., Nanda, N. (2024). *BatchTopK Sparse Autoencoders*
  ([arXiv:2412.06410](https://arxiv.org/abs/2412.06410), `pdf/BatchTopK.pdf`).
  Mécanisme de parcimonie de `ExtendedSAE`/`PhraseLevelSAE`
  (`src/sae/batch.py::BatchTopKEncoder`) — implémentation vérifiée fidèle au papier
  (cf. `docs/references.md`).
- Rajamanoharan, S., Lieberum, T., Sonnerat, N. et al. (2024). *Jumping Ahead:
  Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*
  (`pdf/jumpRELU.pdf`). Architecture du SAE core GemmaScope-2 (Pipeline 1).
- Bussmann, B., Nabeshima, N., Karvonen, A., Nanda, N. (2025). *Learning Multi-Level
  Features with Matryoshka Sparse Autoencoders*
  ([arXiv:2503.17547](https://arxiv.org/abs/2503.17547), `pdf/Matryoshka.pdf`).
  Piste non implémentée pour le résidu non-interprété (chapitre 5) — à ne pas
  confondre avec `MATRYOSHKA_DIM` du projet (cf. `docs/references.md`).
- Le Bail, M., Dentan, J., Buscaldi, D., Vanier, S. (2025). *Unveiling
  Decision-Making in LLMs for Text Classification: Extraction of Influential and
  Interpretable Concepts with Sparse Autoencoders*
  ([arXiv:2506.23951](https://arxiv.org/abs/2506.23951),
  `pdf/UnveilingDecision-MakinginLLMsforTextClassification.pdf`). Introduit
  ClassifSAE (SAE supervisé conjoint SAE+classifieur) — piste non implémentée,
  directement pertinente pour les objectifs détection d'urgence/intention
  (chapitre 5).
- Shu, D., Wu, X., Zhao, H. et al. (2025). *A Survey on Sparse Autoencoders:
  Interpreting the Internal Mechanisms of Large Language Models* (EMNLP 2025
  Findings, `pdf/SurveySAE.pdf`). Taxonomie explications input-based/output-based et
  métriques structurelles/fonctionnelles, utilisée pour cadrer le chapitre 1.
- Resck, L., Augenstein, I., Korhonen, A. (2025). *Explainability and
  Interpretability of Multilingual Large Language Models: A Survey* (EMNLP 2025,
  `pdf/2025.emnlp-main.1033.pdf`). Cité pour le biais multilingue potentiel du juge
  d'auto-interprétation (corpus français), non quantifié dans ce projet.
- Beckmann, P., Queloz, M. (2026). *Mechanistic Indicators of Understanding in Large
  Language Models* (`pdf/MechanisticIndicatorsinLLM.pdf`). Cadrage philosophique
  cité en introduction.
- Documents complémentaires consultés sur l'application des SAE aux embeddings
  denses et à la recherche documentaire (retrieval), disponibles sous `pdf/` :
  `DisentanglingDenseEmbeddingswithSAE.pdf`,
  `DecodingDenseEmbSAEforInterpandDiscretizDenseRetrieval.pdf`,
  `InterpretandControlDenseRetrievalwithSparseLatentFeatures.pdf`,
  `SparseAutoencodersforHypothesisGeneration.pdf`, `Naver.pdf`,
  `12_Towards_Interpretable_Scien.pdf`.

## Dépôts et outils logiciels réutilisés

Cf. `docs/references.md` pour le détail complet (rôle exact dans le projet, statut de
comparaison) :

- **SAELens** — [github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) —
  chargement/encodage du SAE GemmaScope-2 préentraîné.
- **GemmaScope** —
  [github.com/google-deepmind/gemma-scope](https://github.com/google-deepmind/gemma-scope) —
  poids des SAE préentraînés sur Gemma-3.
- **Neuronpedia** — [neuronpedia.org](https://www.neuronpedia.org) — labels officiels
  des features GemmaScope "core".
- **F2LLM-v2** (`codefuse-ai/F2LLM-v2-{80M,160M,330M}`) — modèle d'embeddings de
  phrases, backbone du Pipeline 2.
- **bge-m3** — modèle d'embedding multilingue (pooling CLS), utilisé pour la
  similarité de labels (retrieval/clustering) après comparaison empirique avec F2LLM.

## Document de cadrage

- EDF R&D. *Offre de stage SEQUOIA — Explicabilité de documents par Sparse
  Autoencoders* (`pdf/Offre_Stage_EDF_RD_SEQUOIA_E7S_SAE.pdf`). Document interne,
  origine des objectifs listés en introduction.
