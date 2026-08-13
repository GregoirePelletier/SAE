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
spécifiquement sur le domaine — architecture à cœur gelé identifiée en cours de
stage comme structurellement équivalente à SAE Boost (Koriagin et al., COLM 2025)
— et un second pipeline indépendant fondé sur des embeddings de phrase (F2LLM-v2,
bge-m3). Le pipeline initial, fonctionnel de bout en bout, présentait un taux de
succès faible (20%) au test d'auto-interprétation des features propres au domaine.
Une démarche de diagnostic par ablation contrôlée a établi que ce taux n'était pas
limité par le volume d'entraînement, mais par une erreur de conception du corpus
d'entraînement (uniquement générique, sans texte du domaine cible) — corrigée,
elle porte le taux d'interprétabilité à 45,3%.

Une campagne d'ablations exhaustive (plus de 20 configurations : largeur du SAE,
capacité et parcimonie de l'extension, volume de tokens, graine d'entraînement,
dimension d'embedding, backbone de Pipeline 2) montre qu'**aucun hyperparamètre du
SAE ne modifie significativement ce taux** une fois le domaine corrigé — à
l'exception d'un unique levier : **l'échelle du modèle extracteur/juge**, qui
produit un effet dose-réponse net et hautement significatif (12,0% à 1 milliard de
paramètres, 28,0% à 4 milliards, 45,3% à 12 milliards ; test de tendance de
Cochran-Armitage, p≈1,6×10⁻¹⁰) — de loin le résultat le plus marquant du stage. Un
sanity check contre un décodeur figé aléatoire (Korznikov et al., 2026) confirme
que l'entraînement de l'extension apprend une structure réelle (45,3% contre
29,3%, écart significatif) tout en révélant qu'une classification en aval résiste
beaucoup mieux à cette dégradation que l'interprétation qualitative. Des tests
complémentaires (fidélité et plausibilité de l'explication document-level,
robustesse du protocole de jugement, biais multilingue, fidélité du steering,
évaluation quantitative du retrieval) complètent la validation du système, avec un
audit rétroactif de la méthodologie statistique employée.

**Mots-clés** : Sparse Autoencoders, interprétabilité mécaniste, GemmaScope,
grands modèles de langage, explicabilité, traitement automatique des mails clients,
auto-interprétation par juge LLM, effet d'échelle.

---

## Abstract

This internship addresses automatic explainability of customer emails at EDF using
Sparse Autoencoders (SAE), combining a large pretrained SAE (GemmaScope-2, on
Gemma-3-12B-it activations) extended by a second SAE trained specifically for the
target domain — a frozen-core architecture identified during the internship as
structurally equivalent to SAE Boost (Koriagin et al., COLM 2025) — alongside an
independent sentence-embedding-based pipeline (F2LLM-v2, bge-m3). The initial
end-to-end pipeline showed a low success rate (20%) on the domain-specific feature
auto-interpretation test. A controlled-ablation diagnostic established that this
was not a training-volume limitation but a training-corpus design flaw (generic
text only, no domain-specific text) — once fixed, the interpretability rate rose
to 45.3%.

An exhaustive ablation campaign (20+ configurations: SAE width, extension capacity
and sparsity, token volume, training seed, embedding dimension, Pipeline-2
backbone) shows that **no SAE hyperparameter significantly changes this rate**
once the corpus domain is fixed — except for a single lever: **the scale of the
extractor/judge model**, which produces a clean, highly significant dose-response
effect (12.0% at 1B parameters, 28.0% at 4B, 45.3% at 12B; Cochran-Armitage trend
test, p≈1.6×10⁻¹⁰) — by far the most striking finding of the internship. A sanity
check against a randomly frozen decoder (Korznikov et al., 2026) confirms that the
extension's training learns genuine structure (45.3% vs 29.3%, significant gap)
while also revealing that downstream classification survives this degradation far
better than qualitative interpretation does. Complementary tests (document-level
explanation fidelity and plausibility, judge-protocol robustness, multilingual
bias, steering fidelity, quantitative retrieval evaluation) complete the system's
validation, together with a retroactive audit of the statistical methodology used.

**Keywords**: Sparse Autoencoders, mechanistic interpretability, GemmaScope, large
language models, explainability, customer email analysis, LLM auto-interpretation,
scaling effect.

---

## Sommaire

- Introduction générale
- Chapitre 1 — État de l'art
- Chapitre 2 — Architecture et implémentation
- Chapitre 3 — Démarche expérimentale et résultats
- Chapitre 4 — Limites et perspectives
- Conclusion générale
- Bibliographie

---

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

L'énoncé initial du stage (`pdf/Offre_Stage_EDF_RD_SEQUOIA_E7S_SAE.pdf`)
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
cœur scientifique du rapport. Le chapitre 4 discute les limites actuelles et les
perspectives. Le rapport se conclut par un bilan général du stage.

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

GemmaScope ([huggingface.co/google/gemma-scope](https://huggingface.co/google/gemma-scope))
est une collection de SAE préentraînés par DeepMind sur les modèles de la famille
Gemma, à plusieurs couches et plusieurs largeurs (nombre de features). Le projet utilise
GemmaScope-2 (variante pour Gemma-3) sur le residual stream, couche 24 pour le modèle
12B -- un choix de layer fondé uniquement sur la couverture Neuronpedia (ci-dessous),
jamais sur un critère d'interprétabilité mesuré empiriquement avant le balayage du
chapitre 3, §20-21, qui montre un layer alternatif (31) significativement meilleur.
Le choix de la **largeur** du SAE (parmi 16k/65k/262k/1m disponibles) est un
arbitrage documenté empiriquement dans ce projet, sur le critère de couverture des
labels Neuronpedia (fraction des features disposant d'une explication en langage
naturel) : **65k** offre la meilleure couverture pour ce modèle (87,8%, 57 551/65 536
features labellisées), devant 16k (82,6%, 13 535/16 384), très loin devant 262k (5,3%,
13 851/262 144, confirmant une première estimation manuelle ~10 000/262 144) ; la
largeur 1m n'est pas hébergée par Neuronpedia pour ce modèle (aucune donnée
disponible). 65k est donc retenue pour le run de mise à l'échelle final (chapitre 3).

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
   le LLM à chaque exemple et le rang de l'exemple dans l'ordre de sélection par
   magnitude d'activation (implémentation actuelle, `src/sae/judge.py::odd_one_out_judge`
   — un proxy de rang, pas la valeur d'activation continue réelle, et calculé sur les
   mêmes `pos_examples` que ceux ayant servi à l'auto-interprétation, pas un échantillon
   tenu à l'écart) — une feature bien détectée par le juge devrait aussi bien *classer*
   les exemples par intensité, pas seulement trouver l'intrus. Écart avec le protocole
   de Bills et al. (2023), qui utilise la magnitude réelle sur un échantillon distinct.

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
est une implémentation de **SAE Boost** (Koriagin et al., COLM 2025, *Teach Old SAEs
New Domain Tricks with Boosting*) — une des méthodes de domain adaptation identifiées
dans le cadrage initial du stage. `FrozenCoreResidualSAE`/
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
méthode d'explication à part entière — piste documentée au chapitre 4.

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
augmentées qui en dérivent, pas leur authenticité. Ce rapport n'emploie donc
jamais le terme "réel"/"réels" pour ce corpus.

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
l'entraînement en aval sans erreur explicite (perte `NaN` dès la première epoch). Le
projet utilise bf16 par défaut partout (même en local), qui partage la plage
d'exposant de fp32.

## Infrastructure de calcul

Cluster SLURM à 3 partitions GPU (a100, h100, h100-bis), sans accès réseau direct
depuis les nœuds de calcul — toutes les dépendances (modèles, données) doivent être
prépositionnées sur disque avant soumission d'un job. Cette contrainte a orienté
plusieurs choix : cache local des labels Neuronpedia (pas d'appel réseau au runtime),
environnement Python déjà provisionné sur disque (`.venv/bin/python` plutôt que `uv
run`, qui tenterait de re-résoudre l'environnement en ligne).

---

# Chapitre 3 — Démarche expérimentale et résultats

## 1. Problématique

Le pipeline complet (extraction d'activations → SAE → labellisation) fonctionnait de
bout en bout sans erreur, mais produisait un taux de succès faible au test
d'auto-interprétation odd-one-out des features d'extension (celles qui ne sont pas
couvertes par Neuronpedia et dépendent donc entièrement du juge LLM local pour être
labellisées). Sur le dernier run complet disponible avant cette phase du stage
(`results_v9_full`, Gemma-3-12B-it, 10 features jugées) : seules 2 features sur 10
(20%) passaient le test.

Question posée : **ce taux faible est-il dû à un budget d'entraînement (nombre de
tokens) insuffisant pour l'extension SAE, ou à un autre facteur ?**

## 2. Diagnostic

### 2.1. Élimination de l'hypothèse "features mortes"

Un budget d'entraînement insuffisant se traduirait typiquement par des features
d'extension qui ne s'activent jamais (`dead_feature`), auquel cas le juge ne peut même
pas être interrogé (`odd_one_out_judge` retourne directement `dead_feature` si moins de
3 exemples positifs sont disponibles). Or l'inspection du run `results_v9_full` montre
**0 feature morte sur les 10 jugées** : les features s'activaient bel et bien, avec des
exemples positifs disponibles. L'échec du test n'était donc pas un problème de features
inactives.

### 2.2. Inspection qualitative des exemples présentés au juge

L'inspection directe des `pos_examples` stockés pour les features non interprétables a
révélé le problème : les neuf exemples "positifs" présentés au juge pour une même
feature n'avaient, dans plusieurs cas, **aucun concept commun identifiable** — par
exemple un extrait sur un rappel produit iPad, un extrait sur le système carcéral
norvégien, un extrait de recette de cuisine et un extrait sur l'agriculture canadienne,
présentés ensemble comme activant fortement la même feature. Le test odd-one-out ne
peut, par construction, pas réussir dans ce cas : il n'y a pas de concept partagé à
partir duquel identifier l'intrus.

### 2.3. Cause racine

La lecture du code d'assemblage du corpus (`src/sae/saev5.py`, bloc `__main__`) a
montré que le corpus utilisé pour échantillonner le réservoir de résidus servant à
entraîner l'extension (`N_TOKENS_EXTRA_TRAIN` tokens tirés de `train_texts`) était
constitué **exclusivement** de textes génériques (FineWeb-2/Wikipedia filtrés par
mots-clés sur trois domaines substituts : énergie, sport, support client). Les mails
originaux et leurs variantes augmentées étaient chargés séparément (`email_texts`) et
utilisés **uniquement après l'entraînement**, pour une visualisation UMAP — jamais vus
par le SAE d'extension pendant son entraînement.

Autrement dit : le SAE d'extension apprenait des directions représentant des concepts
Wikipedia génériques et hétérogènes, jamais des concepts liés au contenu réel des
emails EDF. Les "exemples positifs" incohérents observés en 2.2 ne sont pas une
anomalie du juge, mais une conséquence directe et attendue de ce corpus
d'entraînement.

## 3. Correction

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

## 4. Protocole de validation

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

## 5. Résultats

### 5.1. Taux d'interprétabilité (test odd-one-out)

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

### 5.2. Interprétation

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

### 5.3. Qualité des labels obtenus

Contraste direct entre les labels obtenus avant et après correction, pour les features
qui passent le test :

- **Avant** (corpus generic) : labels générés à partir d'extraits Wikipedia sans lien
  avec le domaine (ex. artefacts de formatage — listes, ponctuation).
- **Après** (corpus emails) : `Réclamations Clients`, `Litiges Factures`, `Résiliation
  Énergie`, `Menace Résiliation`, `Demande Urgente`, `Problèmes énergie`, `Insatisfaction
  client` — des concepts directement alignés avec les objectifs métier du projet
  (détection d'urgence, détection d'intention, réclamations).

*Ces labels précis illustrent une catégorie de concepts récurrente, pas des features
individuellement stables : voir §12 pour la faible reproductibilité inter-seed des
labels exacts (28,2% de recouvrement), le taux agrégé restant la seule mesure fiable.*

### 5.4. Résultat additionnel : séparabilité linéaire des axes de perturbation

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

**Réserve** (`RESULTS_TESTS.md` §37) : un baseline TF-IDF sans aucun
contenu sémantique atteint 87,0% sur cette même sonde à 14 classes — soit ~93% du
signal ci-dessus déjà présent dans le simple texte brut, par templating lexical de la
génération augmentée, indépendamment de toute structure apprise par le SAE. Ce chiffre
ne peut donc pas, seul, être lu comme une preuve de compréhension sémantique ; la sonde
sur labels faibles indépendants du corpus augmenté (§7.2) reste la preuve la plus
fiable des objectifs urgence/intention.

## 6. Choix du solveur pour la sonde multi-classe

La sonde de classification multi-classe (§5.4) utilise `LogisticRegression`
avec sélection dynamique du solveur : `liblinear` pour le probe binaire
préexistant (énergie/sport), `lbfgs` (support natif du cas multinomial)
au-delà de deux classes — `liblinear` ne supporte que la classification
binaire dans les versions récentes de scikit-learn.

## 7. Suites données au diagnostic

Deux analyses complémentaires ont été menées après la validation du §5, pour répondre
plus directement aux objectifs métier du stage et à la limite identifiée au §5.4/§7
(taux résiduel non expliqué) — toutes deux réutilisent des activations déjà en cache,
sans calcul GPU supplémentaire pour la seconde.

### 7.1. Le résidu non-interprété est-il dû au protocole de jugement lui-même ?

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

### 7.2. Le SAE prédit-il l'urgence et l'intention sur des mails originaux ?

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
"détection d'intentions" énoncés dans le cadrage initial du projet : les codes
latents du SAE séparent très
nettement l'urgence et la réclamation, sur des mails originaux non augmentés, avec un
gain net important sur la baseline naïve. Le remboursement ne bat pas sa baseline
(déjà forte du fait du déséquilibre de classe, 85,5% de négatifs) — à interpréter
comme une limite du label faible par regex pour cette catégorie plutôt que comme un
échec du SAE, sans donnée annotée manuellement pour trancher entre les deux
hypothèses.

## 8. Qualité de l'explication document-level : fidélité et plausibilité

Question distincte des sections précédentes (qui évaluent une feature isolée ou une
capacité globale du corpus) : pour UN document donné, l'explication produite par le
pipeline (les features les plus actives et leurs labels) est-elle une bonne
explication ? Deux propriétés indépendantes ont été testées.

### 8.1. Fidélité (l'explication reflète-t-elle ce qui pilote réellement la décision ?)

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

### 8.2. Plausibilité (un lecteur trouve-t-il l'explication convaincante ?)

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

### 8.3. Protocole d'évaluation complet du dépôt

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

## 9. Limites de cette investigation

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

## 10. Ablation de mise à l'échelle du volume d'entraînement et de labellisation (v12)

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

### 10.1. Résultats du run combiné

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

### 10.2. Le rang par magnitude n'est pas un bon proxy de l'interprétabilité

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

### 10.3. Bug trouvé pendant l'analyse : chemin de labels figé sur 16k

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

### 10.4. Décomposition largeur / époques / capacité (ablations isolées)

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

## 11. Sanity check : le protocole d'évaluation distingue-t-il un SAE entraîné d'un décodeur aléatoire ?

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
| Interprétabilité odd-one-out | 45,3% (68/150) | 29,3% (44/150) | −16,0 points (z=2,86, p<0,01) |
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

## 12. Ablation de variance de seed

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

## 13. Biais multilingue du juge (français vs anglais traduit)

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

## 14. Ablation volume à grande échelle (25M tokens)

Suite directe du chapitre 1 ("Perspectives critiques") : le papier SAE Boost
montre qu'un SAE résiduel a besoin de 100-200M tokens pour converger sans
dégrader la performance générale — 50 à 100x au-dessus du volume testé dans
l'ablation initiale (§5, jusqu'à 2M). Le corpus emails+augmentés (~6M tokens)
étant insuffisant pour cette échelle, le réservoir de résidus est complété par
un filler échantillonné sur FineWeb2-fr sans filtre thématique (le filler
isole un effet de volume brut de tokens, pas de pertinence thématique),
ajouté **uniquement** au réservoir résiduel (`volume_filler_texts`), jamais
au corpus utilisé pour la sélection des features à labelliser ni pour la
sonde de classification, pour ne pas réintroduire le biais de domaine
diagnostiqué au chapitre 3. Le réservoir de résidus est memory-mapped sur
disque plutôt qu'alloué en RAM (`open_mmap_reservoir`, `src/sae/saev5.py`),
ce qui permet de viser des volumes proches du seuil du papier (jusqu'à 200M
tokens) sans demande mémoire proche de la capacité totale d'un nœud de
calcul.

**Résultat à 25M tokens** (`results_v13_ablation_volume25m`) : **81/150 =
54,0%** d'interprétabilité (odd-one-out), contre 45,3% pour le run principal
(500k tokens) et 44,7% pour l'ablation initiale à 2M — écart numérique de
+8,7 points mais **non significatif** (test z sur deux proportions, z=-1,50,
seuil \|z\|>1,96). `clf_acc_email_axes` recule légèrement (93,5% → 91,3%).
Porter le volume à 25M tokens (12x l'ablation initiale, toujours 50-100x en
dessous du seuil 100-200M de SAE Boost) ne change donc pas la conclusion
qualitative : le problème diagnostiqué en §2 était bien le domaine du
corpus, pas son volume brut. Cet écart directionnel (+8,7 points, non
significatif) est du même ordre de grandeur que celui observé
indépendamment pour l'ablation `K_EXTRA=5` (+9,4 points, §16, également non
significatif) — à prendre comme une piste à répliquer plutôt qu'un résultat
établi. Un run au seuil exact 100-200M reste à exécuter. Détail complet :
`RESULTS_TESTS.md` §23.3/§54.

## 15. Fidélité du steering (`steer_and_decode`) : jamais testé, résultat très hétérogène par intention

Le steering (`steer_activations`/`steer_and_decode`, `src/sae/sae_shared.py`)
existe dans le dépôt depuis le début mais n'était jamais réellement exercé : seule
`run_steering_demo` l'utilise, et uniquement pour une vérification géométrique
superficielle (cosinus avant/après suppression/amplification d'une feature, sans
tâche en aval). Le test d'ablation existant (§8) ablate déjà des features par
intention, mais directement dans l'espace des codes SAE, sans jamais appeler
`decode()`.

Question testée (`scripts/steering_fidelity_test.py`, zéro calcul LLM,
réutilise les activations en cache et le checkpoint entraîné de
`results_v10_emails_main`) : si on décode réellement le code stimulé vers l'espace
résidu puis qu'on RÉ-ENCODE ce résidu décodé, l'intervention (suppression des
top-10 features explicatives d'une intention) tient-elle à travers cet aller-retour ?

| Intention | Chute en place (témoin) | Chute steer_and_decode | Ratio |
|---|---|---|---|
| réclamation | 0,576 | 1,000 | 1,74× |
| remboursement | 1,000 | 0,016 | 0,02× |
| information | 1,000 | 0,004 | 0,00× |
| urgence | 0,646 | 0,584 | 0,90× |

Résultat hétérogène et contre-intuitif : le round-trip decode/encode neutralise
quasi entièrement l'intervention pour deux intentions sur quatre (remboursement,
information), la préserve pour urgence, et l'amplifie même pour réclamation.
Conclusion : `steer_and_decode` n'est pas un mécanisme d'intervention causale
fiable et prévisible à partir du simple test d'ablation en place — son effet
dépend fortement de la structure de corrélation entre features propre à chaque
intention. Détail complet (protocole, limite méthodologique du pooling par
document, fuite résiduelle mesurée) : `RESULTS_TESTS.md` §24.

## 16. Ablation `K_EXTRA=5` (SAE Boost)

Le papier SAE Boost trouve k=5 optimal dans son étude de sensibilité pour un
SAE résiduel — notre `K_EXTRA=32` par défaut n'avait jamais été testé en
dessous de cette valeur. Sur `results_v13_ablation_k_extra5` : **82/150 =
54,7%** d'interprétabilité contre 45,3% pour le run principal — écart de
+9,4 points, **non significatif** (z=-1,62) mais le plus proche du seuil
conventionnel de toutes les ablations de ce chapitre. `rho_sae` (fidélité de
reconstruction du résidu) recule sensiblement (0,906 → 0,849), cohérent avec
un budget de capacité par token plus faible. Direction cohérente avec
l'hypothèse du papier, mais à confirmer (cf. §14 pour la coïncidence
directionnelle avec l'ablation volume). Détail complet : `RESULTS_TESTS.md`
§25.

Deux seeds supplémentaires (7, 99) confirment la direction sur les 3 : 50,0%
et 55,3% d'interprétabilité, contre 54,7% pour le seed original et 45,3%
pour le baseline `K_EXTRA=32`. Test z groupé sur les 3 seeds combinés
(240/450 vs 68/150) : z=1,70, p≈0,089 — toujours sous le seuil conventionnel,
et probablement optimiste (le calcul groupé ignore la corrélation
intra-seed, et le baseline `K_EXTRA=32` n'a qu'un seul seed pour comparer).
La piste `K_EXTRA=5` se renforce (3/3 seeds dans la même direction) sans
pour autant passer d'hypothèse à répliquer à résultat établi. Détail :
`RESULTS_TESTS.md` §45.

## 17. Évaluation quantitative du retrieval Latent Terms

`src/sae/retrieval/latent_terms.py` (BM25 sur le vocabulaire latent d'un SAE
entraîné par pure reconstruction, Clavié et al. 2026) n'était exercé que par
inspection visuelle sur données de substitution. Protocole quantitatif
(`scripts/latent_retrieval_precision_eval.py`) : Precision@10/@20
contre les labels faibles d'intention (§8), sur 4 requêtes en paraphrase, comparé
à une baseline TF-IDF, sur les 3480 mails originaux.

| Intention | P@10 Latent Terms | P@10 TF-IDF |
|---|---|---|
| réclamation | 1,00 | 1,00 |
| remboursement | 1,00 | 0,00 |
| information | 1,00 | 0,20 |
| urgence | 0,00 | 0,80 |

Précision parfaite et nette supériorité sur TF-IDF pour 2 intentions sur 4
(remboursement, information) malgré des requêtes ne reprenant pas les mots exacts
du label — généralisation sémantique réelle. Échec complet sur urgence (0,00),
diagnostiqué précisément : la requête active bien des features latentes non
nulles, mais un seul document sur 3480 dans tout le corpus partage une
intersection non nulle avec elles — limite structurelle du BM25 sur vocabulaire
latent très parcimonieux (k=16), pas un bug ni un raté sémantique. Détail
complet : `RESULTS_TESTS.md` §26.

## 18. Ablation "échelle du modèle" : un effet dose-réponse net et significatif

Toutes les ablations précédentes gardent le modèle extracteur/juge fixé à
gemma-3-12b-it. Test avec gemma-3-1b-it et gemma-3-4b-it (+ leurs GemmaScope
dédiés) à la place de 12b-it :

| Modèle | Taux interp. | z vs 12b | `clf_acc_email_axes` |
|---|---|---|---|
| gemma-3-1b-it | **12,0%** (18/150) | 6,38 (significatif) | 88,2% |
| gemma-3-4b-it | **28,0%** (42/150) | 3,12 (significatif) | 92,0% |
| gemma-3-12b-it (run principal) | **45,3%** (68/150) | — référence | 93,5% |

**Effet dose-réponse net, monotone et significatif à chaque palier** — 12,0% →
28,0% → 45,3%, avec même la comparaison directe 1b vs 4b significative
(z=-3,46). C'est, de très loin, le plus fort effet mesuré dans tout ce projet :
tous les autres écarts (largeur, époques, capacité, volume, seed, K_EXTRA)
restent entre 1 et 9 points, tous non significatifs.

Indice que l'origine est plutôt la qualité du JUGE que celle des features
elles-mêmes : `clf_acc_email_axes` (séparabilité linéaire des axes de
perturbation, indépendante du juge LLM) suit une pente beaucoup plus douce
(88,2% → 92,0% → 93,5%, 5,3 points d'écart total contre 33,3 points pour le
taux d'interprétabilité). Lecture qualitative cohérente : le texte d'hypothèse
généré est confus pour 1B (inversion logique cause/conséquence), plus solide
pour 4B, pleinement cohérent pour 12B. Interprétation retenue : la capacité de
RAISONNEMENT du juge (formuler et vérifier un concept partagé entre 9 exemples)
est probablement le facteur limitant à petite échelle, plus que la qualité des
représentations latentes elles-mêmes — piste actionnable pour la suite :
séparer les rôles extracteur/juge pour isoler laquelle des deux capacités
domine réellement cet effet. Détail complet : `RESULTS_TESTS.md` §28.

Complète également le balayage de largeur du SAE core (16k/65k/262k, job
41487) : 46,7% à 262k, aucun écart significatif (z=-0,23) — confirme qu'aucune
largeur testée ne change l'interprétabilité, et souligne par contraste à quel
point l'effet de l'échelle du MODÈLE ci-dessus est hors norme parmi tous les
leviers testés dans ce projet. Détail complet : `RESULTS_TESTS.md` §29.

## 19. Balayage `MATRYOSHKA_DIM` (F2LLM) : dégradation graduelle, pas abrupte

`MATRYOSHKA_DIM` (troncature de l'embedding F2LLM, défaut 320) n'avait jamais
été varié — fait notable découvert en creusant : F2LLM-80M a `hidden_size=320`,
exactement égal au défaut, donc la "troncature" était un no-op pur pour ce
backbone dans toutes les comparaisons précédentes (§16), jamais remarqué.
4 runs sur F2LLM-160M (`hidden_size=640`), `MATRYOSHKA_DIM` ∈ {64, 128, 320,
640(complet)} :

| Dimension | Fraction du complet | `clf_acc_email_axes` |
|---|---|---|
| 64 | 10% | 72,8% |
| 128 | 20% | 75,0% |
| 320 (défaut) | 50% | 76,8% |
| 640 (complet) | 100% | **77,4%** |

Augmentation monotone mais à rendements très décroissants — 10% des dimensions
conservent déjà 94% de la performance du plein embedding. Dégradation
graduelle, pas de chute brutale à aucun point testé. Ni la documentation F2LLM
ni ce projet ne confirment un entraînement Matryoshka (MRL) au sens strict,
mais le comportement empirique est cohérent avec cette hypothèse. Conclusion
pratique : tronquer à 64-128 dimensions coûterait peu en performance pour un
gain de calcul/mémoire de 5-10x, si jamais nécessaire. Détail complet :
`RESULTS_TESTS.md` §31.

## 20. Balayage du layer d'extraction (12/31/41) : le choix par défaut n'est pas optimal

Le layer 24 (Pipeline 1) a toujours été choisi sur un seul critère : la
couverture des labels Neuronpedia (Chapitre 1), jamais sur un critère
d'interprétabilité mesuré empiriquement. Trois runs à facteur unique (même
protocole que le run principal, seul `LAYER` varie) comblent ce manque :

| Layer | Taux interp. | z vs layer 24 |
|---|---|---|
| 12 | 45,3% (68/150) | z=0,00, p=1,000 |
| 24 (défaut, run principal) | 45,3% (68/150) | — |
| 31 | **58,0% (87/150)** | **z=2,20, p=0,028** |
| 41 | 52,7% (79/150) | z=1,27, p=0,204 |

Layer 31 est le seul écart nominalement significatif du balayage (+12,7
points), et layer 12 reproduit le taux du layer 24 au feature près — une
coïncidence numérique, pas un signe de couplage entre les deux. Comme pour
le reste de ce chapitre (aucune correction multi-tests appliquée, cf. §21
ci-après pour le rappel), ce résultat doit être lu comme une piste à
répliquer sur un second seed avant adoption, pas comme un changement de
configuration déjà acquis -- mais c'est la première fois que le choix du
layer 24 apparaît potentiellement sous-optimal sur le critère qui compte
réellement pour ce projet. Détail complet : `RESULTS_TESTS.md` §51.

## 21. Balayage du point d'extraction (`resid_post` vs `attn_out` vs `mlp_out`)

Dernier volet du balayage : à layer fixe (24), comparer le residual stream
(`resid_post`, utilisé partout dans ce rapport) aux deux autres points de
hook publiés par GemmaScope-2 pour ce modèle (chapitre 1) :

| Point d'extraction | Taux interp. | z vs `resid_post` |
|---|---|---|
| `resid_post` (défaut, run principal) | 45,3% (68/150) | — |
| `mlp_out` | 52,7% (79/150) | z=1,27, p=0,204 |
| `attn_out` | 35,3% (53/150) | z=-1,77, p=0,078 |

Aucun des deux écarts individuels contre `resid_post` n'atteint la
significativité conventionnelle, mais **`mlp_out` contre `attn_out`
directement l'est nettement : z=3,02, p=0,0025**. Cohérent avec un a priori
raisonnable sur l'architecture : `attn_out` capte l'entrée de la projection
de sortie de l'attention (`self_attn.o_proj`), un espace multi-head pas
encore recombiné vers la dimension du residual stream, tandis que `mlp_out`
(sortie du MLP, déjà reprojetée) en reste proche. Illustration qualitative :
la feature choisie pour la démo de steering sur `attn_out` s'est labellisée
"variable assignment `p =`" -- un type de concept qui ne ressemble à rien
de ce qu'on observe sur `resid_post`/`mlp_out`, cohérent avec un espace
moins structuré sémantiquement à ce point précis du réseau. `attn_out` est
le point d'extraction le moins prometteur des cinq configurations testées
dans ce chapitre (layers 12/24/31/41 + `mlp_out`). Détail complet :
`RESULTS_TESTS.md` §53.

**Note transversale sur les §20-21** : ni ce balayage ni le reste du
chapitre 3 n'appliquent de correction pour comparaisons multiples (rappel
déjà fait au §11 pour le sanity check, et documenté comme lacune du chapitre
en `04_limites_et_perspectives.md`) -- sur ~20 ablations à ce stade, un seul
résultat significatif à p<10⁻⁹ (§18, échelle du modèle) et une poignée entre
0,03 et 0,10 (K_EXTRA=5, layer 31, `mlp_out` vs `attn_out`) sont à traiter
comme des pistes cohérentes entre elles, pas comme des résultats
individuellement établis.

## 22. Le core seul égale-t-il core+extension sur les métriques en aval ?

Question centrale pour juger si `FrozenCoreResidualSAE` ajoute réellement du
signal exploitable, ou seulement des dimensions supplémentaires sans effet :
sur les mêmes activations en cache, mêmes folds de validation croisée, comparer
le SAE core seul (16384 dimensions) à core+extension (17408 dimensions) sur la
silhouette (structure de cluster, axes email) et deux sondes de classification
déjà utilisées dans ce chapitre (axes email 14 classes, diffing energy/sports).

| Métrique | Core seul | Core+extension |
|---|---|---|
| `silhouette` | 0,0086430470 | 0,0086430488 |
| `clf_acc_email_axes` | 93,67% | 93,59% |
| `clf_acc_sae` (energy/sports) | 60,0% | 60,0% |

Un test de McNemar apparié (mêmes documents, mêmes folds) ne détecte aucune
différence significative sur la sonde email (p=0,26) ; sur la sonde
energy/sports, les deux conditions classent les 600 documents de façon
**strictement identique** (zéro désaccord, p=1,0). L'extension n'apporte donc
ni gain ni dégradation mesurable sur ces sondes linéaires -- ni pollution du
signal du core, ni signal supplémentaire linéairement décodable pour ces deux
tâches précises. Cohérent avec le reste du chapitre : la valeur mesurée de
l'extension dans ce projet tient à l'interprétabilité individuelle de ses
features (45,3% au test odd-one-out, chapitre 3 §2-5) et à la couverture de
concepts absents du core, pas à un gain de séparabilité linéaire en aval que
le core seul n'atteignait pas déjà. Détail complet : `RESULTS_TESTS.md` §55.

---

# Chapitre 4 — Limites et perspectives

## Limites actuelles

### Taux d'interprétabilité résiduel (~55% de features non interprétées)

Ce résidu n'est pas dû au volume de tokens du corpus (`03_experiences_et_resultats.md`).
Il est en revanche en bonne partie attribuable au bruit du protocole de jugement :
en répétant la question odd-one-out 5 fois par feature avec un ordre de mélange
différent à chaque fois (`RESULTS_TESTS.md` §13.1), seulement 30,7% des features
obtiennent une décision unanime sur les 5 répétitions ; le taux agrégé
d'interprétabilité bouge peu (45,3%→48,7%) mais 31,3% des features changent
individuellement de statut selon l'ordre de présentation. Une partie substantielle
du résidu non interprété est donc due au bruit du protocole (décision greedy
unique, sensible à l'ordre), pas nécessairement à un défaut réel des features —
un vote majoritaire sur plusieurs répétitions serait préférable comme protocole
par défaut à une seule décision greedy.

Le taux de 45,3% cité dans ce rapport n'est pas non plus robuste au choix du
modèle juge : remplacer gemma-3-12b-it par gemma-3-4b-it comme juge (mêmes
features, mêmes exemples) fait chuter le taux mesuré à 24,7%. Une partie de ce
que ce rapport attribue à la qualité des features apprises dépend donc aussi de
la capacité du juge à raisonner sur 9 exemples, pas uniquement des features
elles-mêmes — cohérent avec l'effet dose-réponse de l'échelle du modèle
documenté au §18 de `03_experiences_et_resultats.md`.

L'effet "domaine, pas volume" qui sous-tend l'ensemble du rapport
(`03_experiences_et_resultats.md` §2-5) est statistiquement confirmé à n
apparié : 30,0% (45/150) sur corpus générique contre 45,3% (68/150) sur corpus
emails, z=2,74, p≈0,006 (`RESULTS_TESTS.md` §46).

Piste encore non résolue : la **qualité du contrôle négatif**
(`build_feature_examples_with_control`) reste un document sous un quantile bas
d'activation pour la feature testée, pas nécessairement un contre-exemple
"propre" conceptuellement — une feature réellement monosémantique pourrait
échouer au test si le contrôle négatif choisi partage accidentellement une
propriété de surface avec les exemples positifs. Les ablations de capacité de
l'extension (`D_EXTRA`/`K_EXTRA`, doublées ensemble, `K_EXTRA=5` seul,
`D_EXTRA=2048` seul) ne changent significativement le taux d'interprétabilité
dans aucune configuration testée une fois le corpus corrigé (`RESULTS_TESTS.md`
§17.5/§25/§27).

Une piste plus fondamentale a également été testée : le protocole de la
référence *Interpretable Embeddings with Sparse Autoencoders* (Appendix C) ne
gate jamais la labellisation derrière un test odd-one-out, il génère toujours
un label par contraste direct (10 positifs + 10 négatifs). Sur les 82 features
originellement rejetées par notre gate, la génération contrastive directe
produit un label spécifique et qualitativement plausible pour la totalité
d'entre elles (`scripts/contrastive_labeling_test.py`, `RESULTS_TESTS.md`
§15.4) — exemples : `Mise en service énergie`, `Numéro de contrat`, `Demande
de résiliation`, `Informations bancaires`, `Sentiment d'urgence`. Limite :
le champ `confident` auto-rapporté par le LLM reste à `true` pour 150/150
features dans les deux runs testés — pas un signal de qualité fiable en
l'état. Une validation systématique (comptage sur les 82 labels, pas un
échantillon) montre que 45% partagent leur label avec un autre, et un cas de
feature quasi-morte (freq=0%) reçoit malgré tout un label confiant — le
"100% de récupération" apparent est un artefact de complaisance du juge, pas
un signal de qualité. Le protocole odd-one-out reste la référence retenue
dans ce rapport ; la labellisation contrastive directe n'est pas intégrée au
pipeline de production (changerait le chiffre central du rapport, 45,3%,
sans validation à l'échelle comparable).

### Rigueur statistique des comparaisons d'ablation

Deux comparaisons (biais multilingue, robustesse du juge) testent en réalité
les mêmes 150 features sous deux conditions — un plan apparié — mais avaient
été analysées avec un test à deux proportions indépendantes plutôt que
McNemar. Le recalcul avec le test approprié donne les mêmes conclusions
(p=0,894 et p=0,560, non significatifs), mais méthodologiquement plus
correct (`RESULTS_TESTS.md` §30). Aucune correction pour comparaisons
multiples n'est appliquée aux ~15 tests d'ablation de ce chapitre
(contrairement au diffing par feature, qui utilise déjà Benjamini-Hochberg) —
sans conséquence sur les conclusions actuelles (le seul résultat
significatif, l'échelle du modèle, l'est à p<10⁻⁹), mais une lacune pour
toute extension future où des résultats plus proches du seuil pourraient
apparaître. L'effet dose-réponse de l'échelle du modèle est par ailleurs
confirmé par un test de tendance dédié (Cochran-Armitage, p≈1,6×10⁻¹⁰), plus
adapté qu'une série de tests par paires à un plan à niveaux ordonnés.

### Comparaisons avec l'état de l'art

Une comparaison chiffrée avec SAELens (`scripts/saelens_numeric_comparison.py`,
`docs/references.md`), sur le même SAE natif sae-lens et les mêmes
activations réelles, entre notre formule de variance expliquée et les deux
formules maintenues par `sae_lens.evals`, révèle un désaccord numérique
important entre les trois (0,41 / 0,83 / 1,00) causé par les activations
massives de Gemma-3 — la formule qui somme sur les dimensions avant de
normaliser est mécaniquement dominée par une seule dimension outlier et
rapporte une variance quasi-totalement expliquée, sans rapport avec la
qualité de reconstruction réelle. Ne jamais publier un score de variance
expliquée sur ce modèle sans préciser la formule exacte utilisée. La
comparaison avec `interp_embed` reste partielle (test optionnel dépendant
d'une installation non faite par défaut).

`FrozenCoreResidualSAE`/`ExtendedSAE` est une implémentation de SAE Boost
(Koriagin et al., COLM 2025) : même architecture, un SAE résiduel entraîné
sur l'erreur de reconstruction d'un core gelé, sommé à l'inférence. Deux
écarts avec le papier ont été testés : (1) leur étude de sensibilité montre
qu'un `K_EXTRA` plus faible (k=5 optimal chez eux, contre 32 dans ce projet)
améliore l'interprétabilité au prix d'un peu d'EV domaine — direction
cohérente sur ce corpus (54,7% vs 45,3%, +9,4 points) mais non significatif
(z=-1,62, `RESULTS_TESTS.md` §25) ; (2) leur étude montre qu'un budget de
100-200M tokens est nécessaire pour que le SAE résiduel converge sans
dégrader la performance générale (jusqu'à -31% d'EV en dessous de 100M) —
testé partiellement à 25M tokens (12x l'ablation initiale, toujours 50-100x
en dessous du seuil du papier, `RESULTS_TESTS.md` §23.4) : même conclusion
qualitative (+8,7 points, non significatif). Le run au seuil exact
100-200M reste à exécuter — le réservoir de résidus, initialement une
allocation RAM proportionnelle au volume de tokens (limitant la faisabilité
pratique d'un tel run), est désormais memory-mapped sur disque
(`RESULTS_TESTS.md` §54), rendant ce run schedulable. Aucune comparaison
chiffrée avec les baselines alternatives du papier (Extended SAE
random/most-active init, SAE Stitching, full fine-tuning) n'a été menée sur
ce projet.

Une question plus fondamentale, posée par *Sanity Checks for Sparse
Autoencoders* (Korznikov et al., 2026) : un SAE dont le décodeur est figé à
une initialisation aléatoire (jamais entraîné) égale, dans leur étude, un
SAE réellement entraîné sur interprétabilité automatique, sparse probing et
édition causale. Reproduit sur ce projet (`FrozenDecoderExtendedSAE`,
`SANITY_CHECK_FROZEN_DECODER=1`, `RESULTS_TESTS.md` §19) : résultat nuancé —
l'interprétabilité odd-one-out résiste bien (45,3% entraîné vs 29,3% figé
aléatoire, écart significatif) mais la classification en aval y résiste
beaucoup moins (93,5% vs 91,2%), répliquant partiellement le constat du
papier.

### Biais de génération résiduel dans le corpus augmenté

20,6% des mails augmentés contenaient une ligne "Objet :"/"Subject :" que les
mails originaux n'ont pas (`RESULTS_TESTS.md` §14.1). Fix appliqué au
chargement (`load_augmented`, 0,0% après fix) ; effet mesuré sur le diffing
complet : réduction de 65% et 49% du nombre de features "significatives" sur
les deux axes orthographiques (les plus confondables avec l'artefact), effet
modéré sur l'urgence (−7,5%/−3,1%), négligeable ailleurs. Contrairement à
l'observation initiale sur l'échantillon test à 60 mails (où l'artefact
dominait 8/13 classements), l'artefact ne domine pas la majorité des
features significatives à l'échelle du corpus complet (0,21% des features
significatives portaient un label "Subject:"/"Objet:", avant comme après) —
son effet réel est plus circonscrit qu'attendu.

### Retrieval par propriétés et clustering ciblé

`property_based_retrieval` et `targeted_clustering_by_axis` sélectionnent les
latents pertinents pour une requête par similarité d'embedding
(`select_latents_by_similarity`, `src/sae/saev5.py`, embeddings **bge-m3**),
pas par matching de sous-chaîne littérale — vérifié empiriquement que ce
dernier rate des labels sémantiquement liés mais formulés différemment, et
retourne des faux positifs (mot partagé sans rapport de sens). bge-m3 est
retenu après comparaison à F2LLM (bons résultats sur une requête, résultats
sans rapport sur une autre — pooling dernier-token mal adapté à des labels
courts en contexte cross-lingue), `RESULTS_TESTS.md` §15.1-15.2. Validé sur
les activations déjà en cache (ne change pas la reconstruction elle-même,
seulement la sélection de latents en aval).

### Corrélations "intéressantes" entre features

`find_interesting_pairs` (`src/analysis/cooccurrence.py`) filtre les paires à
NPMI élevé et similarité sémantique des labels faible (méthode interp_embed
§4.2/Appendix E.1), calculé sur `results_v10_emails_main` sans réextraction
Gemma-3 (`scripts/compute_interesting_correlations_retro.py`,
`RESULTS_TESTS.md` §15.3/§16.3) : seulement 3 paires retenues sur 26 579
arêtes du graphe (3 395 nœuds), et 2 des 3 impliquent une feature non
labellisée — résultat peu exploitable en l'état (impossible de juger la
pertinence d'une corrélation quand un des deux côtés n'a pas de label).
Élargir la plage de fréquence ou prioriser les paires où les deux features
sont labellisées reste à faire.

### Qualité de l'explication document-level

Question distincte de tout ce qui précède (qui évalue une feature isolée ou
une capacité globale) : pour UN document donné, l'explication produite
(features actives + labels) est-elle bonne ?
- **Fidélité** (`scripts/explanation_fidelity_test.py`, ablation) : chute de
  58 à 100 points de probabilité en ablatant les 10 features "explicatives",
  chute quasi nulle (<0,4 point) en ablatant des features aléatoires ou peu
  contributives (ratios de 250× à 576 000× selon l'intention) — l'explication
  porte réellement la décision.
- **Plausibilité** (`scripts/explanation_plausibility_test.py`, choix forcé,
  juge Gemma-3-12B-it) : 71,7% (43/60) de choix corrects contre 50% au hasard
  (p < 0,001) — significativement au-dessus du hasard, mais loin d'être
  parfait, cohérent avec le taux d'interprétabilité résiduel.

Détail complet : `RESULTS_TESTS.md` §16.1-16.2, `03_experiences_et_resultats.md`
§8, dashboard (onglet "Explication (fidélité/plausibilité)").

### Backbone d'embedding Pipeline 2 : F2LLM-80M vs -330M

Résultat mixte (`RESULTS_TESTS.md` §16.5) : -330M reconstruit légèrement
mieux (NMSE −7,5%) et sépare un peu mieux le corpus de diffing générique (+2
points), mais sépare légèrement moins bien les axes email (−2,2 points, la
métrique la plus proche des objectifs métier). Aucun écart n'est de l'ordre
d'un problème majeur ; pas de justification claire pour préférer l'un à
l'autre sur ce projet.

### Facteurs non contrôlés dans le corpus augmenté

Les variantes augmentées sont générées par le même modèle (Gemma-3-12B-it)
qui sert aussi de juge d'interprétation et d'extracteur d'activations. Un
style de génération propre au modèle pourrait constituer un facteur de
confusion partagé entre "ce qui rend une variante reconnaissable comme
augmentée" et "ce que le SAE apprend à détecter". Trois vérifications
indépendantes (`RESULTS_TESTS.md` §44/§48/§50/§52) convergent vers une
réponse négative : aucune corrélation significative entre la part
d'exemples "augmentés" parmi les 9 exemples positifs d'une feature et son
statut interprétable (p=0,418) ; des features core totalement étrangères à
cette boucle (labellisées indépendamment par Neuronpedia) présentent le même
taux élevé d'exemples augmentés, confirmant qu'il s'agit d'une propriété du
corpus (92% augmenté) et non d'un biais spécifique au pipeline ; un
re-jugement complet des 150 features avec le même SAE et le même juge, mais
des exemples positifs restreints aux mails originaux uniquement (zéro texte
généré par Gemma vu par le juge), donne 44,7% (67/150) contre 45,3% (68/150)
en référence — écart non significatif (z=-0,12, p=0,908). Comme pour les
autres perturbations testées (ordre de présentation, langue), le statut
d'une feature individuelle reste bruité (55,3% d'accord, 44,7% de bascule)
mais le taux agrégé est stable. **La boucle auto-référentielle
juge/générateur n'explique pas le taux d'interprétabilité mesuré dans ce
rapport.**

En contrepoint, un aspect qui est contrôlé : `src/data/augmentation.py::validate`
rejette une variante générée si elle est trop courte (<30 caractères), si son
ratio de longueur par rapport au mail original sort de l'intervalle
[0,4 ; 2,5], si elle est strictement identique au parent, ou (sauf pour
l'axe orthographe) si elle perd une entité numérique du mail d'origine
(montant, numéro de compte/contrat, date). Sur l'ensemble du corpus augmenté
(45 240 générations), 11,7% (5291) sont rejetées par ce garde-fou — motif de
rejet conservé pour audit (`facts_lost=[...]`, `length_ratio=...`, etc.).
Sans impact sur les résultats de ce rapport : `load_augmented` filtre ces
lignes rejetées avant qu'elles n'atteignent le pipeline SAE.

### Fidélité du steering comme méthode d'explication

`steer_activations`/`steer_and_decode` (`src/sae/sae_shared.py`) — mesuré en
faisant réellement décoder puis ré-encoder un code stimulé (suppression des
top-10 features explicatives d'une intention, `RESULTS_TESTS.md` §24) :
résultat très hétérogène selon l'intention — le round-trip neutralise quasi
entièrement l'intervention pour 2 intentions sur 4 (ratio 0,00-0,02× vs.
l'ablation en place), la préserve pour une troisième (0,90×), et l'amplifie
pour la dernière (1,74×). `steer_and_decode` n'est donc pas un mécanisme
d'intervention causale fiable et prévisible à partir du simple test
d'ablation en place — son effet dépend fortement de la structure de
corrélation entre features propre à chaque intention.

### Biais multilingue et variance de seed

Pas de différence significative d'interprétabilité entre français et anglais
traduit (46,9% vs 45,5%, z=0,24, `RESULTS_TESTS.md` §22), mais 38,6% des
features changent individuellement de statut selon la langue — un taux de
bruit supérieur à celui déjà mesuré pour le réordonnancement des exemples
(§13.1, 31,3%). Pas de biais systématique détecté envers l'anglais sur ce
test précis (l'hypothèse alternative — entraîner le SAE sur un corpus
anglais natif plutôt que de traduire la vue du juge — reste à tester).

Le taux d'interprétabilité agrégé est stable entre deux seeds d'entraînement
du SAE (45,3% vs 47,3%, `RESULTS_TESTS.md` §21), mais seulement 28,2% de
recouvrement exact des labels individuels obtenus entre les deux seeds — les
features individuelles ne sont pas reproductibles à l'identique d'un seed à
l'autre, contrairement au taux agrégé. Les exemples de features cités dans
ce rapport (chapitre 3) sont donc représentatifs d'une catégorie récurrente
de concepts, pas des atomes stables du dictionnaire.

### Pistes non implémentées issues de la littérature

- **Feature splitting/absorption comme cause possible du résidu
  non-interprété** (*Matryoshka SAEs*, Bussmann et al. 2025) : leur travail
  montre qu'agrandir simplement un dictionnaire SAE (notre ablation
  capacité, `D_EXTRA` 1024→2048, `03_experiences_et_resultats.md` §10.4)
  peut dégrader la qualité des features de haut niveau par
  fragmentation/absorption plutôt que de mieux couvrir le domaine — cohérent
  avec l'absence de gain d'interprétabilité observée sur cette ablation.
  Leur solution (dictionnaires SAE emboîtés, entraînés simultanément à
  plusieurs tailles) n'est pas implémentée. À ne pas confondre avec
  `MATRYOSHKA_DIM` (`src/config.py`), qui ne concerne que la troncature des
  embeddings F2LLM, un mécanisme complètement différent
  (`docs/references.md`).
- **Entraînement supervisé conjoint SAE+classifieur** (*ClassifSAE*, Le
  Bail et al. 2025) : ce projet extrait des concepts de façon totalement non
  supervisée puis les relie à la classification par une sonde post-hoc
  (`downstream_classification`). ClassifSAE propose d'entraîner le SAE
  conjointement avec un classifieur (pénalité de parcimonie sur le taux
  d'activation), spécifiquement pour concentrer les concepts pertinents à la
  tâche — directement aligné avec les objectifs de détection
  d'urgence/d'intention. Non implémenté : nécessiterait une nouvelle boucle
  d'entraînement jointe.

## Perspectives

- Clarifier avec le commanditaire si le système, à terme, constitue une
  décision automatisée au sens de l'article 22 du RGPD ou reste une aide à
  la décision (un humain reste dans la boucle) — un fait de conception du
  déploiement final, pas une question expérimentale, mais qui détermine le
  cadre légal applicable et n'est pas encore tranché.
- Passer le vote majoritaire en protocole par défaut de `odd_one_out_judge`
  (actuellement une fonction séparée, `scripts/judge_robustness_check.py`, à
  fusionner dans `src/sae/judge.py`).
- Implémenter la métrique de variance expliquée robuste aux outliers
  proposée en comparaison SAELens (médiane des ratios par token).
- Remplacer les intervalles de confiance approximatifs (Wald) encore utilisés
  ponctuellement au chapitre 3 par `proportion_with_ci` (Wilson,
  `src/analysis/stats.py`), déjà la référence pour le reste du module
  statistique.
- Poursuivre la factorisation de `src/sae/saev5.py` (séparer entraînement et
  extraction, réduire un fichier monolithique) — dette technique qui
  n'affecte pas la validité des résultats mais complique la maintenance.
- Étendre le sanity check "Frozen Decoder" au Pipeline 2 (`PhraseLevelSAE`,
  entraîné from-scratch, jamais testé contre un décodeur figé) ; envisager
  les métriques plus exigeantes du papier (AutoInterp par
  description+détection sur échantillon non vu, sparse probing SAEBench) en
  remplacement de la sonde de classification actuelle, dont ce sanity check
  a montré la faible sensibilité.
- Tester des dictionnaires SAE emboîtés (*Matryoshka SAEs*) pour l'extension
  P1, comme alternative à l'ablation capacité simple pour réduire le résidu
  non interprété.
- Entraîner un SAE supervisé conjointement avec un classifieur (*ClassifSAE*)
  pour la détection d'urgence/intention, en alternative à la sonde post-hoc
  actuelle.
- Étendre le test de plausibilité de l'explication document-level au
  Pipeline 2, et à un échantillon plus large que 60 documents pour resserrer
  l'intervalle de confiance.
- Caractériser ce qui distingue les cas où le round-trip du steering "tient"
  de ceux où il ne tient pas (structure de corrélation entre features ?
  spécificité de l'intention ?) sur un échantillon plus large de
  features/intentions.
- Exécuter le run à 200M tokens (seuil exact SAE Boost, désormais
  schedulable) et comparer chiffré aux baselines alternatives du papier
  (Extended SAE random/most-active init, SAE Stitching, full fine-tuning) ;
  répliquer `K_EXTRA=5` et le volume 25M sur plusieurs seeds pour trancher si
  l'écart directionnel commun aux deux (+8,7/+9,4 points, chacun non
  significatif seul) reflète un effet réel.
- Comparer `find_interesting_pairs` à des biais/artefacts réels connus du
  corpus (ex. le biais "Objet :" avant correction) pour valider
  empiriquement la méthode sur ce projet, à la manière de la validation par
  injection synthétique du papier de référence (§4.2, Appendix E.2) ;
  élargir la plage de fréquence de `cooccurrence_graph` pour augmenter le
  rappel.
- Évaluer un jeu de labels d'urgence/intention annotés manuellement plutôt
  que des labels faibles par regex pour la sonde de détection d'urgence
  (`scripts/intent_urgency_probe.py`), et étendre au Pipeline 2.

---

# Conclusion générale

## Bilan par rapport aux objectifs initiaux

Le stage visait à rendre fonctionnelle et exploitable une plateforme d'analyse
interprétable de mails clients EDF fondée sur des Sparse Autoencoders. Les deux
pipelines (Gemma-3 + GemmaScope étendu ; F2LLM + SAE dédié) fonctionnent de bout
en bout sur le corpus original, avec des résultats quantifiés et reproductibles
sur l'ensemble des capacités visées par l'énoncé initial :

- **Détection d'urgence et d'intention** : séparabilité linéaire forte sur les axes
  synthétiques (93,5%/79,3% selon le pipeline — à lire avec la réserve de
  `03_experiences_et_resultats.md` §5.4, ~93% de ce chiffre étant reproductible par un
  baseline TF-IDF sans sémantique) et gain net mesuré sur des labels faibles
  indépendants tirés de mails originaux non augmentés (+27,0 points sur l'urgence,
  +42,6 points sur la réclamation par rapport à la baseline naïve — preuve la plus
  fiable des deux, non affectée par cette réserve).
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

Le résultat le plus central du stage reste le diagnostic du chapitre 3 : le taux
d'auto-interprétation des features, initialement très faible (20%), n'était pas
limité par le volume d'entraînement mais par une erreur de conception du corpus
d'entraînement de l'extension — un exemple concret de la valeur d'une démarche
d'ablation contrôlée plutôt que d'une intuition non testée ("il faut probablement plus
de données"). À distinguer de l'effet le plus fort mesuré en valeur absolue
(§18, dose-réponse de l'échelle du modèle, p<10⁻⁹) : les deux résultats
répondent à des questions différentes — l'un explique pourquoi le pipeline
fonctionne sur ce corpus, l'autre identifie le levier le plus déterminant
parmi tous ceux testés.

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

Le chapitre 4 détaille les limites et pistes de poursuite technique. Deux directions
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

---

# Bibliographie

*Note de rédaction* : les références ci-dessous listent les travaux effectivement
mobilisés pendant le stage (méthode, comparaison chiffrée, ou lecture critique). Les
métadonnées bibliographiques complètes (auteurs exacts, venue, année) des articles
disponibles uniquement sous forme de PDF local (`pdf/`) sont à vérifier/compléter
avant intégration dans la version finale déposée, conformément à la remarque déjà
présente dans `report/README.md`.

## Références académiques

- Jiang, N., Sun, X. et al. (2025). *Interpretable Embeddings with Sparse
  Autoencoders: A Data Analysis Toolkit*
  ([arXiv:2512.10092](https://arxiv.org/abs/2512.10092), ICML 2026,
  `pdf/InterpretableSAE_Embeddings.pdf`,
  [github.com/nickjiang2378/interp-embed](https://github.com/nickjiang2378/interp-embed)).
  Référence méthodologique principale du stage : protocole de labellisation
  contrastive (Appendix C), détection de corrélations "intéressantes" (§4.2,
  Appendix E), retrieval par propriétés et clustering ciblé par similarité
  d'embedding (§4.3/4.4, Appendix F.1) — méthodes reprises et discutées au
  chapitre 4.
- Bills, S. et al. (2023). *Language models can explain neurons in language models*
  (OpenAI). Origine de la mesure ρ_interp (corrélation de Spearman entre intensité
  jugée par un LLM et activation réelle) utilisée dans le protocole
  d'auto-interprétation local (`src/sae/judge.py`) — implémentation locale utilisant
  un proxy de rang plutôt que l'activation continue réelle, cf. `01_etat_de_lart.md`.
- Karvonen, A. et al. (2025). *SAEBench: A Comprehensive Benchmark for Sparse
  Autoencoders in Language Model Interpretability*
  ([arXiv:2503.09532](https://arxiv.org/abs/2503.09532), ICML 2025). Source du
  protocole odd-one-out cité au chapitre "État de l'art" et de la "sparse probing
  SAEBench" évoquée comme piste alternative en limites.
- Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., Bloom, J.
  (2024). *A is for Absorption: Studying Feature Splitting and Absorption in Sparse
  Autoencoders* ([arXiv:2409.14507](https://arxiv.org/abs/2409.14507)). Article
  définissant le phénomène de "feature absorption", distinct du "feature splitting" —
  pertinent pour le résidu non-interprété (`04_limites_et_perspectives.md`), dont
  Matryoshka SAEs (Bussmann et al. 2025, ci-dessous) est présenté comme correctif
  possible.
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
  ([arXiv:2407.14435](https://arxiv.org/abs/2407.14435), `pdf/jumpRELU.pdf`).
  Architecture du SAE core GemmaScope-2 (Pipeline 1).
- Bussmann, B., Nabeshima, N., Karvonen, A., Nanda, N. (2025). *Learning Multi-Level
  Features with Matryoshka Sparse Autoencoders*
  ([arXiv:2503.17547](https://arxiv.org/abs/2503.17547), `pdf/Matryoshka.pdf`).
  Piste non implémentée pour le résidu non-interprété (chapitre 4) — à ne pas
  confondre avec `MATRYOSHKA_DIM` du projet (cf. `docs/references.md`).
- Le Bail, M., Dentan, J., Buscaldi, D., Vanier, S. (2025). *Unveiling
  Decision-Making in LLMs for Text Classification: Extraction of Influential and
  Interpretable Concepts with Sparse Autoencoders*
  ([arXiv:2506.23951](https://arxiv.org/abs/2506.23951),
  `pdf/UnveilingDecision-MakinginLLMsforTextClassification.pdf`). Introduit
  ClassifSAE (SAE supervisé conjoint SAE+classifieur) — piste non implémentée,
  directement pertinente pour les objectifs détection d'urgence/intention
  (chapitre 4).
- Shu, D., Wu, X., Zhao, H. et al. (2025). *A Survey on Sparse Autoencoders:
  Interpreting the Internal Mechanisms of Large Language Models*
  ([arXiv:2503.05613](https://arxiv.org/abs/2503.05613), EMNLP 2025 Findings,
  `pdf/SurveySAE.pdf`). Taxonomie explications input-based/output-based et
  métriques structurelles/fonctionnelles, utilisée pour cadrer le chapitre 1.
- Resck, L., Augenstein, I., Korhonen, A. (2025). *Explainability and
  Interpretability of Multilingual Large Language Models: A Survey* (EMNLP 2025,
  `pdf/2025.emnlp-main.1033.pdf`). Cité pour le biais multilingue potentiel du juge
  d'auto-interprétation (corpus français) — **mesuré** au chapitre 3, §13 : pas de
  différence significative français/anglais (46,9% vs 45,5%, z=0,24), mais 38,6%
  des features changent de statut interprétable selon la langue.
- *Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse
  Languages* ([arXiv:2507.11230](https://arxiv.org/abs/2507.11230)). Motive le test
  de biais multilingue ci-dessus (features SAE potentiellement langue-spécifiques,
  facteur de confusion pour un juge interrogé hors de la langue du corpus).
- *Unstable Features, Reproducible Subspaces*
  ([arXiv:2606.12138](https://arxiv.org/abs/2606.12138)) et *Toward Identifiable
  Sparse Autoencoders* ([arXiv:2605.31245](https://arxiv.org/abs/2605.31245)).
  Montrent que les features individuelles d'un SAE varient selon la graine
  d'entraînement, le sous-espace de bas rang restant seul reproductible — **testé**
  au chapitre 3, §12 (ablation de variance de seed) : taux agrégé stable (45,3% vs
  47,3%, non significatif) mais seulement 28,2% de recouvrement exact des libellés
  de features entre les deux graines, confirmant la thèse des deux papiers.
- Clavié, B., Lee, S., Shakir, A., Kato, M. P. (2026). *Latent Terms: Dense
  Retrievers Contain Trivially Extractable BM25-ready Zipfian Vocabularies*
  ([arXiv:2605.29384](https://arxiv.org/abs/2605.29384)). Méthode de retrieval BM25
  sur le vocabulaire latent d'un SAE — **implémentée et évaluée quantitativement**
  au chapitre 3, §17 (Precision@10 parfaite sur 3 intentions/4, échec structurel
  diagnostiqué sur la 4ᵉ, `RESULTS_TESTS.md` §26).
- Beckmann, P., Queloz, M. (2026). *Mechanistic Indicators of Understanding in Large
  Language Models* ([arXiv:2507.08017](https://arxiv.org/abs/2507.08017),
  `pdf/MechanisticIndicatorsinLLM.pdf`). Cadrage philosophique cité en introduction.
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
  [huggingface.co/google/gemma-scope](https://huggingface.co/google/gemma-scope) —
  poids des SAE préentraînés sur Gemma-3 (lien GitHub `google-deepmind/gemma-scope`
  précédemment cité corrigé : ce dépôt n'existe pas, les poids sont hébergés sur
  Hugging Face).
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
