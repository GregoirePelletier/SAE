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
conception du corpus d'entraînement (uniquement générique, sans emails réels) —
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
directe de "ce que le modèle a compris du texte". Les Sparse Autoencoders, popularisés
récemment par les travaux d'interprétabilité mécaniste (Anthropic, DeepMind), proposent
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
est l'apport spécifique du projet par rapport à un usage "out of the box" de
GemmaScope ou de SAELens ; elle est documentée comme telle dans `Context.md` (règle
n°3 : "Conserver `FrozenCoreResidualSAE` — spécifique au projet").


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
                    │  Mails EDF réels + variantes          │
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

1. **Nouveau corpus principal** : mails réels (`Mails.tsv`) + variantes augmentées
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

### 3.7.2. Le SAE prédit-il l'urgence et l'intention sur des mails réels ?

Le résultat du §5.4 (séparabilité des axes d'augmentation synthétiques) a été
complété par un test sur des labels **indépendants du corpus augmenté** : des labels
faibles par expression régulière, déjà calculés sur le texte brut des mails réels
(`src/data/dataset.py::INTENT_KEYWORDS_FR` — réclamation, résiliation, remboursement,
information, urgence), appliqués aux 3 300 mails réels du split d'entraînement
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
nettement l'urgence et la réclamation, sur des mails réels non augmentés, avec un
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

Test par ablation (`scripts/explanation_fidelity_test.py`) : sur 200 mails réels par
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
bien les axes email réels (−2,2 points), la métrique la plus proche des objectifs
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

*[Résultats en cours de calcul au moment de la rédaction de cette version du
rapport — jobs SLURM chaînés par dépendance (run principal → diffing par axe →
robustesse du juge, sonde intention/urgence, fidélité, plausibilité, labellisation
contrastive → consolidation). Cette section sera complétée avec le taux
d'interprétabilité obtenu, le NMSE de reconstruction des deux pipelines, et la
comparaison avec le run principal dès la fin de la chaîne de jobs — cf.
`RESULTS_TESTS.md` pour le suivi détaillé et à jour de cette expérience.]*


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
filtrés par mots-clés), les emails réels n'étant chargés que pour une visualisation
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
dépendant d'une installation non faite par défaut), et aucune comparaison avec
"SAE Boost" n'a été entreprise (implémentation officielle non identifiée à date).

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
sépare légèrement MOINS bien les axes email réels (−2,2 points, la métrique la plus
proche des objectifs métier). Aucun écart n'est de l'ordre d'un problème majeur ; pas
de justification claire pour préférer l'un à l'autre sur ce projet à ce stade.

### Facteurs non contrôlés dans le corpus augmenté

Les variantes augmentées sont générées par le même modèle (Gemma-3-12B-it) qui sert
aussi de juge d'interprétation et d'extracteur d'activations. Un style de génération
propre au modèle (tournures récurrentes, longueur, structure) pourrait constituer un
facteur de confusion partagé entre "ce qui rend une variante reconnaissable comme
augmentée" et "ce que le SAE apprend à détecter" — non quantifié dans cette
investigation.

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
   un cas d'usage concret de détection d'urgence/d'intention sur mails réels~~
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


\newpage

---

# Conclusion générale

## Bilan par rapport aux objectifs initiaux

Le stage visait à rendre fonctionnelle et exploitable une plateforme d'analyse
interprétable de mails clients EDF fondée sur des Sparse Autoencoders. Au terme des
quatre phases décrites dans ce rapport, les deux pipelines (Gemma-3 + GemmaScope
étendu ; F2LLM + SAE dédié) fonctionnent de bout en bout sur le corpus réel, avec des
résultats quantifiés et reproductibles sur l'ensemble des capacités visées par
l'énoncé initial :

- **Détection d'urgence et d'intention** : séparabilité linéaire forte sur les axes
  synthétiques (93,5%/79,3% selon le pipeline) et gain net mesuré sur des labels
  faibles indépendants tirés de mails réels non augmentés (+27,0 points sur l'urgence,
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
