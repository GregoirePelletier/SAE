# État de l'art

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
  du projet (`SAEBoostResidualSAE`, `PhraseLevelSAE`).

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

Ce critère (couverture de labels) diffère de celui utilisé par Formal et al.
(SPLARE, 2026, `pdf/Naver.pdf`), qui balaient la largeur du SAE sur une tâche
de retrieval en aval et observent une relation **log-linéaire** entre largeur
et performance (16k à 1M de features) — un arbitrage direct entre coût et
utilité mesurée, plutôt qu'un critère de couverture de labels externes. Les
deux critères ne sont pas nécessairement alignés : une largeur qui maximise
la couverture Neuronpedia ne maximise pas forcément l'utilité en aval du
dictionnaire — question non tranchée par ce projet, cf. chapitre 4.

## Neuronpedia et l'auto-interprétation des features

[Neuronpedia](https://www.neuronpedia.org) héberge des explications en langage naturel
générées automatiquement pour les features de nombreux SAE publics, dont GemmaScope.
Ces explications servent de "vérité de référence" externe pour les features du SAE
préentraîné ("core") dans ce projet. Elles n'existent en revanche pour aucune feature
d'un SAE entraîné spécifiquement pour ce projet (l'extension `SAEBoostResidualSAE`, le
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
`SAEBoostResidualSAE` (`src/sae/frozen_core.py`) reproduit exactement leur méthode (SAE
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
