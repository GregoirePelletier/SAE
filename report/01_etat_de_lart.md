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
