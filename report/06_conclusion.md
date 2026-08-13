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

Le diagnostic du chapitre 3 établit que le taux d'auto-interprétation des
features, initialement très faible (20%), n'était pas limité par le volume
d'entraînement mais par le domaine du corpus d'entraînement de l'extension —
un exemple concret de la valeur d'une démarche d'ablation contrôlée plutôt
que d'une intuition non testée ("il faut probablement plus de données"). Le
chapitre 3 mesure par ailleurs un second effet, de nature différente : une
dose-réponse de l'échelle du modèle (§18, p<10⁻⁹). Les deux résultats
répondent à des questions distinctes — l'un explique pourquoi le pipeline
fonctionne sur ce corpus, l'autre identifie quel levier a le plus fait
varier le taux d'interprétabilité parmi ceux testés.

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
