# E01 : comparaison des représentations (Gemma-3-1B, couche 13, K_EXTRA=5)

Question : l'extension EXTRA apporte-t-elle de l'information en plus du SAE préentraîné CORE ?
On la teste avec une sonde de classification (régression logistique) qui doit retrouver lequel
des 14 axes d'augmentation a produit un email (émotion, urgence, registre, orthographe ou
original). Ce n'est pas une classification d'intentions métier.

Protocole (plan §6.4) : SAE et TF-IDF entraînés sur FIT seul, évaluation sur DEV, sans validation
croisée sur l'ensemble évalué. Comparaisons appariées par test de McNemar et intervalle bootstrap
par email d'origine (521 groupes), correction de Benjamini-Hochberg.

## Résultats avec le découpage corrigé

Dossier `results_post_stage_e01_fit_1b_layer13_k5_v2/` (jobs 50662, 50663, 50664). FIT : 26 112
documents (2 084 emails d'origine), DEV : 6 471.

| Représentation | Exactitude DEV |
|---|---:|
| CORE (16 384 dimensions, GemmaScope-2 gelé) | 90,05 % |
| FULL (CORE et extension, 17 408 dimensions) | 90,11 % |
| TF-IDF (mots, 20 000 termes, ajusté sur FIT) | 91,67 % |
| Dense (bge-m3, pooling CLS) | 78,83 % |

- FULL − CORE : +0,06 point, IC [−0,29 ; +0,42], p = 0,81. L'extension n'améliore pas cette sonde.
- FULL − dense : +11,28 points, IC [+10,35 ; +12,25], p < 10⁻¹⁰⁰. Comme FULL et CORE sont
  équivalents, c'est surtout CORE qui fait mieux que bge-m3 ici. bge-m3 est conçu pour la
  similarité de contenu, pas pour le ton ou le registre : ce résultat ne se transpose pas à la
  recherche d'emails (E03).
- FULL − TF-IDF : −1,56 point, IC [−2,26 ; −0,87], p corrigé = 1,5·10⁻⁴. TF-IDF fait mieux.

Agrégation par email : remplacer le maximum par la moyenne des 3 plus fortes activations ne change
pas significativement le résultat (CORE −0,23 point, IC [−0,77 ; +0,36] ; FULL −0,06 point,
IC [−0,62 ; +0,52]).

Longueur des emails : la longueur seule donne 18,9 % d'exactitude (hasard ≈ 7 %). L'exactitude de
CORE et FULL baisse avec la longueur (de 94 % à 86 %), mais les deux restent à moins de 0,6 point
l'un de l'autre dans chaque quart de la distribution des longueurs : l'équivalence FULL ≈ CORE
n'est pas un effet de longueur.

| Longueur (caractères) | n | CORE | FULL | Longueur seule |
|---|---:|---:|---:|---:|
| 8 à 961 | 1 614 | 93,9 % | 93,8 % | 29,3 % |
| 962 à 1 238 | 1 621 | 90,1 % | 90,7 % | 13,8 % |
| 1 239 à 1 552 | 1 611 | 89,4 % | 89,7 % | 9,7 % |
| 1 553 à 4 394 | 1 625 | 86,8 % | 86,3 % | 22,8 % |

## Conclusion

Sur cette tâche, le signal utile vient du SAE préentraîné : l'extension n'apporte rien de
mesurable, et un TF-IDF fait légèrement mieux. Cela ne dit pas que l'extension est inutile en
général ; son intérêt se juge aussi sur la recherche d'emails (E03) et la comparaison de
populations (E04).

## Première exécution (découpage erroné, historique)

Dossier `results_post_stage_e01_fit_1b_layer13_k5/` : FIT 25 970 documents, DEV 6 518. Exactitude
CORE 88,68 %, FULL 88,80 %, TF-IDF 87,94 %, dense 77,05 %. FULL − CORE : +0,12 point, IC [−0,08 ;
+0,32] ; FULL − TF-IDF : +0,86 point, non significatif après correction (p = 0,056). Mêmes
conclusions pour l'agrégation et la longueur. Environ 39 % des variantes de FIT n'appartenaient
pas à FIT (voir `docs/RESULTS_STATUS.md`) : ces chiffres ne sont plus la référence.

## Limites

- Le réservoir de 8 millions de tokens a été entièrement rempli : FIT contient au moins ce nombre
  de tokens, la valeur exacte n'a pas été mesurée.
- Évaluation sur DEV, une seule graine d'entraînement (la variabilité entre graines est étudiée
  dans E05).

## Fichiers

- Checkpoints : `p1_extended_sae.pt`, `p1_frozen_core_d1024_k5.pt` dans le dossier de résultats.
- Résultats : `e01_representation_comparison.json`, `e01_pooling_and_length_analysis.json`.
- Scripts : `scripts/post_stage/e01_compare_representations.py`,
  `scripts/post_stage/e01_pooling_and_length_analysis.py`.
