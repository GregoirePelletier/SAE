# E05 — Stabilité inter-graines des features EXTRA (1B/layer13/K5)

Trois extensions (`D_EXTRA=1024`, `K_EXTRA=5`) entraînées sur les mêmes
données FIT, mêmes hyperparamètres, graines 42 (référence, job 48585), 43
(job 49089) et 44 (job 49090) ; le cœur GemmaScope-2 gelé, identique par
construction, n'est jamais comparé. Les deux runs supplémentaires
réutilisent le réservoir de tokens du cache d'extraction partagé (jamais
ré-extrait). Analyse : job CPU 49240 (4 min), `scripts/post_stage/
e05_stability.py`, DEV = 6518 documents.

## Limite structurelle à lire avant tout chiffre

**Les trois entraînements partagent la même initialisation.**
`SAEBoostResidualSAE` initialise le décodeur EXTRA par les 1024 premières
directions PCA du résidu, calculées sur le réservoir partagé (« 1024
directions PCA injectées » dans les logs des trois runs). Seul l'ordre des
mini-lots change avec `SEED`. **Ce qui est mesuré : la robustesse à l'ordre
d'entraînement depuis une init PCA commune — pas l'indépendance à une
initialisation aléatoire**, qui n'est pas testée (aucun bras « init
aléatoire entraînée » dans le pipeline : le seul autre chemin,
`SANITY_CHECK_FROZEN_DECODER`, fige le décodeur). Les résultats ci-dessous
sont donc probablement plus favorables que ceux d'un vrai test d'init
aléatoire. Les décodeurs sont pourtant bien distincts (cosinus moyen à
indice égal 0,16–0,17, différence max de poids ≈1,1–1,2) : l'entraînement
déplace fortement les features hors de leur emplacement PCA, différemment
d'un run à l'autre — comparer les features par indice conclurait à tort à
une instabilité totale.

## Population testée

| Run | features EXTRA | supportées | support insuffisant (<50 actifs FIT) | quasi-universelles (fréq FIT>0,5) |
|---|---:|---:|---:|---:|
| graine 42 | 1024 | 668 | 47 | 309 |
| graine 43 | 1024 | 654 | 48 | 322 |
| graine 44 | 1024 | 657 | 55 | 312 |

**Près d'un tiers de l'extension (~30 %) est quasi-universel** (actif sur >50 %
des documents FIT) — exclu de l'appariement et des groupes (mêmes features
"sink" que celles qui avaient faussé E06 à la découverte), mais à garder
visible : ces directions ne portent pas de thème discriminant.

## Appariement individuel (6 ordres de paires, 654–668 features sources)

Plus proche voisin par cosinus signé du décodeur, plusieurs-à-un autorisé,
candidats = features supportées de l'autre run.

| Mesure | Réel | Témoin |
|---|---:|---:|
| Meilleur cosinus, médiane | 0,94 | 0,32 (dictionnaire nul gaussien, covariance anisotrope réelle) |
| Fraction avec cos ≥ 0,7 | 0,80–0,83 | 0,000 |
| Fraction avec cos ≥ 0,5 | 0,90–0,92 | 0,001–0,003 |
| Corrélation de profils DEV des paires appariées, médiane | 0,95–0,96 | 0,00–0,01 (appariements mélangés) |
| Cos ≥ 0,7 **et** corrélation de profils ≥ 0,5 | 0,79–0,83 | — |
| Voisin mutuel (A→B→A) | 0,80–0,83 | — |
| Jaccard@20 des documents les plus activants, moyenne | 0,32–0,34 | — |
| Rapport de fréquence B/A, médiane | ≈1,01 | — |

**485 des 668 features de la graine 42 (72,6 %) sont retrouvées dans les
deux autres graines** (critère : cos ≥ 0,7 et profil ≥ 0,5 dans les deux) —
formulation à conserver telle quelle : « retrouvée dans les deux répétitions
disponibles », **pas** « reproductible à 100 % » ni un taux de
reproductibilité générale (3 graines, init commune). Les ~20 % restants
(cos < 0,7) sont des features réellement non appariées, listées dans
`feature_matches_reference_side`.

## Groupes et sous-espaces

Par dictionnaire : graphe de voisinages mutuels (k=10, cosinus du décodeur),
arêtes conservées si la corrélation de profils DEV dépasse le q99 d'un
témoin de permutation colonne par colonne (τ=0,030), Louvain à une seule
résolution, isolats conservés : **20 groupes (≥3 features) par dictionnaire**,
26–31 isolats, 613–629 features en groupes, plus grand groupe 66–75 (aucun
groupe surdimensionné >150). Chaque groupe réel est comparé à 100 groupes
aléatoires de même taille et de même strate de fréquence, passés par la
**même** procédure de partenaire (meilleur appariement contre meilleur
appariement) ; FDR-BH sur les 20 groupes de chaque paire. Moyennes sur les
groupes, réel / nul, puis nombre de groupes /20 significatifs après FDR :

| Paire (A→B) | Pureté du partenaire | Recouvrement de sous-espace | Spearman score (max) | Spearman score (moy. actifs) | Jaccard@100 (max) |
|---|---|---|---|---|---|
| 42→43 | 0,55/0,17 — 17 | 0,35/0,18 — 17 | 0,34/0,15 — 11 | 0,39/0,22 — 8 | 0,05/0,02 — 0 |
| 42→44 | 0,59/0,18 — 19 | 0,38/0,18 — 17 | 0,46/0,19 — 12 | 0,49/0,19 — 12 | 0,06/0,04 — 0 |
| 43→42 | 0,60/0,19 — 18 | 0,35/0,18 — 17 | 0,37/0,18 — 9 | 0,40/0,21 — 12 | 0,05/0,02 — 0 |
| 43→44 | 0,57/0,19 — 18 | 0,34/0,18 — 16 | 0,42/0,20 — 12 | 0,46/0,18 — 13 | 0,12/0,04 — 0 |
| 44→42 | 0,59/0,19 — 18 | 0,37/0,18 — 17 | 0,44/0,18 — 13 | 0,47/0,21 — 14 | 0,06/0,02 — 0 |
| 44→43 | 0,57/0,19 — 17 | 0,33/0,18 — 15 | 0,35/0,14 — 10 | 0,40/0,20 — 9 | 0,10/0,05 — 0 |

(Les 6 paires ne sont pas 6 confirmations indépendantes : elles partagent
les mêmes trois entraînements. Le Jaccard@20 et sa version moyenne des
actifs ne dépassent le nul pour aucun groupe dans aucune paire.)

Lecture :

1. **Géométrie de groupe stable, au-delà du hasard** : les membres d'un
   groupe se retrouvent majoritairement dans un même groupe partenaire
   (pureté ≈0,57 contre 0,19 ; 15–19 groupes sur 20 significatifs), les
   partenaires sont de taille comparable (rapport médian 1,3, jamais <3
   membres) et leurs sous-espaces se recouvrent deux fois plus que ceux de
   groupes aléatoires appariés (≈0,35 contre 0,18). **Le rang du recouvrement
   est le plafond fixé (16) pour la plupart des groupes** (médiane 16, min
   3) : sur rang 16 dans un espace de dimension 1152, le recouvrement isotrope
   de référence serait ≈0,01 — le témoin anisotrope (0,18) est le
   comparateur pertinent, pas ce chiffre.
2. **La stabilité s'arrête aux groupes, elle ne descend pas jusqu'aux
   emails du haut de classement.** La corrélation de rang des scores de groupe
   sur les 6518 documents DEV est nettement au-dessus du nul (≈0,34–0,49 contre
   ≈0,14–0,22) pour environ la moitié des groupes (8–14 sur 20 selon le
   score), mais le recouvrement des 100 premiers documents n'est significatif
   pour **aucun** groupe. Les features individuelles, elles, gardent des
   documents-tête assez proches (Jaccard@20 moyen 0,32) : c'est
   l'agrégation en groupes (max ou moyenne de dizaines de features) qui écrase
   la finesse du haut de classement — le score « max » sature pour les grands
   groupes (plan §10.7 le prévoyait), le score « moyenne des actifs »
   n'améliore pas le haut de classement.

## Conclusion (modèle du plan §10)

Correspond à : **« géométrie stable mais retrieval de thèmes variable :
sous-espace partagé possible, utilité documentaire non démontrée »**.
Les features et leurs groupes se retrouvent d'un entraînement à l'autre
bien au-delà du hasard (à init PCA commune), mais un groupe ne redonne pas
les mêmes emails en tête de classement d'une graine à l'autre — ce qui
importe pour un usage de recherche/filtrage documentaire. La carte
descriptive (`map_positions_reference`, PCA 2D des directions de la graine 42,
colorée par groupe, dans la page dashboard) est livrée ; elle sert à naviguer,
pas de preuve statistique.

## Ce qui n'est PAS démontré

- Indépendance à une initialisation aléatoire (init PCA commune, cf. plus
  haut). Un bras « init aléatoire » entraîné nécessiterait de modifier le
  pipeline (constructeur sans PCA + calibration d'échelle séparée) — non fait.
- Reproductibilité au-delà de 3 graines ; les p-values sont des p-values de
  randomisation contre le nul spécifié, pas une généralisation à tous les
  entraînements possibles.
- Utilité pour des thèmes nommés (Jaccard@20/P@10 liés aux requêtes E03 non
  calculés) ; groupes non nommés, pas de validation humaine des groupes.
- Profils = activations **document-level** (max-pool), pas les 50–100k tokens
  partagés du plan ; permutation témoin non stratifiée par longueur/parent.
- Seuils cos ≥ 0,7 et profil ≥ 0,5 : repères déclarés à l'avance, non
  optimisés ; 20 % de features non appariées restent visibles.

## Fichiers

- `results_post_stage_e01_fit_1b_layer13_k5/e05_stability.json` (appariements
  côté référence, groupes, comparaisons par groupe/paire, positions 2D)
- `scripts/post_stage/e05_stability.py`, `src/post_stage/stability.py`
  (10 tests, `tests/post_stage/test_stability.py`)
- Jobs : 48585 (référence), 49089/49090 (graines 43/44), 49237/49240
  (analyse ; 49240 = rerun avec scores de groupe supplémentaires)
