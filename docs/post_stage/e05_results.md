# E05 — Stabilité inter-graines des features EXTRA (1B/layer13/K5)

> **Statut : résultats antérieurs au correctif de filiation des emails parents. La séparation FIT/DEV/CONFIRM n'était pas effective pour les variantes. Ces résultats et checkpoints restent consultables comme historique, mais ne constituent pas une validation hors apprentissage. Un rejeu avec le manifeste corrigé est nécessaire. Les évaluations humaines n'ont pas été réalisées.**
(Détail et impact : `docs/RESULTS_STATUS.md`, section Corpus.)

Cinq extensions (`D_EXTRA=1024`, `K_EXTRA=5`) entraînées sur les mêmes
données FIT, mêmes hyperparamètres, cœur GemmaScope-2 gelé jamais comparé
(identique par construction) : trois avec l'init par défaut « PCA » (graines
42 job 48585, 43 job 49089, 44 job 49090) et deux avec une **init aléatoire**
(graines 45 job 49254, 46 job 49255). Tous réutilisent le réservoir de tokens
du cache d'extraction partagé. Deux analyses CPU, DEV = 6518 documents :
`e05_stability.json` (graines 42/43/44, job 49240) et
`e05_stability_random_init.json` (42, 43, 45, 46, job 49305).

## Résultat en une ligne

**La stabilité mesurée ne dépend pas de l'initialisation** : entre deux runs à
initialisation aléatoire indépendante, ~80 % des features supportées se
retrouvent (contre ~82 % entre deux runs PCA) et les groupes se retrouvent
aussi bien — mais les emails de tête d'un groupe ne se retrouvent pas d'une
graine à l'autre.

## Pourquoi un bras « init aléatoire »

`SAEBoostResidualSAE` initialise par défaut le décodeur EXTRA avec les 1024
premières directions PCA du résidu sur le réservoir **partagé** : cette init
est déterministe et identique pour toute `SEED` (test :
`tests/test_saeboost_decoder_init.py`), donc dans `e05_stability.json` les
trois graines ne diffèrent que par l'ordre des mini-lots. Pour tester
l'indépendance à l'initialisation, `SAEBoostResidualSAE` accepte désormais
`decoder_init="random"` (variable `EXTRA_DECODER_INIT`, enregistrée dans la
config du checkpoint) : les directions restent celles du parent (gaussien
isotrope normalisé sous `SEED`) et **tout le reste est calibré à l'identique**
(`input_scale`, `encoder_input_scale`, biais de l'encodeur). Les logs
confirment la branche (« mode aléatoire : directions parent conservées »).
Le cosinus moyen à indice égal entre décodeurs finaux vaut 0,166 entre les deux
runs PCA (init commune, empreinte résiduelle) contre 0,043–0,053 pour toute
paire impliquant une init aléatoire — les inits sont bien indépendantes.

## Les runs aléatoires sont-ils aussi bien entraînés ?

| Run | init | FVE étendue | features EXTRA mortes | rho_sae | sonde 14 classes (axes d'augmentation, cf. E01) |
|---|---|---:|---:|---:|---:|
| 42 | PCA | 0,9051 | 4,59 % | 0,950 | 0,877 |
| 43 | PCA | 0,9053 | 4,69 % | 0,953 | 0,878 |
| 45 | aléatoire | 0,9062 | 0,88 % | 0,953 | 0,877 |
| 46 | aléatoire | 0,9059 | 2,25 % | 0,952 | 0,879 |

Aucun bras n'est sous-entraîné : l'init aléatoire atteint la même
reconstruction avec moins de features mortes (4,6 % → 0,9–2,2 %, donc plus de
features supportées : 696 et 674 contre 668 et 654) et la même précision de
sonde. La comparaison de stabilité n'oppose donc pas un bras bien entraîné à
un bras dégradé.

## Population testée (analyse à 4 runs)

Supportées (≥50 documents FIT actifs, fréquence FIT ≤0,5) : 668 / 654 / 696 /
674 sur 1024. Environ 30 % de l'extension est quasi-universelle (309–327 par
run, fréquence FIT >0,5) : exclue de l'appariement et des groupes (mêmes
features « sink » qui avaient faussé E06 à la découverte), mais à garder
visible — ces directions ne portent pas de thème discriminant.

## Appariement individuel, par type de paire

Plus proche voisin par cosinus signé du décodeur, plusieurs-à-un autorisé,
controlé par la corrélation de profils DEV (activations max-poolées par
document). Moyennes sur les ordres de paires de chaque type ; nul = dictionnaire
gaussien à covariance anisotrope réelle (cos ≥0,7 : 0,000 partout, cos médian
0,32) et appariements mélangés (corrélation de profils médiane ≈0,01).

| Type de paire (A→B) | ordres | cos ≥0,7 | cos ≥0,7 **et** profil ≥0,5 | voisin mutuel | cos médian | corr. profils médiane | Jaccard@20 moyen |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCA→PCA | 2 | 0,825 | 0,824 | 0,825 | 0,94 | 0,955 | 0,33 |
| PCA→aléatoire | 4 | 0,817 | 0,813 | 0,814 | 0,92 | 0,952 | 0,32 |
| aléatoire→PCA | 4 | 0,788 | 0,785 | 0,785 | 0,92 | 0,950 | 0,31 |
| **aléatoire→aléatoire** | 2 | 0,808 | 0,805 | 0,808 | 0,94 | 0,963 | 0,34 |

Les features de la graine 42 (PCA, 668 supportées) retrouvées : **545 (81,6 %)**
dans la graine 43 (PCA), **478 (71,6 %)** dans les deux runs aléatoires,
**438 (65,6 %)** dans les trois autres runs à la fois. Formulation à
conserver : « retrouvée dans les répétitions disponibles », pas un taux de
reproductibilité générale. L'asymétrie légère aléatoire→PCA (0,785) contre
PCA→aléatoire (0,813) est cohérente avec des runs aléatoires qui gardent
davantage de features vivantes (dont certaines sans jumelle PCA) ; à deux runs
par bras, pas plus qu'un indice.

## Groupes et sous-espaces, par type de paire

Par dictionnaire : voisinages mutuels k=10 (cosinus du décodeur), arêtes
conservées si la corrélation de profils DEV dépasse le q99 d'un témoin de
permutation colonne par colonne, Louvain à une seule résolution, isolats
conservés — 20 groupes (≥3 features) par dictionnaire sur l'analyse à 3 runs
PCA. Chaque groupe est comparé à 100 groupes aléatoires de même taille et de
même strate de fréquence, passés par la **même** procédure de partenaire ;
FDR-BH par paire. Cellules : réel / nul — nombre moyen de groupes /20
significatifs après FDR.

| Type | Pureté du partenaire | Recouvrement de sous-espace | Spearman, score max | Spearman, moy. actifs | Jaccard@100, score max |
|---|---|---|---|---|---|
| PCA→PCA | 0,58/0,18 — 18,0 | 0,37/0,18 — 16,5 | 0,38/0,17 — 9,5 | 0,41/0,21 — 10,5 | 0,06/0,02 — 0 |
| PCA→aléatoire | 0,59/0,19 — 18,5 | 0,38/0,19 — 17,5 | 0,42/0,19 — 13,2 | 0,45/0,19 — 12,5 | 0,06/0,03 — 0 |
| aléatoire→PCA | 0,59/0,18 — 17,8 | 0,38/0,18 — 17,5 | 0,38/0,15 — 12,0 | 0,42/0,20 — 12,5 | 0,05/0,02 — 0 |
| **aléatoire→aléatoire** | 0,61/0,17 — 18,5 | 0,42/0,19 — 18,5 | 0,43/0,17 — 11,5 | 0,48/0,20 — 14,0 | 0,04/0,02 — 0 |

Lecture :

1. **Géométrie de groupe stable, au-delà du hasard, quelle que soit l'init** :
   pureté ≈0,6 contre ≈0,18 ; recouvrement ≈0,4 contre ≈0,18 ; 16–19 groupes sur
   20 significatifs. Le rang du recouvrement est le plafond fixé (16) pour la
   plupart des groupes (médiane 16, min 3, dans un espace de dimension 1152) : le
   témoin anisotrope (≈0,18) est le comparateur pertinent, pas la référence
   isotrope r/d≈0,01.
2. **La stabilité s'arrête aux groupes, elle ne descend pas jusqu'aux emails
   du haut de classement.** La corrélation de rang des scores de groupe sur les
   6518 documents DEV dépasse nettement le nul (≈0,38–0,48 contre ≈0,15–0,21)
   pour la moitié des groupes environ, mais le recouvrement des 100 premiers
   documents n'est significatif pour **aucun** groupe avec le score « max »
   (en moyenne 4–5 groupes /20 pour le score « moyenne des actifs », comparable
   pour PCA→PCA et pour aléatoire→aléatoire, donc pas un effet d'init). Les
   features individuelles gardent des documents-tête proches (Jaccard@20 moyen
   0,31–0,34) : c'est l'agrégation en groupes qui écrase la finesse du haut de
   classement (le score « max » sature pour les grands groupes, plan §10.7).
3. **Les 6 (ou 12) ordres de paires ne sont pas des confirmations
   indépendantes** : ils partagent quatre entraînements. Le type
   aléatoire→aléatoire repose sur **une seule paire indépendante** (45↔46, les
   deux sens).

## Conclusion (modèle du plan §10)

**« Géométrie stable mais retrieval de thèmes variable : sous-espace partagé
possible, utilité documentaire non démontrée »**, et désormais **sans
réserve sur l'initialisation** : les features et leurs groupes se retrouvent
d'un entraînement à l'autre bien au-delà du hasard, y compris depuis des
initialisations aléatoires indépendantes, mais un groupe ne redonne pas les
mêmes emails en tête de classement d'une graine à l'autre — ce qui importe pour
un usage de recherche/filtrage documentaire. La carte descriptive
(`map_positions_reference`, PCA 2D des directions de la graine 42, colorée par
groupe) est dans la page dashboard ; elle sert à naviguer, pas de preuve.

## Ce qui n'est PAS démontré

- Reproductibilité au-delà de 2–3 entraînements par bras ; les p-values sont des
  p-values de randomisation contre le nul spécifié, pas une généralisation à
  tous les entraînements possibles.
- Utilité pour des thèmes nommés (Jaccard@20/P@10 liés aux requêtes E03 non
  calculés) ; groupes non nommés, pas de validation humaine des groupes.
- Profils = activations **document-level** (max-pool), pas les 50–100k tokens
  partagés du plan ; permutation témoin non stratifiée par longueur/parent.
- Seuils cos ≥0,7 et profil ≥0,5 : repères déclarés à l'avance, non optimisés ;
  ~20 % de features non appariées restent visibles (`feature_matches_reference_side`).
- Stabilité ≠ interprétabilité : une feature stable n'est pas pour autant
  interprétable ; aucun lien fait ici avec le registre E02.

## Fichiers

- `results_post_stage_e01_fit_1b_layer13_k5/e05_stability.json` (3 runs PCA) et
  `e05_stability_random_init.json` (2 PCA + 2 aléatoires) : appariements côté
  référence, groupes, comparaisons par groupe/paire, résumé par type de paire,
  qualité d'entraînement, positions 2D.
- `scripts/post_stage/e05_stability.py`, `src/post_stage/stability.py` (10
  tests), `src/sae/frozen_core.py::SAEBoostResidualSAE(decoder_init=...)` (4
  tests, `tests/test_saeboost_decoder_init.py`).
- Jobs : 48585/49089/49090 (PCA 42/43/44), 49254/49255 (aléatoire 45/46),
  49240 (analyse à 3 runs), 49305 (analyse à 4 runs).
