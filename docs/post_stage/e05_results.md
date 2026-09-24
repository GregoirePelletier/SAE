# E05 : stabilité des features entre entraînements (Gemma-3-1B, couche 13, K_EXTRA=5)

Question : si l'on réentraîne l'extension EXTRA, retrouve-t-on les mêmes features, les mêmes
groupes de features et les mêmes emails ? Le SAE CORE est gelé, donc identique d'un entraînement
à l'autre, et n'est pas comparé.

Par défaut, le décodeur de l'extension est initialisé avec les 1 024 premières directions d'une
ACP du résidu : cette initialisation ne dépend pas de `SEED`, qui ne change alors que l'ordre des
mini-lots. Pour tester réellement l'effet de l'initialisation, l'option `decoder_init="random"`
(variable `EXTRA_DECODER_INIT`) remplace ces directions par des directions aléatoires, le reste
de la calibration étant identique (`tests/test_saeboost_decoder_init.py`).

Plan d'expérience : cinq entraînements sur FIT avec les mêmes hyperparamètres, trois initialisés
par ACP (graines 42, 43, 44) et deux aléatoirement (45, 46). Deux analyses sur DEV :
`e05_stability.json` (graines 42, 43, 44) et `e05_stability_random_init.json` (42, 43, 45, 46).

## Méthode

- Features comparées : celles actives sur au moins 50 documents FIT et sur au plus la moitié
  d'entre eux (environ 670 sur 1 024 ; environ 30 % de l'extension est active presque partout et
  ne porte pas de thème).
- Une feature est « retrouvée » si sa plus proche voisine dans l'autre entraînement a un cosinus
  de décodeur d'au moins 0,7 et un profil d'activation sur DEV corrélé à au moins 0,5. Témoin :
  dictionnaire aléatoire de même covariance (aucune feature retrouvée).
- Groupes : graphe de voisinage entre features, découpé par l'algorithme de Louvain (18 à 25
  groupes de 3 features ou plus par entraînement). Chaque groupe est comparé à 100 groupes
  aléatoires de même taille, avec correction de Benjamini-Hochberg.
- Emails : on compare, d'un entraînement à l'autre, les 100 emails les plus activés par un groupe
  (score maximal ou moyenne des features actives).

## Résultats avec le découpage corrigé

Entraînements dans `results_post_stage_e01_fit_1b_layer13_k5_v2/` (graine 42) et
`results_post_stage_e05_{seed43,seed44,randinit45,randinit46}_v2/` (jobs 50662, 50937, 50668,
50938, 50669) ; analyses dans le premier dossier (jobs 50944, 50945).

Les cinq entraînements atteignent la même qualité : part de variance expliquée 0,905 à 0,906,
1,4 à 5,8 % de features EXTRA mortes (moins avec l'initialisation aléatoire), sonde E01 à 0,892.
Les initialisations sont bien indépendantes : cosinus moyen entre décodeurs de même indice de
0,17 entre les deux runs ACP, de 0,05 dès qu'un run aléatoire est impliqué.

| Paire d'entraînements | Features retrouvées | Pureté des groupes (réel / hasard) | Groupes significatifs (pureté) | Groupes dont les 100 emails de tête se retrouvent |
|---|---:|---|---:|---:|
| ACP → ACP | 81 % | 0,57 / 0,17 | 17 | 0 (max), 6,5 (moyenne) |
| ACP → aléatoire | 81 % | 0,53 / 0,16 | 18 | 0 (max), 1 (moyenne) |
| aléatoire → ACP | 79 % | 0,63 / 0,20 | 20,5 | 0 (max), 1,3 (moyenne) |
| aléatoire → aléatoire | 82 % | 0,58 / 0,18 | 21,5 | 0 (max), 2,5 (moyenne) |

Les valeurs sont des moyennes sur les deux sens de chaque paire. Sur les 674 features de la
graine 42, 544 sont retrouvées dans la graine 43, 494 dans les deux runs aléatoires, 478 dans les
graines 43 et 44, et 448 dans les trois autres runs à la fois.

## Conclusion

Les directions apprises et leurs regroupements se retrouvent d'un entraînement à l'autre bien
au-delà du hasard, et ce n'est pas un effet de l'initialisation commune : les runs aléatoires
sont aussi stables que les runs ACP. En revanche, un même groupe ne remonte pas les mêmes emails
en tête d'un entraînement à l'autre. Pour un usage de recherche d'emails, un groupe de features
n'est donc pas un filtre reproductible. La carte des features du dashboard (page « Stabilité
inter-graines (E05) ») sert à naviguer, pas de preuve.

## Première exécution (découpage erroné, historique)

Dossier `results_post_stage_e01_fit_1b_layer13_k5/` et entraînements sans suffixe (jobs 48585,
49089, 49090, 49254, 49255 ; analyses 49240 et 49305). Résultats très proches : 79 à 82 % de
features retrouvées selon le type de paire, 485 features sur 668 retrouvées dans les trois graines
ACP, groupes significatifs pour 16 à 19 groupes sur 20, aucun groupe dont les 100 emails de tête
se retrouvent avec le score maximal (4 à 5 avec la moyenne). Même conclusion.

## Limites

- Peu d'entraînements : les paires d'un même type partagent des runs, et la comparaison
  aléatoire → aléatoire repose sur une seule paire indépendante (45 et 46).
- Une seule résolution de Louvain ; aucun nom de groupe validé par un humain.

## Fichiers

- Scripts : `scripts/post_stage/e05_stability.py`, `src/post_stage/stability.py`.
- Recettes : `slurm/post_stage/10_*` à `15_*`.
