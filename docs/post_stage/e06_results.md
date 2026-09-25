# E06 : associations entre propriétés (Gemma-3-1B, couche 13, K_EXTRA=5)

Question : les features permettent-elles de repérer des propriétés qui apparaissent souvent
ensemble dans les emails, et ces associations se confirment-elles sur d'autres emails ?

## Protocole

- Découverte sur FIT et DEV : pour chaque paire de features nommées en E02, cooccurrence dans
  les emails d'origine, mesurée par l'information mutuelle ponctuelle normalisée (NPMI, de −1 à
  1). Les features actives dans moins de 1 % ou plus de 50 % des emails sont écartées (une feature
  présente partout est associée à tout), ainsi que les paires de même nom ou quasi synonymes.
- Les 8 paires au NPMI le plus fort (en valeur absolue) sont figées avant toute lecture de CONFIRM.
- Vérification sur 350 emails d'origine de CONFIRM (graine 42) : le modèle juge dit, pour chaque
  email, si chacune des propriétés est présente ; la cooccurrence est recalculée sur ces jugements,
  et non sur les activations. Une association est retenue si l'intervalle bootstrap du NPMI exclut
  0 et si le test de Fisher reste significatif après correction de Benjamini-Hochberg.

## Résultats avec le découpage corrigé

Dossier `results_post_stage_e01_fit_1b_layer13_k5_v2_eval/` (job 50942), 1 951 paires candidates.
Les paires diffèrent de la première exécution, puisque le checkpoint et le catalogue de features
ont changé.

| Paire | Nombre d'emails (A / B / les deux) | NPMI sur CONFIRM | Retenue |
|---|---:|---|---|
| Contestation de facture × Augmentation de facture | 102 / 91 / 91 | 0,92 [0,87 ; 0,96] | oui |
| Raccordement énergie × Fournisseur précédent | 66 / 33 / 32 | 0,68 [0,60 ; 0,77] | oui |
| Fournisseur précédent × « Raccordement à » | 33 / 62 / 30 | 0,67 [0,57 ; 0,75] | oui |
| Travail à domicile × Heure de coupure | 61 / 31 / 19 | 0,43 [0,29 ; 0,55] | oui |
| Consommation stable × Contestation de facture | 9 / 102 / 9 | 0,34 [0,27 ; 0,40] | oui |
| Raccordement énergie × « Électricité et/ou » | 66 / 34 / 0 | non défini | non |
| Date de l'incident × Date de fin | 25 / 8 / 0 | non défini | non |
| Champs nom et prénom × Coordonnées client | 0 / 219 / 0 | — | support insuffisant |

5 paires sur 8 sont retenues. Toutes n'apportent pas la même information :

- *Contestation × augmentation de facture* et les deux paires autour du raccordement sont en
  grande partie attendues (une hausse de facture motive une contestation ; un raccordement se fait
  souvent en quittant un autre fournisseur). « Raccordement à » est très probablement la même
  notion que « Raccordement énergie ».
- *Travail à domicile × heure de coupure* et *consommation stable × contestation* sont les pistes
  les plus intéressantes : les clients qui travaillent chez eux précisent l'heure de la coupure,
  et une partie des contestations s'appuie sur une consommation jugée stable. Ce sont des
  observations sur un corpus synthétique, pas des constats sur la clientèle réelle.
- *Raccordement énergie × « Électricité et/ou »* n'apparaissent jamais ensemble alors que
  chacune est fréquente (test de Fisher significatif, p = 0,001) : c'est une exclusion, que la
  règle fondée sur le NPMI ne retient pas.

## Première exécution (découpage erroné, historique)

Dossier `results_post_stage_e01_fit_1b_layer13_k5/` (job 49077), 1 246 paires candidates. 4 paires
sur 8 retenues : coordonnées bancaires × IBAN (NPMI 0,66), identification SIREN × numéro SIREN
(0,98, un même concept en double), formule de politesse × facture de clôture (0,14) et
dysfonctionnement électrique × réfrigérateur (0,35), la seule association alors jugée
intéressante ; 3 non retenues, 1 sans support suffisant.

## Limites

- Aucun audit humain des associations (le plan en prévoit un sur 40 à 80 emails).
- Aucune paire n'associe deux features EXTRA : le classement par NPMI ne force pas l'équilibre
  entre CORE et EXTRA demandé par le plan.
- Le filtre des quasi-synonymes compare les noms mot à mot ; il laisse passer des doublons
  (« Raccordement énergie » et « Raccordement à »). Certains noms de features sont tronqués.
- Pas de comparaison avec des cooccurrences de mots calculées sur les mêmes emails.
- Un seul email par email d'origine est jugé (pas ses variantes).

## Fichiers

- Script : `scripts/post_stage/e06_correlations.py` ; recette `08_*`.
- Résultat : `e06_correlations.json` dans le dossier de résultats.
