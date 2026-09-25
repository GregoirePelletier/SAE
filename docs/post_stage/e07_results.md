# E07 : regrouper les emails selon une question (Gemma-3-1B, couche 13, K_EXTRA=5)

Question : peut-on regrouper les mêmes emails de façons différentes selon la question posée
(type de problème, action attendue, ton), là où un regroupement classique donne toujours la même
partition ?

## Protocole

- 400 emails d'origine de CONFIRM (un email par origine, graine 42), 4 groupes par méthode.
- Trois axes, chacun décrit par une phrase : `type_probleme` (facturation, coupure, résiliation…),
  `action_attendue` (remboursement, explication, intervention…), `registre_urgence` (calme,
  urgent, en colère, neutre).
- CORE et FULL : on garde les 40 features dont le nom est le plus proche de la phrase de l'axe,
  puis on regroupe les emails selon ces seules features (clustering spectral, similarité de
  Jaccard). Dense (bge-m3, emails entiers) et TF-IDF : k-moyennes, sans lien avec l'axe.
- Contrôles : le modèle juge nomme chaque groupe, puis doit réattribuer des emails aux groupes à
  partir de ces noms (proportion de réattributions correctes) ; compacité des groupes dans l'espace
  bge-m3 comparée à des groupes tirés au hasard (score z, négatif = plus compact que le hasard).

## Résultats avec le découpage corrigé

Dossier `results_post_stage_e01_fit_1b_layer13_k5_v2_eval/` (job 50943).

Dense et TF-IDF produisent exactement la même partition pour les trois axes, puisqu'ils ne
tiennent pas compte de la question. Leurs groupes sont nets (réattribution de 0,95 à 1,0 pour
dense, de 0,35 à 1,0 pour TF-IDF) et suivent le type de problème (litiges de facture, mises en
service, résiliations, coupures). Ces deux partitions sont identiques à celles de la première
exécution : les emails d'origine de CONFIRM n'ont pas changé.

CORE et FULL produisent bien des partitions différentes selon l'axe, mais leur contenu ne suit
pas la question posée :

| Axe | Ce que décrivent les groupes FULL | Réattribution FULL (par groupe) |
|---|---|---|
| `type_probleme` | en partie le type de problème (coupures, résiliation et déménagement, litiges de facture), mêlé à la langue et au ton | 0,45 ; 0,96 ; 0,11 ; 0,80 |
| `action_attendue` | surtout la langue et le degré de formalité, pas l'action demandée | 0,72 ; 0,18 ; 0,18 ; 0,72 |
| `registre_urgence` | le type de demande (mise en service, facture, résiliation), pas le ton | 0,82 ; 0,78 ; 0,00 ; 0,13 |

Les groupes CORE ont des réattributions plus faibles (de 0,00 à 0,54) et, pour l'axe
`registre_urgence`, ne sont pas plus compacts que des groupes tirés au hasard (score z de −2,4
à +1,1). Les groupes FULL sont plus compacts que le hasard sur les trois axes (z de −3 à −26).

## Conclusion

Choisir des features selon la question change bien le regroupement, ce que les méthodes
classiques ne permettent pas. Mais avec le catalogue de features actuel, le regroupement obtenu
ne correspond pas à l'axe demandé, y compris pour le type de problème, qui fonctionnait dans la
première exécution. Le regroupement ciblé n'est donc pas démontré ; un catalogue de features plus
large ou nommé autrement serait nécessaire avant de le proposer comme fonctionnalité.

## Première exécution (découpage erroné, historique)

Dossier `results_post_stage_e01_fit_1b_layer13_k5/` (job 49087). Même constat mécanique (CORE et
FULL changent de partition selon l'axe). FULL suivait bien le type de problème (réattributions de
0,72 à 0,97), partiellement l'action attendue (0,12 à 0,96) et pas le ton (0,02 à 0,53).

## Limites

- Aucun audit humain des groupes (le plan prévoit un contrôle en aveugle de 20 à 30 paires
  d'emails par axe) : la réattribution par le modèle juge n'est qu'un contrôle secondaire.
- Nombre de groupes fixé à 4 pour toutes les méthodes et tous les axes.
- TF-IDF et dense sont ajustés sur l'échantillon de CONFIRM lui-même.
- Les résultats dépendent fortement des noms de features disponibles (76 CORE et 207 FULL
  interprétables) : un axe mal représenté dans le catalogue ne peut pas être isolé.

## Fichiers

- Script : `scripts/post_stage/e07_clustering.py` ; recette `09_*`.
- Résultat : `e07_clustering.json` dans le dossier de résultats.
