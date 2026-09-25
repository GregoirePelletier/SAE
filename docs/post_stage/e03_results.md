# E03 : retrouver des emails selon une propriété (Gemma-3-1B, couche 13, K_EXTRA=5)

Question : les features du SAE permettent-elles de retrouver des emails qui ont une propriété
donnée, et font-elles mieux qu'un moteur de recherche classique ?

## Protocole

- Collection : les emails de CONFIRM (10 840 documents dans le rejeu).
- 12 requêtes : 6 propriétés (relances répétées, menace de résiliation, incident collectif,
  explication d'un montant, coupures répétées, urgence implicite), chacune formulée une fois avec
  les mots attendus (« lexicale ») et une fois autrement (« paraphrase »).
- 5 méthodes :
  - CORE et FULL : on choisit les features dont le nom est le plus proche de la requête, puis on
    classe les emails selon l'activation de ces features. CORE ne dispose que des features CORE
    jugées interprétables en E02 (76), FULL de toutes (207).
  - Dense : similarité avec la requête dans l'espace bge-m3.
  - TF-IDF et BM25 : recherche lexicale, indexée directement sur CONFIRM.
- Pertinence : les 10 premiers résultats de chaque méthode sont jugés par Qwen3.8-27B (0, 1 ou 2).
  La mesure est P@10 : la part des 10 premiers résultats jugée pleinement pertinente (score 2).
  Tous les top-10 étant jugés, cette mesure ne demande pas de juger le reste de la collection.
- Comparaisons : moyenne sur les 6 propriétés, intervalle de confiance bootstrap sur ces 6
  familles de requêtes.

## Résultats avec le découpage corrigé

Dossier `results_post_stage_e01_fit_1b_layer13_k5_v2_eval/` (job 50940), checkpoint FIT du rejeu
et registre E02 du rejeu.

| Propriété | Formulation | CORE | FULL | Dense | TF-IDF | BM25 |
|---|---|---:|---:|---:|---:|---:|
| relances répétées | lexicale | 0,5 | 1,0 | 1,0 | 1,0 | 1,0 |
| relances répétées | paraphrase | 0,2 | 1,0 | 1,0 | 1,0 | 0,8 |
| menace de résiliation | lexicale | 0,1 | 0,7 | 0,4 | 0,6 | 0,5 |
| menace de résiliation | paraphrase | 0,3 | 0,6 | 0,1 | 0,1 | 0,0 |
| incident collectif | lexicale | 0,1 | 0,0 | 0,0 | 0,2 | 0,2 |
| incident collectif | paraphrase | 0,0 | 0,0 | 0,6 | 0,5 | 0,2 |
| explication d'un montant | lexicale | 0,4 | 0,5 | 0,7 | 0,2 | 0,0 |
| explication d'un montant | paraphrase | 0,2 | 0,6 | 0,7 | 1,0 | 1,0 |
| coupures répétées | lexicale | 0,0 | 0,2 | 0,5 | 0,2 | 0,2 |
| coupures répétées | paraphrase | 0,0 | 0,1 | 0,5 | 0,2 | 0,2 |
| urgence implicite | lexicale | 1,0 | 0,8 | 1,0 | 0,9 | 1,0 |
| urgence implicite | paraphrase | 1,0 | 0,4 | 0,7 | 0,7 | 1,0 |
| **Moyenne** | | **31,7 %** | **49,2 %** | **60,0 %** | **55,0 %** | **50,8 %** |

Écarts de FULL avec les autres méthodes (points de P@10, intervalle à 95 %) :

- FULL − CORE : +17,5, IC [−9,2 ; +45,0] ;
- FULL − dense : −10,8, IC [−28,3 ; +11,7] ;
- FULL − TF-IDF : −5,8, IC [−20,0 ; +10,9] ;
- FULL − BM25 : −1,7, IC [−20,0 ; +19,2].

Aucun de ces écarts n'est significatif avec 6 familles de requêtes. FULL fait mieux que CORE en
moyenne, surtout sur les relances et la menace de résiliation, mais nettement moins bien sur
l'urgence implicite ; la méthode dense reste la meilleure en moyenne.

L'incident collectif est presque introuvable avec les features (0 ou 0,1), alors que les méthodes
dense et lexicales en trouvent en paraphrase (0,2 à 0,6) : c'est une limite du catalogue de
features, pas une absence de la propriété dans le corpus.

## Conclusion

Les features SAE permettent de retrouver des emails pour certaines propriétés, mais ne battent pas
un moteur dense ou lexical en moyenne. Le seul avantage net observé dans la première exécution
(FULL au-dessus de CORE) n'est plus significatif avec le découpage corrigé. Si l'outil est
poursuivi, l'usage raisonnable est un moteur dense ou lexical pour classer les emails, et les
features pour décrire ce qui les distingue.

## Première exécution (découpage erroné, historique)

Dossier `results_post_stage_e01_fit_1b_layer13_k5/` (10 865 documents). Moyennes de P@10 :
CORE 25,0 %, FULL 50,0 %, dense 65,0 %, TF-IDF 58,3 %, BM25 65,0 %. Deux écarts étaient
significatifs : FULL − CORE (+25,0 points, IC [+5,8 ; +44,2]) et FULL − dense (−15,0 points,
IC [−24,2 ; −6,7]). Une partie des variantes de CONFIRM avait servi à l'entraînement : ces chiffres
ne sont pas comparables au rejeu.

## Limites

- Jugements faits par le modèle juge, sans vérification humaine.
- 6 familles de requêtes seulement : les intervalles sont larges.
- Le top-10 n'est pas dédupliqué par email d'origine : plusieurs variantes d'un même email peuvent
  compter comme plusieurs résultats.
- TF-IDF, BM25 et la normalisation des scores SAE sont ajustés sur CONFIRM lui-même (indexation de
  la collection recherchée), contrairement au protocole d'E01.
- CORE dispose de moins de features nommées que FULL (76 contre 207) : l'écart FULL − CORE mêle
  l'apport des directions EXTRA et celui d'un catalogue plus grand.

## Fichiers

- Script : `scripts/post_stage/e03_property_retrieval.py` ; recettes `06_*` et `06b_*`.
- Résultat : `e03_property_retrieval.json` (contient des extraits d'emails, hors Git).
