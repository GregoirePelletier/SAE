# E03 — Retrieval par propriété sur CONFIRM (1B/layer13/K5)

> **Statut : résultats antérieurs au correctif de filiation des emails parents. La séparation FIT/DEV/CONFIRM n'était pas effective pour les variantes. Ces résultats et checkpoints restent consultables comme historique, mais ne constituent pas une validation hors apprentissage. Un rejeu avec le manifeste corrigé est nécessaire. Les évaluations humaines n'ont pas été réalisées.**
(Détail et impact : `docs/RESULTS_STATUS.md`, section Corpus.)

12 requêtes (6 propriétés × formulation lexicale/paraphrase), CONFIRM de
l'ancien manifeste (10 865 documents ; sa séparation d'avec l'entraînement
n'était pas effective pour les variantes, cf. statut ci-dessus). Pertinence jugée par Qwen (0/1/2) sur
l'union des top-10 de chaque méthode — **simplification assumée** : pas de
double-annotation humaine (plan §8.3), à calibrer dès que possible.

## P@10 strict (relevance==2) par requête

| Propriété | Formulation | CORE | FULL | DENSE | TFIDF | BM25 |
|---|---|---:|---:|---:|---:|---:|
| relances_repetees | lexicale | 0,30 | 0,90 | 1,00 | 1,00 | 1,00 |
| relances_repetees | paraphrase | 0,20 | 0,90 | 1,00 | 0,70 | 1,00 |
| menace_resiliation | lexicale | 0,20 | 0,10 | 0,50 | 1,00 | 0,90 |
| menace_resiliation | paraphrase | 0,20 | 0,10 | 0,00 | 0,00 | 0,00 |
| incident_collectif | lexicale | 0,00 | 0,00 | 0,10 | 0,00 | 0,10 |
| incident_collectif | paraphrase | 0,00 | 0,00 | 0,30 | 0,70 | 0,60 |
| explication_montant | lexicale | 0,20 | 0,80 | 0,70 | 0,20 | 0,10 |
| explication_montant | paraphrase | 0,40 | 0,70 | 1,00 | 0,90 | 1,00 |
| coupure_repetee | lexicale | 0,00 | 0,30 | 0,80 | 0,40 | 0,30 |
| coupure_repetee | paraphrase | 0,00 | 0,20 | 0,50 | 0,40 | 0,80 |
| urgence_implicite | lexicale | 0,70 | 1,00 | 1,00 | 0,80 | 1,00 |
| urgence_implicite | paraphrase | 0,70 | 0,90 | 0,90 | 0,80 | 1,00 |

## Comparaisons par famille (moyenne lexicale+paraphrase, IC bootstrap n=6)

**FULL − CORE : +25 points, IC [+5,8 ; +45].** N'inclut pas zéro — c'est le
**premier écart favorable à l'extension** de la campagne historique (IC
bootstrap par famille, n=6 ; contraste avec E01, où FULL≈CORE sur la sonde des
axes d'augmentation). CORE seul n'a que 77
labels utilisables contre 197 pour FULL (catalogue GemmaScope-2 générique,
peu de directions correspondant à des propriétés métier comme "menace de
résiliation" ou "relance répétée") — cohérent avec l'hypothèse du plan
(§6.5/§9) : l'extension apporte un catalogue de features propre au domaine
que le cœur seul n'a pas, visible ici parce que le retrieval par propriété
dépend directement du nombre de latents pertinents disponibles, contrairement
à une sonde de classification qui peut réussir avec un signal diffus.

**FULL − DENSE : -15,8 points, IC [-26,7 ; -6,7].** N'inclut pas zéro dans
l'autre sens : l'écart en faveur de bge-m3 est statistiquement distinct de zéro sur cet échantillon. **FULL −
BM25 : -15,8 points, IC [-30,0 ; +2,5]** (borne haute tout juste positive,
proche de zéro mais pas établi). **FULL − TFIDF : -8,3 points, IC [-26,7 ;
+10,0]**, croise zéro, non établi.

## Lecture d'ensemble

Résultat à deux faces, à ne pas aplatir en un seul verdict :

1. **FULL dépasse CORE sur cet échantillon historique** (+25 points, IC
   n'incluant pas zéro à n=6 familles) — le seul écart de ce type dans la
   campagne, à revalider après rejeu sous le manifeste corrigé.
2. **FULL n'est pas encore compétitif avec un moteur dense/lexical classique**
   pour le classement pur — DENSE et BM25 restent (au moins) aussi bons,
   souvent meilleurs. Conforme à l'attente déjà écrite dans le plan (§8.5) :
   "le résultat acceptable peut être la conservation d'un moteur dense/lexical
   pour le classement et du SAE pour expliquer les thèmes."

Pour `incident_collectif`, CORE et FULL sont à 0,00 sur les deux formulations,
mais DENSE/TFIDF/BM25 ne sont pas nuls en paraphrase (0,30 / 0,70 / 0,60) : c'est
un échec des features SAE sur cette propriété, pas une preuve qu'elle est absente
du corpus. `menace_resiliation` (paraphrase) tombe à 0 pour DENSE/TFIDF/BM25 --
paraphrase possiblement trop éloignée du vocabulaire du corpus, à vérifier avant
réutilisation.

## Limites connues

- n=6 familles : les IC sont larges, lus comme une indication de
  variabilité, pas un test définitif (le plan le dit explicitement, §8.5).
- Jugements de pertinence 100% modèle (Qwen), pas de calibration humaine.
- p90 de normalisation et TFIDF ajustés directement sur CONFIRM (piste
  exploratoire d'indexation, §4.5) — pas la piste stricte FIT-only utilisée
  pour E01.
- Union jugée par requête (36-46 documents) : tous les top-10 de chaque
  méthode sont jugés, ce qui suffit pour P@10 (pas besoin de juger tout le
  corpus). Les documents jamais retournés ne sont pas jugés : un rappel ou une
  MAP demanderait un jugement exhaustif, hors budget ici.

## Fichiers

- `results_post_stage_e01_fit_1b_layer13_k5/e03_property_retrieval.json`
- Script : `scripts/post_stage/e03_property_retrieval.py`
