# E03 — Retrieval par propriété sur CONFIRM (1B/layer13/K5)

12 requêtes (6 propriétés × formulation lexicale/paraphrase), CONFIRM
(10 865 documents, jamais entraîné). Pertinence jugée par Qwen (0/1/2) sur
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
**premier gain établi de l'extension** dans toute la campagne (contraste
avec E01, où FULL≈CORE sur la sonde d'intention). CORE seul n'a que 77
labels utilisables contre 197 pour FULL (catalogue GemmaScope-2 générique,
peu de directions correspondant à des propriétés métier comme "menace de
résiliation" ou "relance répétée") — cohérent avec l'hypothèse du plan
(§6.5/§9) : l'extension apporte un catalogue de features propre au domaine
que le cœur seul n'a pas, visible ici parce que le retrieval par propriété
dépend directement du nombre de latents pertinents disponibles, contrairement
à une sonde de classification qui peut réussir avec un signal diffus.

**FULL − DENSE : -15,8 points, IC [-26,7 ; -6,7].** N'inclut pas zéro dans
l'autre sens : bge-m3 bat FULL de façon établie sur cette tâche. **FULL −
BM25 : -15,8 points, IC [-30,0 ; +2,5]** (borne haute tout juste positive,
proche de zéro mais pas établi). **FULL − TFIDF : -8,3 points, IC [-26,7 ;
+10,0]**, croise zéro, non établi.

## Lecture d'ensemble

Résultat à deux faces, à ne pas aplatir en un seul verdict :

1. **L'extension apporte un gain net et établi sur CORE** — le seul de toute
   la campagne jusqu'ici. Elle mérite sa place dans le catalogue de features
   utilisé pour le retrieval, pas seulement comme diagnostic.
2. **FULL n'est pas encore compétitif avec un moteur dense/lexical classique**
   pour le classement pur — DENSE et BM25 restent (au moins) aussi bons,
   souvent meilleurs. Conforme à l'attente déjà écrite dans le plan (§8.5) :
   "le résultat acceptable peut être la conservation d'un moteur dense/lexical
   pour le classement et du SAE pour expliquer les thèmes."

Deux propriétés (`incident_collectif`) ressortent quasi nulles pour TOUTES
les méthodes (P@10 ≤ 0,10 pour core/full sur les deux formulations) — soit
la propriété est rare/mal formulée pour ce corpus synthétique, soit CONFIRM
n'a pas assez d'exemples réellement collectifs. `menace_resiliation` (paraphrase)
tombe aussi à 0 pour DENSE/TFIDF/BM25 -- possible paraphrase trop éloignée
du vocabulaire du corpus, à vérifier avant de la réutiliser telle quelle.

## Limites connues

- n=6 familles : les IC sont larges, lus comme une indication de
  variabilité, pas un test définitif (le plan le dit explicitement, §8.5).
- Jugements de pertinence 100% modèle (Qwen), pas de calibration humaine.
- p90 de normalisation et TFIDF ajustés directement sur CONFIRM (piste
  exploratoire d'indexation, §4.5) — pas la piste stricte FIT-only utilisée
  pour E01.
- Union jugée par requête (36-46 documents) : les documents jamais retournés
  par aucune méthode ne sont pas jugés — un vrai P@10 populationnel
  nécessiterait un jugement exhaustif du corpus, hors budget ici.

## Fichiers

- `results_post_stage_e01_fit_1b_layer13_k5/e03_property_retrieval.json`
- Script : `scripts/post_stage/e03_property_retrieval.py`
