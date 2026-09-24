# E01 — Représentations comparables (résultat FIT→DEV, 1B/layer13/K5)

> **Statut : résultats antérieurs au correctif de filiation des emails parents. La séparation FIT/DEV/CONFIRM n'était pas effective pour les variantes. Ces résultats et checkpoints restent consultables comme historique, mais ne constituent pas une validation hors apprentissage. Un rejeu avec le manifeste corrigé est nécessaire. Les évaluations humaines n'ont pas été réalisées.**
(Détail et impact : `docs/RESULTS_STATUS.md`, section Corpus.)

Protocole strict (plan §6.4) : entraînement (SAE, TFIDF) sur **FIT seul**
(25 970 documents, 2 084 mails d'origine), évaluation sur **DEV tenu à
l'écart** (6 518 documents, 521 mails d'origine), aucune validation croisée
interne sur l'ensemble d'évaluation (contrairement à la sonde historique
`acc_axes_email`, qui fait sa propre CV à 5 plis à l'intérieur d'un seul
ensemble). CONFIRM non consulté à ce stade (ancien manifeste).

## Ce qui a été mesuré

Sonde logistique à 14 classes (axes de perturbation : émotion, urgence,
registre, orthographe, original), même protocole pour les quatre bras :

| Représentation | Accuracy DEV | Détail |
|---|---:|---|
| CORE (16 384 dim, GemmaScope-2 gelé) | 88,68 % | |
| FULL (CORE + extension, 17 408 dim) | 88,80 % | |
| TFIDF (mots 1-gramme, max_features=20000, sublinear_tf) | 87,94 % | ajusté sur FIT seul |
| DENSE (bge-m3, pooling CLS, max_length=2048) | 77,05 % | 0 % de troncature |

## Comparaisons appariées (McNemar + IC bootstrap par parent, 521 groupes)

**FULL − CORE : +0,12 point, IC [-0,08 ; +0,32], McNemar p=0,29 (p_fdr=0,29).**
L'intervalle croise zéro. Sur cette tâche, à cet effectif, l'extension
n'apporte pas de gain établi par rapport au cœur GemmaScope-2 seul. Ce
résultat ne démontre pas que l'extension est inutile — un effet plus petit
que ce que 521 groupes peuvent détecter de façon fiable reste possible, et
E03/E04 (retrieval, diffing) restent à évaluer avant toute décision finale
sur FULL vs CORE (§6.5 : le critère de représentation combine plusieurs
expériences, pas seulement les sondes).

**FULL − DENSE : +11,75 points, IC [10,73 ; 12,79], McNemar p≈1,5e-111
(survit à la correction FDR-BH).** Écart large et robuste. Comme FULL≈CORE
ici, c'est essentiellement CORE (pas l'extension) qui domine DENSE sur cette
sonde précise. Ne pas généraliser à toute tâche : bge-m3 en pooling CLS est
conçu pour la similarité sémantique de contenu, pas nécessairement pour
capturer un axe de registre/ton — sa faiblesse relative ici est plausible
mais spécifique à cette sonde, à ne pas transposer telle quelle au retrieval
sémantique (E03).

**FULL − TFIDF : +0,86 point, McNemar p=0,037 (nominal) mais p_fdr_bh=0,056
-- ne survit PAS à la correction multi-test.** Résultat non robuste à cet
effectif ; ne pas le citer comme un gain établi de FULL sur TFIDF.

## Lecture d'ensemble

Sur cette sonde, dans cette configuration (1B, layer13, K_EXTRA=5, FIT
seul), le signal utile vient presque entièrement du **cœur GemmaScope-2
gelé** : ni l'extension entraînée ni un enrichissement au-delà de TFIDF ne
sont établis comme apportant un gain à cet effectif. La comparaison qui
tient est FULL/CORE contre DENSE, pas la question motivant l'extension
elle-même (apporte-t-elle quelque chose au-delà du cœur ?). Ne pas
conclure "l'extension ne sert à rien" pour autant : c'est un résultat par
tâche et par échelle, à recouper avec E03 (retrieval par propriété, où
l'extension a un catalogue de features propre au domaine que CORE seul n'a
pas) et E04 (diffing) avant toute décision sur la représentation du
prototype.

## Pooling alternatif : moyenne des 3 plus fortes activations vs max-pooling

Même protocole FIT→DEV, même sonde à 14 classes, seule la mise en commun
token→document change (`doc_topk_mean_pool`, k=3, contre `doc_maxpool`) :

| | max-pooling (référence) | top-3-moyenne | diff | McNemar p | IC bootstrap (521 groupes) |
|---|---:|---:|---:|---:|---|
| CORE | 88,68 % | 89,09 % | -0,41 pt | 0,22 (p_fdr=0,44) | [-1,03 ; +0,20] |
| FULL | 88,80 % | 88,97 % | -0,17 pt | 0,63 (p_fdr=0,63) | [-0,77 ; +0,43] |

Léger avantage numérique du top-3-moyenne dans les deux cas, mais **aucun
des deux écarts n'est établi** (IC croisant zéro, p_fdr>0,05). Conclusion :
pas de motif de changer le max-pooling de référence sur la seule base de
cette sonde -- cohérent avec le plan (§6.3 : "Aucun choix entre dix
agrégations sur le test").

## Baseline longueur seule et stratification par longueur (DEV)

Sonde n'utilisant QUE `len(texte)` comme feature : **18,9 % d'accuracy**
(chance uniforme ≈ 7,1% à 14 classes, plus si classes déséquilibrées) --
la longueur seule porte un signal non trivial mais très inférieur à CORE/
FULL/TFIDF (~88 %), écartant l'hypothèse que ces représentations ne
feraient que capturer un artefact de longueur.

| Tranche (caractères) | n | acc CORE | acc FULL | acc longueur seule |
|---|---:|---:|---:|---:|
| 8-956 | 1626 | 93,1 % | 93,4 % | 28,6 % |
| 957-1235 | 1633 | 88,5 % | 88,2 % | 15,1 % |
| 1236-1554 | 1628 | 87,7 % | 87,8 % | 9,5 % |
| 1555-4394 | 1631 | 85,5 % | 85,8 % | 22,5 % |

Accuracy CORE/FULL décroît régulièrement avec la longueur (93→86 %) : les
documents courts sont plus faciles à classer sur cet axe. CORE et FULL
restent à moins de 0,3 point l'un de l'autre dans **chaque** tranche -- le
constat FULL≈CORE de la section précédente n'est pas masqué par un effet
de longueur qui favoriserait l'un des deux bras sur un sous-ensemble
particulier.

## Limites connues restantes

- Entraînement effectué avec un plafond réservoir à 8 000 000 tokens
  **entièrement rempli** (`N_TOKENS_EXTRA_TRAIN=8000000`) : FIT offre donc
  *au moins* 8M tokens uniques, la borne réelle n'est pas mesurée (plafond
  choisi trop bas pour la mesurer, cf. §6.1 du plan qui demande de rapporter
  ce nombre). À refaire avec un plafond plus haut si on veut le chiffre
  exact plutôt qu'une borne.
- Comparaison faite sur DEV, pas CONFIRM (attendu à ce stade -- DEV sert aux
  choix de développement, CONFIRM est réservé à l'évaluation finale une fois
  le protocole figé, §4.2/§6.4).
- Une seule graine d'entraînement (E05 prévoit 2-3 graines pour évaluer la
  stabilité, pas encore fait ici).
- Stratification par longueur faite uniquement pour CORE/FULL max-pooling
  (pas DENSE/TFIDF ni le pooling alternatif) -- suffisant pour vérifier
  l'absence de confusion longueur/FULL-vs-CORE, pas un audit complet de
  toutes les combinaisons.

## Fichiers

- Checkpoint : `results_post_stage_e01_fit_1b_layer13_k5/p1_extended_sae.pt`,
  `p1_frozen_core_d1024_k5.pt`.
- Comparaison représentations : `results_post_stage_e01_fit_1b_layer13_k5/e01_representation_comparison.json`.
- Pooling/longueur : `results_post_stage_e01_fit_1b_layer13_k5/e01_pooling_and_length_analysis.json`.
- Scripts : `scripts/post_stage/e01_compare_representations.py`,
  `scripts/post_stage/e01_pooling_and_length_analysis.py`.
