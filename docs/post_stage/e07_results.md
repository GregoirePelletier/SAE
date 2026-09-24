# E07 — Clustering ciblé par axe, comparé aux alternatives (1B/layer13/K5)

> Ce document décrit la première exécution, faite avec un découpage erroné des variantes
> augmentées (voir `docs/RESULTS_STATUS.md`) : ce ne sont pas des évaluations hors
> apprentissage. Le rejeu avec le découpage corrigé est en cours.

3 axes fixes sur 400 parents CONFIRM échantillonnés (1 email représentatif
chacun, graine 42) : `type_probleme`, `action_attendue`, `registre_urgence`.
CORE et FULL restreints au même budget de 40 features par similarité
label↔requête (`select_latents_by_similarity`), SpectralClustering sur
affinité Jaccard. DENSE (bge-m3, documents entiers, aucune troncature —
0% de troncation à max_length=2048) et TFIDF : KMeans cosine sur les mêmes
400 documents. k=4 fixé pour toutes les méthodes/tous les axes.

## Résultat principal : le contrôle de l'axe fonctionne, mais seulement pour CORE/FULL

**DENSE et TFIDF produisent EXACTEMENT les mêmes clusters (mêmes tailles,
mêmes libellés LLM, même accuracy, même conductance) sur les 3 axes** —
attendu structurellement (ni l'un ni l'autre n'a de mécanisme de sélection
par axe dans ce protocole), mais la conséquence pratique doit être dite
clairement : **un utilisateur qui demande un regroupement par
`action_attendue` ou `registre_urgence` avec ces deux baselines obtient la
même partition que pour `type_probleme`** — les deux se re-regroupent
systématiquement autour de la structure "type de problème" (facturation,
coupure, activation, résiliation), qui domine apparemment la représentation
dense/lexicale de ce corpus indépendamment de la question posée.

**CORE et FULL, à l'inverse, produisent des partitions numériquement
différentes sur les 3 axes** (tailles de cluster différentes, features
sélectionnées différentes) — le mécanisme de contrôle par axe fonctionne
bien mécaniquement. C'est la question centrale du plan (§12 : "la question
est le contrôle de l'axe de regroupement") et la réponse est positive sur ce
point précis : **c'est une capacité que CORE/FULL ont et que DENSE/TFIDF
n'ont pas par construction.**

## Mais le contenu sémantique ne suit pas l'axe demandé pour 2/3 axes

| Axe | FULL reflète bien l'axe ? | Accuracy de réassignation FULL (par cluster) |
|---|---|---|
| `type_probleme` | **Oui** — libellés authentiquement typés par problème (facturation, résiliation/activation, coupures) | 0,72 – 0,97 |
| `action_attendue` | Partiel — libellés encore dominés par le type de problème, pas par l'action demandée | 0,12 – 0,96 (un cluster à 0,12) |
| `registre_urgence` | Faible — un seul cluster ("informel/mixte") touche vaguement au registre, les 3 autres restent thématiques | 0,02 – 0,53 (deux clusters <0,3) |

Lecture : la sélection de features par similarité change bien selon l'axe
(40 features différentes à chaque fois), et le clustering qui en résulte
change aussi — mais le catalogue interprétable disponible (77 CORE / 197
FULL sur ce run) semble dominé par des directions de type "problème", pas
par des directions d'action ou de registre/tonalité, même quand la requête
demande explicitement ces axes. Une accuracy de réassignation proche de 0
sur un cluster (ex. FULL/`registre_urgence` cluster 3, 0,015 ; CORE/`registre_urgence`
cluster 3, 0,0) signale un libellé essentiellement non reproductible —
ne pas le citer comme un thème établi.

## CORE vs FULL

Sur `type_probleme`, FULL bat nettement CORE en accuracy de réassignation
(0,72–0,97 contre 0,04–0,79) — cohérent avec E03 (l'extension apporte un
catalogue plus riche pour le retrieval par propriété). Sur les deux autres
axes, l'écart n'est **pas** univoque (CORE et FULL ont chacun des clusters
à accuracy très basse) — ne pas généraliser le gain de `type_probleme` aux
deux autres axes.

## Conductance en espace DENSE (diagnostique, pas une preuve de supériorité)

Tous les z-scores sont fortement négatifs (clusters plus compacts qu'un
tirage aléatoire de même taille, dans l'espace bge-m3) — y compris pour
CORE/FULL, formés dans un espace Jaccard totalement différent : signal
rassurant que leurs clusters correspondent à une vraie structure documentaire,
pas une partition arbitraire. DENSE lui-même est mécaniquement le plus
compact dans son propre espace (z jusqu'à -41, contre -3 à -37 pour
CORE/FULL) — à lire comme un artefact de circularité (DENSE optimise
directement dans l'espace où on le mesure), pas une preuve de supériorité.

## Couverture ("hors axe / sans signal")

3-4 documents sur 400 (coverage ≥0,99) exclus car leur vecteur restreint
aux 40 features de l'axe est nul — couverture large, pas un problème pour
ce run.

## Écart trouvé et corrigé avant le résultat final

Un premier run (job 49084, `top_k_features=150`) donnait des
`cluster_sizes` CORE **strictement identiques sur les 3 axes** — le
catalogue CORE interprétable (77 features) est plus petit que 150, donc
`select_latents_by_similarity` renvoyait le catalogue ENTIER quel que soit
l'axe, rendant la restriction inopérante pour CORE (seul FULL, avec 197
candidats, était réellement restreint). Corrigé (`TOP_K_FEATURES=40`,
job 49087) — vérifié que les tailles de cluster diffèrent bien désormais
sur les 3 axes pour CORE et FULL.

## Limites connues

- **Audit humain aveugle de paires intra/inter-cluster (20-30 par axe,
  plan §12.3) non fait** — nécessite Grégoire, reste en attente. Les
  accuracies de réassignation LLM ci-dessus sont un diagnostic secondaire,
  pas une validation humaine.
- TFIDF/DENSE ajustés directement sur l'échantillon CONFIRM (piste
  exploratoire), pas de baseline FIT-only.
- k=4 fixé a priori pour toutes les méthodes — pertinent pour comparer
  à budget égal, mais pourrait masquer qu'un axe se prête mieux à un
  nombre de groupes différent.
- Catalogue interprétable restreint (77 CORE / 197 FULL au total sur ce
  run) — la difficulté à faire ressortir les axes `action_attendue`/
  `registre_urgence` peut refléter la taille du catalogue autant que le
  choix de méthode.

## Fichiers

- `results_post_stage_e01_fit_1b_layer13_k5/e07_clustering.json`
- Script : `scripts/post_stage/e07_clustering.py`
- Job SLURM : 49087 (h100, 1 GPU, COMPLETED 00:10:41 ; job 49084 avec
  `top_k_features=150` avait le bug de restriction inopérante ci-dessus)
