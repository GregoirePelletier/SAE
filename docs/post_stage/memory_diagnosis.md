# E00 — Diagnostic mémoire (première mesure réelle)

Analyse de `results_post_stage_e00_profile_1b/cache/resources_profile.jsonl`
(job Slurm 48515, 1B/layer13, 250k tokens résidus, 20k chunks filler --
échelle réduite volontairement, cf. limite méthodologique en fin de
document). Rien n'a été corrigé avant cette mesure (§5.4/§5.7 du plan) :
ceci décrit le comportement du code tel qu'audité.

## 1. Ce que le job a réellement fait

Le job a été tué par `TIMEOUT` (2h) pendant l'étape "Sonde logistique sur SAE
activations (axes email, corpus principal)" -- **pas un OOM**
(`memory.events.oom_kill = 0` sur toute la durée). Chronologie reconstruite
à partir des dates de mtime des fichiers de cache et des marqueurs
`[timing]` :

| t (s) | Étape | Durée |
|---:|---|---:|
| 0–20 | Chargement corpus (emails+augmentés) | ~2s |
| 20–24 | Préparation corpus diffing (energy/sports/support) | ~5s |
| ~24–1005 | Extraction Gemma-3-1B (36 602 docs train + 20 000 filler) | ~16 min |
| ~1005–1025 | `torch.stack` + sauvegarde compacte + libération du modèle | ~20s (voir §2) |
| ~1025–1987 | Entraînement `SAEBoostResidualSAE` (10 époques, 250k tokens) | ~16 min |
| ~1987–2628 | Chargement du juge Qwen3.8-27B + labellisation (10 features) | ~11 min |
| ~2628–2788 | Ré-encodage + NPMI | ~2,5 min |
| ~2788–7200 (tué) | Sondes logistiques en aval (`downstream_classification`) | **>72 min sans terminer** |

## 2. Confirmation empirique de C1 (filler dans `all_doc_sae_acts`)

Le pic `t=1005s` est net et isolé (une seule ligne d'échantillonnage,
intervalle 20s) :

| t (s) | VmRSS | anon (cgroup) |
|---:|---:|---:|
| 984 | 12,93 Go | 11,87 Go |
| **1005** | **16,44 Go** | **15,38 Go** |
| 1025 | 4,84 Go | 3,74 Go |

Ce pic correspond exactement à la séquence attendue de `saev5.py` (fin de la
boucle d'extraction) : la liste Python `all_doc_sae_acts` (36 602 lignes
réelles + 20 000 lignes filler à zéro, ~56,6k lignes au total) et le tenseur
`torch.stack(all_doc_sae_acts)` **coexistent transitoirement** avant que
`del llm, tokenizer, ...` + `_trim_host_memory()` ne libère l'espace --
exactement le mécanisme décrit par le plan (C1, §5.3). Chute mesurée :
16,44 → 4,84 Go, soit **~11,6 Go pour ~56,6k lignes** dans cette
configuration réduite.

**Extrapolation à l'échelle production** (job 45803, référence C10, ~1,2M
lignes filler + 36,6k train ≈ 1,24M lignes, soit **~22× plus de lignes**
qu'ici) : `~11,6 Go × 22 ≈ 255 Go` rien que pour ce doublement transitoire --
cohérent en ordre de grandeur avec le MaxRSS de 358,43 Go effectivement
mesuré sur ce job (préflight, §sacct). **C1 est donc vraisemblablement le
contributeur dominant du dépassement mémoire historique, pas seulement une
inefficacité parmi d'autres** -- conclusion qui n'était pas établie
quantitativement avant cette mesure (le plan la formulait comme hypothèse
de lecture de code, pas comme mesure).

## 3. Second contributeur identifié, non documenté par le plan initial : mmap du juge

Entre t=2207s et t=2628s, VmRSS grimpe de 11,9 à 63,5 Go **alors que `anon`
(cgroup) reste plat à 9,65 Go** -- la croissance colle presque
exactement à la croissance du champ cgroup `file` (55,75 → 93,07 Go). Ceci
est la signature d'une lecture **memory-mapped** (pas une allocation
anonyme) : le chargement de Qwen3.8-27B bf16 (~52 Go de poids) via
`from_pretrained` mappe les fichiers safetensors en pages, comptées en
`VmRSS`/`file` mais pas en `anon`. Chute nette à t=2628-2648s (63,5 → 8,05 Go)
une fois les poids transférés sur GPU. **Ce n'est pas un bug** (comportement
mmap attendu pour un modèle de cette taille) mais un poste de mémoire réel,
distinct de C1, qui explique une partie substantielle du "file cache" déjà
observé en production sans qu'il ait été isolé du reste jusqu'ici. À garder
en tête pour dimensionner `--mem` sur tout job qui charge le juge (marge
d'au moins ~55 Go rien que pour ce mmap, en plus du reste).

## 4. `downstream_classification` : lent, probablement pas cassé

Le "blocage" de 72+ minutes en fin de job a d'abord semblé anormal (voir
échange précédent). Reconsidéré à la lumière des logs : 3 messages
`ConvergenceWarning` (lbfgs, max_iter=1000 atteint) sont apparus avant que
le job soit tué, ce qui signifie que **3 des 5 plis de validation croisée
ont réellement terminé** (~24-25 min/pli), pas un blocage infini. Le code
utilise bien `scipy.sparse.csr_matrix` avant `LogisticRegression.fit`
(`src/analysis/metrics.py:154-196`, correctif déjà en place) -- la lenteur
vient du problème lui-même : régression logistique multinomiale (14 classes,
solveur `lbfgs`) sur **36 602 documents** (pas 250k tokens -- cf. §5 ci-dessous)
en ~17 408 dimensions. `RESULTS_TESTS.md` documente des runs historiques
ayant terminé cette même sonde (79-93,5% selon la config), généralement sous
des budgets de 48h -- cette durée est donc probablement dans l'ordre de
grandeur attendu à cette échelle de corpus, pas une régression. Pas de
correctif proposé ici ; à surveiller si un budget de temps plus serré est
nécessaire pour E01 (qui répète cette sonde sur plusieurs bras de
représentation, §6.4).

## 5. Défaut méthodologique de CE protocole de profilage (à corriger avant le prochain run)

`N_TOKENS_EXTRA_TRAIN=250000` ne borne que le réservoir de résidus pour
l'entraînement de l'extension -- **pas la taille du corpus documentaire**.
Correction après vérification du code (`saev5.py:298`) : `MAX_AUGMENTED_PER_MAIL`
défaut réellement à **13**, pas à l'illimité comme affirmé dans une première
version de ce diagnostic (l'appel direct utilisé pour compter les labels,
§ci-dessous, passait `None` explicitement -- une erreur de méthode dans le
diagnostic, pas dans le pipeline). 36 602 documents train est donc la taille
de corpus **normale et attendue** à ce défaut (3 474 mails × jusqu'à 13
variantes, cohérent avec les 39 949 variantes acceptées mentionnées dans le
plan, §2.1), pas un signe que le job a échappé à une borne. Ce "profilage
borné" n'était donc borné qu'en tokens résidus, jamais en nombre de
documents -- même à `N_TOKENS_EXTRA_TRAIN` minuscule, tout job qui ne
positionne pas explicitement `MAX_AUGMENTED_PER_MAIL` à une petite valeur
(ex. 1-2) traite le corpus à sa taille de production pour l'extraction
document-niveau et les sondes en aval. Un futur profilage réellement rapide
doit positionner `MAX_AUGMENTED_PER_MAIL` explicitement bas en plus de
`N_TOKENS_EXTRA_TRAIN`.

## 6. Validation du correctif C1

Deux validations complémentaires, l'une isolée (sans confondre avec la
réduction de corpus), l'autre en conditions réelles.

**Benchmark synthétique isolé** (CPU, aucun modèle/donnée réelle, échelle
volontairement réduite après qu'une première tentative à l'échelle
production -- 1,2M lignes filler -- a été tuée par OOM en tournant **par
erreur sur le nœud frontal** ; ne pas répéter à cette échelle hors `sbatch`) :
à 2 000 documents réels + 20 000 filler, construction "ancienne" (zéro par
filler + `torch.stack`) → pic transitoire de 690 Mo ; construction "corrigée"
(filler jamais ajouté) → 65 Mo. Ratio ~10,6×, cohérent avec le ratio de
lignes (22 000/2 000 = 11×) -- confirme le mécanisme indépendamment de toute
mesure sur pipeline réel.

**Run réel post-correctif** (job 48530, même config que 48515 --
`MAX_AUGMENTED_PER_MAIL=1` en plus pour tenir dans la fenêtre) :
`COMPLETED` en 35 min 39 s (le job 48515 équivalent avait été tué par
`TIMEOUT` à 2h). Pic mémoire de la transition extraction→entraînement :
9,76 → 4,82 Go (Δ 4,94 Go, à comparer au format toujours compact sur disque,
vérifié `n_total=27 814, kept_rows.shape=[7 814, 16384]` -- exactement
`n_train(6 567) + n_test+n_diff(1 247)`, filler totalement absent). Le
run entier a terminé avec des métriques dans la plage historique attendue
(ΔFVE domaine +0,2856, `acc_SAE` axes email 84,4% sur 14 classes) : le
correctif ne change aucun résultat scientifique, seulement la mémoire de
construction.

**Attention à ne pas sur-interpréter** : ce run a un corpus bien plus petit
que 48515 (`MAX_AUGMENTED_PER_MAIL=1` vs 13 par défaut) et a probablement
bénéficié d'un cache page OS déjà chaud pour les poids Qwen (même nœud
`dgx-h100` que 48515) -- le pic mémoire global du run (18,15 Go) n'est donc
**pas** directement comparable au pic de 48515 (103 Go) comme mesure isolée
du correctif ; c'est le benchmark synthétique ci-dessus, à corpus égal, qui
isole proprement l'effet du correctif.

## 7. Prochaine étape

Le correctif prioritaire identifié par cette mesure : ne jamais construire
`all_doc_sae_acts` avec les lignes filler en mémoire (ni pendant la
reprise après coupure, ni en fin d'extraction), plutôt que de les retirer
seulement à la sérialisation (déjà fait, `save_doc_acts_sparse_filler`).
Implique de retravailler les trois découpages positionnels
(`train_doc_acts`/`test_doc_acts`/`diff_doc_acts`, `saev5.py:1728-1730`) et
les deux écritures indexées du ré-encodage (`saev5.py:1590,1666`) pour
utiliser un index compact plutôt que `doc_global_idx` direct -- le
chargement legacy (`load_all_doc_acts`, ~20 scripts consommateurs) n'a pas
besoin de changer, il continue de reconstruire un tenseur zero-paddé à la
demande depuis le fichier compact déjà existant.
