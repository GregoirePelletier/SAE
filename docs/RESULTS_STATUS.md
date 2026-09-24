# État des résultats

Ce document dit, pour chaque expérience, ce qui a été mesuré, sur quelles données, et ce qu'il
reste à faire. Le détail de chaque expérience est dans `docs/post_stage/eNN_results.md`, le
journal des expériences antérieures à la soutenance dans `RESULTS_TESTS.md`.

## Le point à connaître avant tout

La campagne post-soutenance (E01 à E07) a d'abord été exécutée avec un rattachement erroné des
variantes augmentées à leur mail d'origine. Le split FIT / DEV / CONFIRM des mails d'origine était
correct, mais pas celui des variantes :

- 21 817 variantes sur 39 879 (54,7 %) étaient dans le mauvais split ;
- sur les 23 886 variantes utilisées pour l'entraînement (FIT), 5 803 appartenaient en réalité à
  CONFIRM et 3 587 à DEV ;
- sur les 9 996 variantes utilisées pour l'évaluation (CONFIRM), 2 645 seulement étaient bien CONFIRM.

**Cause.** `run_augmentation.py` numérote `parent_id` sur les 3 480 lignes renvoyées par
`load_mails_tsv`. Le rattachement utilisait les 3 474 lignes gardées par `load_and_clean_emails`,
qui écarte 6 mails devenus vides après nettoyage : au-delà du 44ᵉ mail, chaque variante était
associée à un voisin décalé de 1 à 6 rangs.

**Correction.** Le code utilise désormais les positions d'origine (tests associés), et
`configs/post_stage/split_assignments.json` a été régénéré. L'ancien manifeste est conservé dans
`configs/post_stage/legacy_pre_parent_join_fix/`.

**Conséquence.** Les résultats de la première exécution (dossiers `results_post_stage_*` sans
suffixe) restent consultables mais ne sont pas des évaluations hors apprentissage. La campagne a
été relancée avec le split corrigé (dossiers suffixés `_v2`), voir ci-dessous.

## Résultat de référence du rapport de stage

Taux d'interprétabilité des features de l'extension : **65,7 % (197/300)**, IC95 % [60,1 % ;
70,8 %], Gemma-3-12B, couche 31, `K_EXTRA=5`, juge Qwen3.8-27B (`RESULTS_TESTS.md` §119). Toutes
les valeurs antérieures (§79, §82, §94 à §96) sont remplacées par celle-ci. Avec ce protocole,
l'interprétabilité n'augmente pas avec la taille du modèle (1B, 4B, 12B, 27B ; p=0,34).
Cette mesure n'utilise pas le split FIT / DEV / CONFIRM et n'est pas concernée par le problème
ci-dessus.

## Campagne post-soutenance (Gemma-3-1B, couche 13)

| | Question | Première exécution (split erroné) | Rejeu, split corrigé |
|---|---|---|---|
| E00 | Mémoire du pipeline | Correctif des vecteurs documentaires validé | — |
| E01 | L'extension améliore-t-elle une sonde de classification (14 axes d'augmentation) ? | Non | Non ; TF-IDF fait mieux que FULL de 1,6 pt |
| E02 | Part de features interprétables (150 CORE + 150 EXTRA) | 197/300 | 207/300 |
| E03 | Retrouver des emails selon une propriété (P@10) | FULL > CORE de 25 pt, FULL < dense de 16 pt | en cours |
| E04 | Comparer deux populations (urgence panique / calme) | 6 hypothèses sur 8 confirmées | en cours |
| E05 | Stabilité des features entre entraînements | Directions stables, emails retrouvés instables | Même conclusion |
| E06 | Associations entre propriétés | 4 paires sur 8 confirmées | en cours |
| E07 | Regroupement selon une question | Fonctionne pour « type de problème » seulement | en cours |
| E08 | Mini-pilote avec des analystes | Outil prêt, aucune séance tenue | — |
| E09 | Run à 100 M tokens | Non fait (budget à zéro) | — |

Résultats détaillés du rejeu :

- **E01** (`results_post_stage_e01_fit_1b_layer13_k5_v2/`, FIT 26 112 documents, DEV 6 471) :
  FULL − CORE = +0,06 pt, IC [−0,29 ; +0,42] ; FULL − dense = +11,3 pt ; FULL − TF-IDF = −1,6 pt,
  IC [−2,3 ; −0,9]. Dans la première exécution, TF-IDF et FULL n'étaient pas distinguables.
- **E02** : 207 features interprétables sur 300 (76 CORE, 131 EXTRA), contre 197 auparavant, sur un
  autre checkpoint. Vérification humaine non faite.
- **E05** (4 entraînements de l'extension : 2 initialisations PCA, 2 aléatoires) : environ 80 %
  des features se retrouvent d'un entraînement à l'autre (cosinus ≥ 0,7 et profils corrélés), quel
  que soit le type d'initialisation ; 478 sur 674 dans les trois graines PCA. Les groupes de
  features sont bien plus cohérents qu'au hasard, mais les 100 emails les plus activés par un
  groupe changent d'un entraînement à l'autre (aucun groupe significatif avec le score max, 1 à 7
  avec la moyenne des features actives).

La qualité d'entraînement est identique à la première exécution (part de variance expliquée 0,905,
1,4 à 5,8 % de features EXTRA mortes).

État des jobs au moment de la rédaction : encodage de CONFIRM terminé (job 50939, dossier
`results_post_stage_e01_fit_1b_layer13_k5_v2_eval/`), E03, E04, E06 et E07 en cours d'exécution
(jobs 50940 à 50943).

## Ce qu'il reste à faire

1. **Finir le rejeu** : vérifier les jobs 50940 à 50943 (`sacct`), reporter les résultats d'E03,
   E04, E06 et E07 dans leurs fichiers `docs/post_stage/` et dans le tableau ci-dessus.
2. **Évaluations humaines**, aucune n'a été faite : vérifier 60 à 100 features (E02), contrôler les
   jugements du modèle juge (E03, E04), auditer les associations (E06) et les groupes (E07),
   organiser le mini-pilote avec 2 ou 3 analystes (E08).
3. **Correctifs de méthode non faits** :
   - E06 : équilibrer les paires candidates entre CORE et EXTRA (demande un rejeu) ;
   - E04 : tenir compte des mails d'origine communs aux deux groupes comparés ;
   - E03 : dédupliquer le top-10 par mail d'origine ;
   - régénérer `augmented_mails.jsonl` avec `parent_sha1` pour ne plus dépendre de positions.
4. **Mémoire** : le second poste mémoire (`p1_all_doc_acts_ext_d*.pt`) n'est pas corrigé ; ne pas
   lancer de run volumineux avant (voir `docs/post_stage/memory_diagnosis.md`).

Déjà corrigé : le rattachement des variantes (ci-dessus) et le champ `percentage_difference`
d'E04, qui recevait un log-odds au lieu d'un écart de fréquence (pris en compte par le rejeu d'E04).
