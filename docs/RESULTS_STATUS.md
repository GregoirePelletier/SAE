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

Le rejeu avec le découpage corrigé est terminé pour E01 à E07. Ses résultats remplacent ceux de la
première exécution, qui restent décrits dans chaque compte rendu à titre historique.

| | Question | Rejeu (découpage corrigé) | Première exécution |
|---|---|---|---|
| E00 | Mémoire du pipeline | Correctif des vecteurs documentaires validé ; un second poste reste à corriger | — |
| E01 | L'extension améliore-t-elle une sonde de classification (14 axes d'augmentation) ? | Non ; TF-IDF fait mieux que FULL de 1,6 point | Non |
| E02 | Part de features interprétables (150 CORE + 150 EXTRA) | 207/300 | 197/300 |
| E03 | Retrouver des emails selon une propriété (P@10 moyen) | FULL 49 %, dense 60 % ; aucun écart significatif | FULL > CORE de 25 points (significatif) |
| E04 | Comparer deux populations (ton paniqué / calme) | 8 hypothèses sur 8 confirmées, dans le sens annoncé | 6 sur 8, une inversée |
| E05 | Stabilité des features entre entraînements | Directions stables, emails retrouvés instables | Même conclusion |
| E06 | Associations entre propriétés | 5 paires sur 8 confirmées, 2 intéressantes | 4 sur 8 |
| E07 | Regroupement selon une question | Les groupes changent selon l'axe mais ne le suivent pas | Suivait le type de problème |
| E08 | Mini-pilote avec des analystes | Outil prêt, aucune séance tenue | — |
| E09 | Run à 100 M tokens | Non fait (budget à zéro) | — |

En résumé :

- **E01** : l'extension n'apporte rien sur la sonde de classification (FULL − CORE = +0,06 point),
  et TF-IDF fait légèrement mieux que FULL.
- **E02** : l'extension fournit l'essentiel des features nommables (131 EXTRA contre 76 CORE sur
  150 chacune).
- **E03** : les features retrouvent des emails pour certaines propriétés, mais ne battent pas
  les moteurs dense ou lexicaux ; l'avantage de FULL sur CORE vu dans la première exécution
  n'est plus significatif.
- **E04** : la chaîne « découverte, hypothèses, vérification » fonctionne, sur un contraste
  synthétique très marqué.
- **E05** : environ 80 % des features se retrouvent d'un entraînement à l'autre, quelle que soit
  l'initialisation, mais les emails de tête d'un groupe de features changent.
- **E06** : associations vérifiées, pour la plupart attendues ; deux pistes à examiner (travail
  à domicile et heure de coupure, consommation stable et contestation).
- **E07** : le regroupement ciblé n'est pas démontré avec le catalogue actuel.

Emplacements : `results_post_stage_e01_fit_1b_layer13_k5_v2/` (E01, E02, analyses E05),
`results_post_stage_e01_fit_1b_layer13_k5_v2_eval/` (E03, E04, E06, E07), et les
entraînements E05 dans `results_post_stage_e05_*_v2/`. Le checkpoint utilisé pour E03 à E07 est
le même que celui d'E01 (empreintes identiques, chargé sans réentraînement).

## Ce qu'il reste à faire

1. **Évaluations humaines**, aucune n'a été faite : vérifier 60 à 100 features (E02), contrôler les
   jugements du modèle juge (E03, E04), auditer les associations (E06) et les groupes (E07),
   organiser le mini-pilote avec 2 ou 3 analystes (E08).
2. **Correctifs de méthode non faits** :
   - E03 : dédupliquer le top-10 par email d'origine et comparer CORE et FULL avec des catalogues
     de même taille ;
   - E04 : tenir compte des emails d'origine communs aux deux groupes comparés, garder la liste des
     emails tirés, tester d'autres contrastes ;
   - E06 : équilibrer les paires candidates entre CORE et EXTRA, mieux filtrer les synonymes ;
   - E07 : essayer un catalogue plus large ou un nombre de groupes adapté à chaque axe ;
   - régénérer `augmented_mails.jsonl` avec `parent_sha1` pour ne plus dépendre de positions.
3. **Mémoire** : le second poste mémoire (`p1_all_doc_acts_ext_d*.pt`) n'est pas corrigé ; ne pas
   lancer de run volumineux avant (voir `docs/post_stage/memory_diagnosis.md`).
4. **Données réelles** : toutes les conclusions portent sur un corpus synthétique ; aucune n'a été
   vérifiée sur des emails réels.
