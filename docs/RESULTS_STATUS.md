# Statut des résultats — post-soutenance (E00-E09)

Ce document répond à une seule question : **où en est chaque expérience, et à
quel point peut-on en citer le résultat sans réserve ?** Le détail statistique
complet reste dans `docs/post_stage/eNN_results.md` (un run) et
`RESULTS_TESTS.md` (chiffres historiques, avant la soutenance) — ce document
n'en recopie pas les nombres, il pointe vers eux et signale ce qui reste
ouvert.

Run canonique de la campagne post-soutenance : `results_post_stage_
e01_fit_1b_layer13_k5/` (1B, layer 13, K_EXTRA=5, FIT/DEV/CONFIRM figés dans
`configs/post_stage/`). E00 (profilage mémoire) a ses propres répertoires
séparés (`results_post_stage_e00_profile_1b*/`), non lus par E01-E07.

**Réserve la plus importante, à lire avant tout le reste** : les
résultats E01-E07 ont été calculés sous un split FIT/DEV/CONFIRM dont le
rattachement des variantes était décalé (corrigé depuis, cf. section
« Corpus »). Ils sont à rejouer avant d'être cités comme évalués hors
entraînement.

## Taux d'interprétabilité de référence (tout le dépôt, pas seulement post-soutenance)

**65,7% (197/300)**, IC95% [60,1% ; 70,8%] — run `R0`, `RESULTS_TESTS.md` §119,
config par défaut du dépôt (`MODEL_SIZE=12b`, layer 31, `K_EXTRA=5`), sous
protocole intégralement corrigé (sélection stratifiée, juge `Qwen3.8-27B`
découplé de l'extracteur, négatif odd-one-out corrigé, déduplication par mail
parent). Toute mesure antérieure (§79 89,3%, §82 82,0%, §94 94,0%, §95/§96)
est supersédée — cf. `CLAUDE.md` section Diagnostics point 5 pour le détail
de pourquoi chacune ne tient plus. Aucune tendance d'échelle du modèle
extracteur/juge n'est détectable sous ce protocole (§119, Cochran-Armitage
p=0,34) — ne pas citer de progression 1B→27B comme résultat établi.

## E00 — Diagnostic mémoire et correctif

**Clos.** Fuite mémoire (`all_doc_sae_acts`) diagnostiquée et corrigée
(commit `a640ed7`), validée par benchmark synthétique + rerun réel (job
48530). Un second sink (`p1_all_doc_acts_ext_d*.pt`) identifié mais **non
corrigé** — nécessite d'auditer 20 scripts consommateurs avant de le
toucher sans casser leur indexation. Détail : `docs/post_stage/
memory_diagnosis.md`.

## Corpus — freeze FIT/DEV/CONFIRM — jointure corrigée, résultats E01-E07 à rejouer

**Cause (confirmée, corrigée)** : `load_mails_tsv` renvoie 3480 lignes,
`load_and_clean_emails` en garde 3474 (6 lignes vides après retrait de
l'objet : positions 43, 707, 1243, 1344, 1366, 2149).
`run_augmentation.py` numérote `parent_id` sur les 3480 ; le repli positionnel
(`parent_sha1` absent du JSONL gelé) énumérait les 3474 survivantes, donc
tout `parent_id >= 44` était rattaché à un parent décalé de 1 à 6 rangs,
sans erreur visible. Code corrigé (`return_positions`, commit dédié + tests) ;
les splits par PARENT sont inchangés (3474 parents, 2084/521/869, identiques
à l'octet près), seul le rattachement des variantes change.

**Impact mesuré** (variantes : 39 879 avant, 39 949 après — les 70 « non
rattachées » l'étaient uniquement par dépassement de plage) :

- 21 817 variantes (54,7 %) changent de split.
- Ancien FIT (23 886 variantes, celui de l'entraînement du SAE de référence,
  de TF-IDF et des sondes) : 14 496 sont réellement FIT, **5 803 réellement
  CONFIRM (24,3 %) et 3 587 réellement DEV** — soit 39,3 % du corpus
  d'entraînement hors FIT.
- Ancien CONFIRM (9 996 variantes, évalué en E03/E04/E06/E07) : seulement
  2 645 réellement CONFIRM, 5 909 réellement FIT, 1 442 DEV.

**Conséquence** : la séparation FIT/DEV/CONFIRM revendiquée pour les
variantes n'était pas effective ; E01, E03, E04, E06, E07 (et le SAE de
référence, E05, E02 qui en dépendent) ont été produits sous l'ancien split
(archivé : `configs/post_stage/legacy_pre_parent_join_fix/`). Leurs chiffres
sont à traiter comme « split antérieur, contamination FIT↔CONFIRM » tant
qu'un rejeu (GPU, non lancé ici, autorisation requise) n'a pas été fait avec
`configs/post_stage/split_assignments.json` corrigé. Les checkpoints FIT
existants ne sont pas réutilisables comme « FIT-only ». Le sens des
conclusions n'est pas préjugé — seule leur validité méthodologique l'est.

## E01 — Représentations comparables (FIT→DEV)

**Fait, verdict négatif honnête.** Gain FULL−CORE **non établi** sur la
sonde 14 classes (axes d'augmentation, pas un test d'intention réelle —
terminologie à ne pas confondre, cf. correctif de ce nettoyage sur
`e05_results.md`) : +0,12pt, IC bootstrap [-0,08;+0,32] croise zéro (n=521
groupes DEV). FULL bat DENSE de +11,75pt (robuste) mais c'est surtout CORE.
Write-up : `docs/post_stage/e01_results.md`.

## E02 — Catalogue de features

**Fait.** 300 features (150 core/150 extra), 197/300=65,7% interprétables
(même run que R0, cohérence attendue). **Vérification humaine (60-100
features, plan §7) toujours en attente — nécessite Grégoire.**

## E03 — Retrieval par propriété (CONFIRM)

**Fait.** Premier gain établi de l'extension sur toute la campagne :
FULL−CORE = +25pt P@10, IC [+5,8;+45]. FULL perd contre DENSE et BM25 (SAE
pour expliquer, pas pour classer). P@10 mesuré sur des top-10 entièrement
jugés (pas besoin de juger tout le corpus, et le texte le dit explicitement).
`incident_collectif` quasi-nul pour toutes les méthodes — probable défaut de
la propriété, pas un résultat. Calibration humaine **non faite**. Write-up :
`docs/post_stage/e03_results.md`.

## E04 — Diffing structuré (urgence panique/calme)

**Fait.** 6/8 hypothèses confirmées dans le sens attendu, 1/8 confirmée mais
**inversée par rapport au signe de découverte** (`RESULTS_TESTS.md`-style
mise en garde : toujours lire le tableau par hypothèse, jamais le résumé
`verification_rate`/`coverage` seul), 1/8 ne survit pas FDR-BH. **Corrigé** :
le champ `percentage_difference` de `e04_diffing.py` recevait un log-odds
ratio au lieu d'un écart de fréquence borné [-1,1] (patch isolé et testé,
commit dédié — n'affecte que le signal de découverte envoyé au générateur
d'hypothèses, ne change rétroactivement aucun des 8 comptes CONFIRM
ci-dessus). Un nouveau run de diffing (autre contraste, ou rejeu de
panique/calme) bénéficiera du signal corrigé ; les résultats déjà gelés
ci-dessus restent valides tels quels. Audit humain **non fait**. Write-up :
`docs/post_stage/e04_results.md`.

## E05 — Stabilité inter-graines de l'extension

**Fait, bras random réellement exécuté et analysé** (pas seulement codé) :
2 seeds PCA + 2 seeds init aléatoire, indépendance vérifiée par cosinus
décodeur. Résultat : stabilité géométrique ne dépend pas du type d'init
(~80% des features supportées appariées), mais **le retrieval de thèmes par
groupe reste instable** (Jaccard@100 non significatif pour aucun groupe) —
"géométrie stable, retrieval variable". Write-up : `docs/post_stage/
e05_results.md`.

## E06 — Corrélations entre propriétés

**Fait — incohérence table/prose corrigée par ce nettoyage** (arbitrée
depuis `e06_correlations.json` : 4/8 paires établies, pas 5 comme l'ancienne
prose l'affirmait ; la table listait déjà les 4 bonnes lignes). Une seule
association vraiment actionnable (réfrigérateur × dysfonctionnement
électrique). Budget candidat CORE/FULL non équilibré par rapport au plan
(0 paire EXTRA-seule parmi les 8 gelées) — non corrigé, nécessiterait un
rerun. Audit humain **non fait**. Write-up : `docs/post_stage/
e06_results.md`.

## E07 — Clustering ciblé par axe

**Fait.** Contrôle d'axe fonctionne mécaniquement pour CORE/FULL (partitions
différentes par axe), pas du tout pour DENSE/TFIDF (clusters identiques quel
que soit l'axe demandé). Contenu sémantique aligné sur l'axe demandé
seulement pour `type_probleme` (accuracy 0,72-0,97) ; `action_attendue`/
`registre_urgence` restent dominés par le type de problème (accuracy jusqu'à
0,015 — à ne pas citer comme thème établi). Audit humain de paires **non
fait**. Write-up : `docs/post_stage/e07_results.md`.

## E08 — Mini-pilote analyste

**Outil prêt, séances humaines non tenues.** `e08_recette.md` l'indique
explicitement ; `e08_pilot_leads.json` est un template vide, aucune preuve
d'exécution dans le dépôt. Nécessite 2-3 participants réels (dont Grégoire),
~20min chacun — ne peut pas être simulé. Ce nettoyage a déplacé
l'emplacement d'écriture du formulaire hors du fichier suivi par Git (cf.
`docs/CLEANUP_MANIFEST.md`) ; les futures pistes n'apparaîtront donc plus
comme des diffs sur un fichier suivi.

## E09 — Run 100M (optionnel)

Non trouvé de trace d'exécution dans le dépôt au moment de ce nettoyage —
à confirmer avec Grégoire si une tentative a eu lieu ailleurs.

## Ce qui reste ouvert et nécessite Grégoire

- Vérification humaine E02 (60-100 features).
- Calibration humaine E03/E04 (jugements Qwen non recoupés par relecture).
- Audits humains E06 (paires) et E07 (clusters).
- Séances E08 (2-3 participants réels).
- Décision sur le rerun `percentage_difference` (E04) et sur le rééquilibrage
  CORE/FULL du budget candidat (E06) — tous deux non corrigés ici,
  volontairement, car une correction de formule/sélection doit être un
  patch isolé avec rejeu, pas un nettoyage de passation.
- Second sink mémoire (E00, `p1_all_doc_acts_ext_d*.pt`) — audit des 20
  scripts consommateurs avant correctif.
