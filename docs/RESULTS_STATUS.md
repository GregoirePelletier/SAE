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

**Réserve la plus importante de ce document, à lire avant tout le reste** :
la jointure variante→parent du split FIT/DEV/CONFIRM présente un
désalignement positionnel systématique confirmé (pas seulement suspecté) —
section "Corpus" ci-dessous. Tant qu'elle n'est pas résolue, traiter
l'indépendance FIT/DEV/CONFIRM comme non garantie pour les variantes
augmentées (les splits par PARENT, eux, restent corrects — c'est le
rattachement des variantes à leur parent qui est en cause).

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

## Corpus — freeze FIT/DEV/CONFIRM — **désaccord positionnel confirmé, pas seulement documenté**

`configs/post_stage/{corpus.yaml,corpus_manifest.json,split_assignments.json}`
(3474 parents → 2084/521/869). Le manifeste affiche `"method":
"positional_via_parent_id"`, `"positional_join_fallback": true`,
`"n_variants_unmatched": 70` — jusqu'ici documenté comme une limitation
acceptée. Une vérification bornée (CPU, texte déjà en local, aucun modèle
chargé, ~40s) faite pendant ce nettoyage va plus loin et **confirme un
désalignement systématique, pas seulement un risque théorique** :

- `src/data/dataset.py::load_mails_tsv` (utilisé par les deux côtés de la
  jointure) renvoie **3480** lignes après son propre filtre
  (`min_chars=30` + `drop_duplicates("text")`). `scripts/run_augmentation.py`
  numérote `doc_id` directement sur cette sortie (`.reset_index()`), donc
  `parent_id` dans `augmented_mails.jsonl` vit dans cet espace à 3480
  positions.
- `src/data/preparation.py::load_and_clean_emails` (utilisé par
  `dataset_contract.py` pour construire `pos_to_hash`) part du même
  `load_mails_tsv`, puis applique un filtre **supplémentaire**
  (`strip_leading_objet_line` + suppression d'un motif `[{"start"...}]`,
  ligne supprimée si le résultat est vide) — sortie à **3474** lignes.
- Les deux comptes ne coïncident pas : **6 lignes** (positions 43, 707,
  1243, 1344, 1366, 2149 dans l'espace à 3480) deviennent vides sous ce
  filtre supplémentaire et disparaissent de `real_hashes`, mais restent
  comptées dans l'espace `doc_id` de l'augmentation.
- `pos_to_hash = {i: h for i, h in enumerate(real_hashes)}` (`dataset_
  contract.py`) énumère donc un espace **plus court de 6** que celui dans
  lequel `parent_id` a été écrit. Conséquence directement calculable :
  pour tout `parent_id >= 44` (soit ~99% des positions du corpus, le
  premier écart tombant à la position 43/3480), `pos_to_hash.get(int(
  parent_id))` résout un index décalé de 1 à 6 rangs selon combien des 6
  positions le précèdent — **pas une erreur qui se voit** (le lookup
  réussit, avec un hash de parent différent de celui qui a réellement
  généré la variante), silencieuse par construction.

**Ce que ça n'établit pas** : ce n'est pas la preuve qu'une variante a
changé de split (le décalage peut aussi bien retomber dans le même split
que le vrai parent) — seulement que le mécanisme de rattachement n'est,
pour l'écrasante majorité des variantes, pas celui que le code croit
utiliser. `n_variants_unmatched: 70` (des `parent_id` complètement hors
plage ou `None`) est probablement sous-compté du vrai problème : un
`parent_id` décalé qui retombe sur un index valide ne remonte **aucune**
erreur.

**Ne pas regeler les splits pour faire disparaître cet avertissement.**
Correctif possible mais non fait ici (scientifique, hors mandat de ce
nettoyage) : faire écrire `parent_id` par `run_augmentation.py` dans le
même espace de positions que `load_and_clean_emails` (ou, mieux, s'appuyer
sur `parent_sha1` — déjà calculé et écrit par le code actuel de
`src/data/augmentation.py`, mais absent du fichier `augmented_mails.jsonl`
actuellement gelé, généré par une version antérieure du pipeline qui ne
l'écrivait pas encore) puis réévaluer si les checkpoints/splits actuels
restent valides ou doivent être régénérés. Nécessite Grégoire — c'est le
point P0 le plus déterminant avant de présenter les résultats E01/E03/E04/
E06/E07 comme reposant sur une séparation FIT/DEV/CONFIRM fiable.

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
