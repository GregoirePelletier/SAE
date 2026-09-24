# Manifeste de nettoyage — passation post-soutenance

Portée : maintenance de passation sur la branche locale
`cleanup/post-soutenance-handoff`, HEAD au moment de ce nettoyage
`1844a4986c7c79d97a47b283e5b7b89fe39867f2` (l'audit de référence portait sur
`2f6fc2418e4ac084526dfc57b858431be426360b`, deux commits en amont — encore
ancêtre direct, pas de divergence). Aucune donnée, checkpoint, résultat ou
log supprimé. Aucun job Slurm soumis ou annulé. Aucun sous-module ni
dépendance mis à jour.

## Inventaire (avant modification, mesures bornées)

- Fichiers suivis par Git : 387. Poids de `.git` : 226 Mo
  (`git count-objects -vH` : 17 Mo lâches + 2,65 Mo empaquetés — le dépôt
  Git lui-même est petit ; ce n'est pas la source du poids du répertoire).
- Le plus gros contenu sous `git ls-files` (8,4 Go) est un artefact du
  répertoire de travail, pas Git : `external/interp_embed/.venv/` (un venv
  Python complet avec torch/CUDA, non ignoré par `.gitignore`, non suivi
  par Git non plus — `du` le compte car c'est un vrai dossier sur disque).
  **Non touché** : environnement potentiellement nécessaire pour ce
  sous-module, purge hors mandat de ce nettoyage.
- Artefacts locaux non suivis, non touchés (`.gitignore` déjà correct) :
  `local_data/` 3,9 To, `datasets/` 87 Go, `models/` 18 Go, `logs/` 2,5 Go,
  plusieurs dizaines de `results_v*/`/`results_post_stage_*/` (tailles
  individuelles non mesurées quand `du` dépassait 12s — stockage réseau,
  borné volontairement plutôt que scanné en entier).
- **Aucun de ces volumes n'a été déplacé, archivé ni supprimé** — les
  déplacer dans `archive/` n'aurait de toute façon rien allégé (ce sont des
  artefacts hors Git). Le poids réel du dépôt est déjà dans son historique
  Git (226 Mo, raisonnable) ; le poids réel du répertoire de travail est
  dans ces artefacts hors Git, hors mandat "suppression de données" de
  cette mission.

## Ce qui a été fait

### Lot 1 — corrections documentaires (commit `5779962`)

Aucune donnée/JSON de résultat réécrite, aucune formule/sélection/prompt
modifié. Uniquement des pointeurs vers des résultats déjà calculés et déjà
écrits ailleurs dans le dépôt :

- `README.md`, `CLAUDE.md`, `docs/evaluation_protocol.md` : la figure
  89,3% (§79) citée comme référence actuelle était supersédée deux fois
  (juge auto-référent, puis un biais de construction du négatif corrigé en
  §113-120) — remplacée par la référence déjà établie et déjà nommée comme
  telle dans `RESULTS_TESTS.md` (§119, R0, 65,7%/197-300). `CLAUDE.md`
  signalait aussi un effet d'échelle du modèle comme "massif et répliqué" —
  ce même §119 montre qu'il ne survit pas à la correction BH ; signalé en
  place, sans réécrire le paragraphe historique.
- `README.md` : `K_EXTRA` par défaut affiché (32) ne correspondait plus à
  `src/config.py:51` (5) — corrigé.
- `docs/post_stage/e05_results.md` : en-tête de tableau réutilisait le nom
  "sonde d'intention 14 classes" pour la sonde d'axes d'augmentation (déjà
  distinguée correctement dans E01) — même conflation que l'audit avait
  signalée pour E01, réapparue ailleurs — corrigé.
- `docs/post_stage/e06_results.md` : prose disait "5/8 établies", la table
  (et les IC bootstrap + FDR-BH du JSON `e06_correlations.json`, vérifiés
  directement) n'en soutiennent que 4/8 — table déjà correcte, seule la
  prose a été corrigée (arbitré depuis le JSON, comme demandé).
- `docs/post_stage/e04_results.md` : mise en garde documentée sur
  `percentage_difference` — reçoit un log-odds, pas un écart de fréquence,
  stade découverte uniquement. **Corrigé séparément dans le lot 5** (voir
  plus bas), pas mélangé à cette passe de documentation. Suppression au
  passage d'un artefact de syntaxe de liaison mémoire (`[[project_sae_qwen_
  judge_policy]]`) laissé par erreur dans ce fichier par une session
  antérieure — sans rapport avec le contenu scientifique.

### Lot 2 — portabilité (commit `b93e93f`)

- 141 fichiers `.slurm` (tout `slurm/` sauf `slurm/archive/`, vide) : chemin
  personnel `/home/h21486/SAE` remplacé par `${SAE_ROOT}` (variable
  introduite avec un défaut qui reproduit exactement l'ancien chemin absolu
  — aucun changement de comportement sur ce cluster tant que `SAE_ROOT`
  n'est pas positionnée). Directives `#SBATCH --output=`/`--error=` rendues
  relatives (Slurm ne développe pas les variables shell dans ces
  directives ; un chemin relatif se résout contre le répertoire de
  soumission, déjà toujours la racine du dépôt par convention documentée
  dans `slurm/README.md`). Aucune ligne de ressource (`--partition`,
  `--gres`, `--cpus-per-task`, `--mem`, `--time`, `--signal`) modifiée.
- `src/config.py` : `JUDGE_MODEL_ID` et `EMB_MODEL` étaient les deux seules
  valeurs par défaut en chemin absolu, alors que le reste du fichier
  (`SAVE_DIR`, `LOCAL_MAILS_PATH`, ...) utilise déjà une convention de
  chemin relatif — alignées sur cette convention existante, aucun nouveau
  mécanisme introduit.
- 7 scripts sous `scripts/` (`b1_stratified_mixte_qwen_rejudge.py`,
  `judge_model_separation_test.py`, `imdb_genre_diffing_test.py`,
  `email_interp_embed_encode_test.py`,
  `email_interp_embed_stats_from_cache.py`,
  `validate_interp_embed_tutorial.py`, `diagnose_email_zero_activations.py`)
  n'avaient aucune surcharge possible pour leurs chemins `interp_embed`/
  `local_data` — dérivent maintenant `SAE_ROOT` depuis `Path(__file__)`,
  surchargeable par la même variable d'environnement.
- `src/visualization/dashboard.py` : le formulaire de pistes E08 écrivait
  directement dans `docs/post_stage/e08_pilot_leads.json`, **suivi par
  Git** — chaque piste enregistrée serait apparue comme une modification
  d'un fichier suivi, avec risque d'écrasement silencieux si deux personnes
  lancent le dashboard en parallèle (lecture-modification-écriture non
  verrouillée). Écritures déplacées vers `local_data/dashboard_state/`
  (déjà ignoré par `.gitignore`, racine configurable via
  `SAE_DASHBOARD_STATE_DIR`), avec migration automatique, non destructive,
  à la première lecture : si des pistes existaient déjà dans l'ancien
  fichier suivi (ici vide, `[]`) et qu'aucun fichier local n'existe encore,
  elles sont recopiées une fois — l'ancien fichier n'est jamais supprimé ni
  écrasé. Limite de concurrence non résolue, documentée dans l'UI (caption
  du formulaire) plutôt que silencieusement laissée non dite.

### Lot 3 — correctif de validation (commit `119f182`)

Un test isolé (fixture synthétique, répertoire temporaire, jamais le vrai
`docs/post_stage/e08_pilot_leads.json`) reproduisant le scénario réel de ce
dépôt — template legacy vide `[]`, migration donc un no-op — a trouvé que
`_append_lead()` plantait (`FileNotFoundError`) à la toute première piste
enregistrée : `local_data/dashboard_state/` n'était créé que par le chemin
de migration, jamais par le chemin d'écriture direct. Corrigé (`os.makedirs`
ajouté dans `_append_lead`), revalidé sur le même scénario. Trouvé par ce
nettoyage, pas par l'audit d'origine.

### Non fait — identifié, laissé en l'état

- **Archivage des scripts d'audit ponctuels** (`scripts/*.py`, hors
  `scripts/post_stage/`) : un sondage sur 10 des ~54 scripts a montré 7
  encore référencés (recette Slurm active, README, `docs/architecture.md`)
  et seulement 3 sans aucune référence trouvée (`augmentation_lexical_
  leakage_audit.py`, `plot_d1_intent_bars.py`, `compare_to_frozen_
  benchmark.py`) — l'absence d'import ne prouvant pas l'absence d'usage,
  un balayage complet des ~44 scripts restants (imports, appels `__file__`,
  consommateurs externes) n'a pas été fait dans cette passe. Aucun
  déplacement fait sur la base d'un échantillon partiel. `docs/
  experiments.md` (index par question de recherche) et `docs/
  archived_runs_manifest.md` (répertoires `results_v*` supprimés → script
  de reproduction) existaient déjà et remplissent déjà le rôle d'index
  minimal demandé pour l'historique — non dupliqués.
- **Second sink mémoire E00** (`p1_all_doc_acts_ext_d*.pt`) : nécessite
  d'auditer 20 scripts consommateurs avant de toucher son schéma — hors
  mandat de ce nettoyage, déjà signalé dans `docs/post_stage/
  memory_diagnosis.md` et repris dans `docs/RESULTS_STATUS.md`.
- **Symlinks du cache d'extraction partagé**
  (`results_post_stage_e01_fit_1b_layer13_k5/cache/*`) pointent en chemin
  absolu vers `local_data/activation_cache/` — non corrigés (opération sur
  des données/caches, hors mandat), documentés dans `docs/HANDOVER.md`.

## Ce qui nécessite un accord explicite avant d'aller plus loin

- **Rééquilibrage du budget candidat CORE/FULL d'E06** — nécessiterait un
  nouveau rerun (coût GPU), pas fait ici.
- **Purge de `external/interp_embed/.venv/` (8,4 Go, non suivi, non
  ignoré)** — probablement un venv de développement local du sous-module,
  jamais confirmé indispensable ni obsolète dans cette passe.
- **Balayage complet des scripts d'audit restants** pour un archivage
  réellement complet (voir ci-dessus) — recommandé en une passe dédiée
  plutôt que dans ce nettoyage, avec un budget de vérification par script
  (imports, Slurm, tests, docs) suffisant pour ne pas se fier à un
  échantillon.
- **`report/RAPPORT_STAGE_ENTREPRISE.tex`** : `pytest tests/ -q` échoue sur
  `test_docs.py::test_check_docs_clean` (5 liens relatifs morts, ligne 949,
  labels `s2`-`s6`) — confirmé pré-existant à ce nettoyage (même échec
  reproduit en isolant les modifications de ce nettoyage via `git stash`).
  Provient de la modification en cours de ce fichier, non suivie et non
  touchée par ce nettoyage (déjà modifié avant cette passe de nettoyage) —
  à corriger avant de committer ce fichier.

## Lot 4 — réconciliation avec l'audit externe (`docs/01-04_*.md`)

Ces quatre documents (audit indépendant, même commit de référence
`2f6fc241` que cette mission) sont apparus dans `docs/` après les lots 1-3
ci-dessus, non suivis par Git. Confrontés au travail déjà fait :

**Déjà couvert par les lots 1-3, sans écart** : README (89,3%/K_EXTRA=32
périmés), écriture E08 hors Git, incohérence E06 "4 vs 5". L'audit et ce
nettoyage convergent indépendamment sur ces trois points.

**Réclamation de l'audit obsolète par rapport au HEAD actuel** : doc 01 §4.5
et doc 03 §8 décrivent le bras "init aléatoire" d'E05 (seeds 45/46) comme
non restitué ("aucune restitution chiffrée... n'apparaît dans les documents
de résultats lus"). `docs/post_stage/e05_results.md` contient déjà, au
moment de cette réconciliation, un tableau chiffré complet pour ce bras
(FVE, features mortes, rho_sae, sonde) et une analyse par type de paire —
l'audit travaillait sur une lecture antérieure à la rédaction complète de
ce document. Statut correct : "fait", pas "à récupérer".

**Nouveau, confirmé par ce nettoyage au-delà de ce que l'audit soupçonnait**
(doc 01 §3, doc 02 §4.3, doc 03 §3, checklist C) : la jointure positionnelle
variante→parent n'est pas seulement une limitation documentée mais un
désalignement systématique **mesuré** (`load_mails_tsv` : 3480 lignes ;
`load_and_clean_emails` : 3474 ; 6 positions exactes identifiées : 43, 707,
1243, 1344, 1366, 2149) — détail dans `docs/RESULTS_STATUS.md`, section
Corpus. L'audit avait deviné le bon ordre de grandeur ("six parents
nettoyés/dédupliqués", doc 02 §4.3) sans le vérifier par le code ; ce
nettoyage confirme le mécanisme exact et son ampleur (~99% des positions du
corpus concernées par un décalage). **Non corrigé ici** (scientifique, hors
mandat) : reste le point le plus déterminant avant de présenter E01/E03/
E04/E06/E07 comme reposant sur une séparation FIT/DEV/CONFIRM fiable.

**Précondition d'encodage CONFIRM documentée** (mission d'origine, section
7 ; doc 02 §6.5) : ajoutée à `docs/HANDOVER.md`, avec citation exacte du
garde-fou `RuntimeError` de `saev5.py:965-971` (SAVE_DIR réutilisé avec une
clé de cache différente) — absente des lots 1-3, comblée dans ce lot.

**Empreintes SHA-256 des petits artefacts** (checklist D, mission section
8) : calculées pour les 12 JSON du run canonique, ajoutées à
`docs/HANDOVER.md` — absentes des lots 1-3, comblées dans ce lot.

## Lot 5 — correctif scientifique isolé : `percentage_difference` (E04)

L'audit classe ce champ "correction bloquante" (checklist E) ; le lot 1
l'avait documenté sans le corriger, par prudence (isolement des correctifs
scientifiques demandé par la mission d'origine). Confirmation explicite
reçue : correctif fait dans un commit séparé des lots de nettoyage.

- `scripts/post_stage/e04_diffing.py` : la construction de la liste
  `features` (auparavant en ligne dans `main()`) extraite dans
  `build_feature_diff_blocks(selected)`, seule sa colonne source change —
  `row["freq_diff"]` (déjà calculée par `select_top_diff_features_by_
  frequency`, `freq_A - freq_B`, bornée [-1,1] par construction) au lieu de
  `row["log_odds_ratio"]` (non borné). Extraction justifiée par le besoin
  d'un test sans GPU/juge (règle CLAUDE.md sur l'extraction ciblée), pas une
  réécriture.
- Test de non-régression :
  `tests/post_stage/test_e04_diffing_feature_blocks.py` (3 tests : valeur
  correcte à `freq_A=0.8`/`freq_B=0.2` → `+0.6` et non le log-odds ; le
  signe s'inverse si A/B sont échangés ; la valeur reste dans [-1,1] même
  pour un écart extrême où le log-odds ne le serait pas).
- Impact : ne touche que le signal de découverte transmis au générateur
  d'hypothèses (stade FIT, avant gel). Ne modifie aucun JSON déjà écrit —
  les 8 comptes de vérification CONFIRM déjà mesurés dans `e04_diffing.json`
  restent valides tels quels (jugements déjà faits sur les hypothèses déjà
  gelées, pas recalculés). **Nécessité de rejeu** : uniquement si une
  nouvelle campagne de diffing (nouveau contraste, ou reprise de
  panique/calme) est lancée — le signal de découverte corrigé peut alors
  produire des hypothèses différentes de celles de ce run.
- `docs/post_stage/e04_results.md` et `docs/RESULTS_STATUS.md` mis à jour
  pour refléter le correctif et son périmètre exact.

## Reste ouvert, non résolu par ce nettoyage

Nécessite une décision de Grégoire, pas une action de nettoyage :

- Dépendance de parents partagés entre groupes A/B dans l'échantillonnage
  E04 (doc 01 §4.4, doc 03 §5) et déduplication par parent du top-10 dans
  E03 (doc 01 §4.3, doc 02 §4.2) : découvertes par l'audit, non vérifiées
  ni corrigées par ce nettoyage.
- Révisions exactes des poids (Gemma, GemmaScope, bge-m3, juge Qwen) non
  consignées dans un manifeste (checklist B) — non fait.
- Le paquet d'audit référence `SOURCES.md` (codes S01-S27) et
  `outils/preflight_passation_sae.py` : **ni l'un ni l'autre n'existe dans
  le dépôt** au moment de cette réconciliation — liens internes cassés du
  paquet de passation lui-même, à signaler à la source de ces documents.

## Lot 6 — correction de la jointure parent (variantes → parent)

Correction demandée explicitement après la découverte du lot 4.

- Code (commit `9397895`) : `load_and_clean_emails(return_positions=True)` ;
  `build_email_train_test_corpus` et `dataset_contract.py` traduisent
  `parent_id` via la position d'origine au lieu d'un `enumerate()` décalé.
  Deux tests de non-régression (ligne « Objet seul » reproduisant l'écart).
  Les 5 scripts `replicate_load_and_clean_emails_with_index` étaient déjà
  corrects (dupliqués, non factorisés ici).
- Données : `configs/post_stage/{corpus_manifest,split_assignments}.json`
  régénérés (même graine, parents identiques, 70 variantes auparavant
  « non rattachées » désormais rattachées). Anciennes versions conservées
  dans `configs/post_stage/legacy_pre_parent_join_fix/` (manifeste + assignations gzip).
- Impact et nécessité de rejeu : `docs/RESULTS_STATUS.md`, section Corpus
  (54,7 % des variantes déplacées ; 24,3 % de l'ancien FIT était CONFIRM).
  Aucun rejeu lancé (GPU) ; aucun JSON de résultat réécrit.
- `parent_sha1` reste absent du `augmented_mails.jsonl` gelé (généré avant
  son ajout) : la jointure reste positionnelle, désormais correcte ; la
  régénérer avec `parent_sha1` supprimerait cette dépendance.

## Lot 7 — passe de finition (statuts, démarrage, rangement)

Non commité à la rédaction. Aucun job lancé, aucune donnée touchée. Statut du rejeu `_v2` : partiel
et non validé (`docs/RESULTS_STATUS.md`).

| Ancien chemin | Nouveau chemin / action | Raison |
|---|---|---|
| `docs/PDF_APPENDICES_EXTRACT.md` | `docs/archive/references/PDF_APPENDICES_EXTRACT.md` (contenu inchangé) | extraction de travail ; références mises à jour (`src/analysis`, 2 scripts, 2 recettes, `check_docs.py`, `INTERP_EMBED_COVERAGE.md`) |
| `CLAUDE.md` (long) | `docs/archive/CLAUDE_long_2026-09.md` ; `CLAUDE.md` racine réduit à ~25 lignes | informations uniques conservées (diagnostics, seeds) ; pièges essentiels repris dans `HANDOVER.md` |
| `scripts/augmentation_rejection_length_bias_test.py` | `scripts/archive/` | audit ponctuel, aucune référence active |
| `scripts/seed_label_overlap_r0_test.py` | `scripts/archive/` | idem (`RESULTS_TESTS.md`, append-only, cite l'ancien chemin) |
| `docs/archive/audits/AUDIT_SAE_2026-08.md` | conservé à la racine | cité par `report/03_*.md` (protégé) et `RESULTS_TESTS.md` |
| `docs/INTERP_EMBED_COVERAGE.md` | conservé | référence de méthode, citée par `src/analysis/hypothesis_verifier.py` |
| `scripts/dictionary_width_quality_audit.py`, `compare_to_frozen_benchmark.py`, `plot_*.py` | conservés | chemins calculés depuis `__file__` / figures du rapport |

Contenu : bandeau de statut (README, HANDOVER, E01-E07) ; formulations « gain établi / jamais vu /
restent valides » corrigées sans toucher aux valeurs ; E00, E02≠R0, E03 (P@10, `incident_collectif`), E04
(pas de rejeu), E05, E06, E09 précisés ; HANDOVER : parcours consultation / reprise ; `slurm/README.md` :
lancement, `RUN_SUFFIX`, juge ; dashboard : bandeau par run et « sonde des axes d'augmentation » (E01).
Hook `.claude/settings.json` inchangé (relance de la suite après édition `.py`).

## Lot 8 — revue de passation finale

| Ancien chemin | Nouveau chemin / action | Raison |
|---|---|---|
| `AUDIT_SAE_2026-08.md` (racine) | `docs/archive/audits/` | audit historique ; ~100 références de commentaires/docs mises à jour ; les `.tex` du rapport le citent par son nom (non modifiés) |
| `docs/audit_2026_08_*.json` | `docs/archive/audits/` | sorties d'audit ; le dashboard les lit désormais à cet endroit (l'ancien motif ne les trouvait pas) |
| `Plan_execution_SAE_15_jours_Claude_Code.md` (non suivi) | `docs/post_stage/PLAN_E00-E09.md` (suivi) | cité « plan §N » dans toute la documentation : dépendance réelle, jamais versionnée |
| `docs/01-04_*.md` (non suivis, audit externe) | `docs/archive/audit_externe_2026-09/` (suivis, contenu inchangé) | instantané du 22 septembre, en partie dépassé ; index dans `docs/archive/README.md` |
| `.github/agents/ml-engineer-cpu-test.agent.md` | retiré (récupérable dans Git) | prompt d'agent générique, sans information propre au projet |
| `.claude/settings.json` | plugin personnel retiré, hook `pytest` conservé | outillage personnel |
| `slurm/**/*.slurm` (141) | défaut `SAE_ROOT` = `$SLURM_SUBMIT_DIR` au lieu d'un chemin personnel | portabilité (même comportement si `sbatch` est lancé depuis la racine) |
| `README.md` | réécrit | besoin, données, statut, installation, consultation, recalcul, tests, carte, hors dépôt |

Aussi : `.env.example` (chemins des mails), `docs/architecture.md` (défauts `K_EXTRA`, `EMB_MODEL`),
`docs/experiments.md` (index marqué historique), `docs/HANDOVER.md` (carte des documents,
`SAE_ROOT`), `docs/RESULTS_STATUS.md` (liste « reste à faire »), `scripts/check_docs.py`
(`docs/archive/` et le plan externe exclus du contrôle éditorial).

Non traités, décision de l'auteur : modification locale non commitée de
`report/RAPPORT_STAGE_ENTREPRISE.tex` (liens `s2`-`s6` morts ligne 949), coexistence de
`RAPPORT_STAGE_ENTREPRISE.tex` et `Rapport_stage_EDF_relecture.tex`, pointeur du sous-module
`external/sae-lens` déplacé localement, fichiers non suivis `archive/dual_pipeline_sae.py`,
`Prompt_finition_rapide_passation_SAE.md` et la capture d'écran de `report/`.

## Lot 9 — retrait du rapport de stage

À la demande de l'auteur (version finale remise hors dépôt) : `report/` retiré du suivi
(chapitres `.md`, trois `.tex`, figures ; récupérable dans l'historique jusqu'au commit
`e925d01`). La modification locale non commitée de `RAPPORT_STAGE_ENTREPRISE.tex`, les PDF, la
capture d'écran et `report/dist/` ont été sauvegardés hors du dépôt avant retrait. Retirés avec
lui : `scripts/build_report.py` et `slurm/validation/run_build_report.slurm` (assemblage du
rapport) ; `scripts/plot_*.py` passés dans `scripts/archive/` (figures du rapport). `report/`
retiré de `scripts/check_docs.py` et de `.gitignore`. Les mentions historiques du rapport dans
`RESULTS_TESTS.md` et des commentaires de code sont laissées telles quelles.
`Prompt_finition_rapide_passation_SAE.md` (non suivi) : non livré.
