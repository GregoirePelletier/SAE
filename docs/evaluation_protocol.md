# Protocole d'évaluation complet — "tester l'intégralité du repo"

Objectif : évaluer, sous des **conditions fixées et contrôlées** (un seul LLM, un seul
jeu d'hyperparamètres, un seul embedding par rôle), l'ensemble des capacités du
pipeline sur le corpus EDF (mails originaux + augmentés), et comparer toutes les méthodes/alternatives
disponibles entre elles. Ce document sert de référence reproductible : les mêmes
étapes, avec d'autres valeurs dans la section "Conditions fixées", permettront plus
tard une comparaison multi-modèles (explicitement mise de côté pour l'instant, cf.
critères de décision plus bas).

## Conditions fixées de cette passe

| Paramètre | Valeur | Justification |
|---|---|---|
| LLM (extraction + juge) | `google/gemma-3-12b-it` | Cible de production du projet, seul LLM utilisé dans toutes les expériences validées à ce jour. |
| SAE préentraîné (Pipeline 1, "core") | GemmaScope-2 `layer_24_width_16k_l0_medium` | Couverture Neuronpedia la plus dense en proportion (`report/01_etat_de_lart.md`) — **mais le balayage de couche fait après ce choix (`RESULTS_TESTS.md` §51) trouve la couche 31 significativement supérieure à la couche 24 (z=2.20) ; le choix de couche n'a pas été révisé depuis.** |
| Extension SAE (Pipeline 1) | `D_EXTRA=1024`, `K_EXTRA=32` | Valeurs par défaut, non ré-optimisées dans cette passe (piste de suite, `report/04`). |
| Budget de tokens extension | `N_TOKENS_EXTRA_TRAIN=500000` | Validé non-limitant par ablation (100k/500k/2M statistiquement indistinguables, `RESULTS_TESTS.md` §12) — **mais cette ablation reste elle-même 50 à 100× en dessous du seuil de convergence documenté par la littérature (SAE Boost, `RESULTS_TESTS.md` §18.3) : sa conclusion ne s'extrapole pas au régime production sans le tester directement. Confirmation à 200M tokens (§23.5) toujours `PENDING`.** |
| Corpus principal (entraînement) | Emails originaux + augmentés (`local_data/emails/`), corpus generic energy/sports/support réduit à un rôle secondaire post-hoc | `RESULTS_TESTS.md` §12 : c'est le facteur qui a le plus d'effet sur l'interprétabilité. |
| Embedding Pipeline 2 (backbone `PhraseLevelSAE`) | `F2LLM-v2-330M` (au lieu de -80M) | Cf. §"Comparaison des embeddings" ci-dessous. |
| Embedding pour similarité de labels (retrieval/clustering/corrélations) | `bge-m3` | Seul modèle validé fiable sur les deux requêtes de test (`RESULTS_TESTS.md` §15.2) -- F2LLM y donnait des résultats sans rapport sur une des deux. |
| Nombre de features jugées (juge LLM) | `N_FEATURES_TO_LABEL=150` | Puissance statistique correcte (IC95% ≈ ±8 points), cf. §12. |
| Graine aléatoire | `SEED=42` partout | Reproductibilité. |

Ces valeurs correspondent au run déjà produit `results_v10_emails_main/` (Pipeline 1 +
Pipeline 2 avec F2LLM-80M) complété par `results_v10_p2_f2llm330m/` (Pipeline 2 avec
F2LLM-330M, cf. tableau ci-dessous) -- **aucun nouveau run Pipeline 1 n'est nécessaire**
pour cette passe, seules les analyses en aval (déjà listées) restaient à produire.

## Inventaire des méthodes à tester, et de leurs alternatives comparées

Pour chaque capacité : la commande pour la (re)produire, où lire le résultat, et à
quoi la comparer.

| # | Capacité | Commande | Résultat | Comparaison |
|---|---|---|---|---|
| 1 | Reconstruction SAE (P1 core+ext, P2) | *(déjà produit)* `slurm/pipeline_runs/run_sae_v10_emails.slurm` | `results_v10_emails_main/results.json` | P1 vs P2 (NMSE/L0/dead%/ρ_SAE/silhouette côte à côte, table "Bilan comparatif") |
| 2 | Cohérence métrique FVE avec SAELens | `scripts/saelens_numeric_comparison.py` | `.../cache/saelens_numeric_comparison.json` | Notre formule vs 2 formules natives sae_lens (§13/`docs/references.md`) |
| 3 | Labellisation core | *(déjà produit, cache Neuronpedia)* | `p1_top_core_features.json` | vs labellisation extension (juge) — couverture/densité différente par construction |
| 4 | Labellisation extension (gate odd-one-out) | *(déjà produit)* | `.../cache/p1_judge_labels_extended.json` | vs labellisation contrastive directe (méthode #5) |
| 5 | Labellisation extension (contrastive directe, sans gate) | `scripts/contrastive_labeling_test.py` | `.../cache/p1_contrastive_labels.json` | vs #4 : taux de récupération sur les features rejetées par le gate (`RESULTS_TESTS.md` §15.4) |
| 6 | Robustesse du jugement (ordre des exemples) | `scripts/judge_robustness_check.py` | `.../cache/p1_judge_robustness.json` | single-shot vs vote majoritaire (5 répétitions), §13.1 |
| 7 | Séparabilité des axes d'augmentation (synthétique) | *(déjà produit, dans results.json)* | `clf_acc_email_axes` (P1 et P2) | axes synthétiques vs intentions réelles (méthode #8) |
| 8 | Détection d'urgence/intention (réelle) | `scripts/intent_urgency_probe.py` | `.../cache/intent_urgency_probe_results.json` | vs baseline classe majoritaire ; vs #7 (label faible réel vs perturbation simulée) |
| 9 | **Fidélité de l'explication document-level (nouveau)** | `scripts/explanation_fidelity_test.py` | `.../cache/explanation_fidelity_results.json` | ablation top-K vs random-K vs bottom-K |
| 10 | **Plausibilité de l'explication document-level (nouveau)** | `scripts/explanation_plausibility_test.py` (GPU, juge) | `.../cache/explanation_plausibility_results.json` | choix forcé réel vs décoy aléatoire, vs hasard (50%) |
| 11 | Retrieval par propriétés / clustering ciblé | *(dans results.json, section P1)* — `select_latents_by_similarity` | `p1_diff_energy_sports.csv`, sortie console "Task 3/4" | bge-m3 vs F2LLM vs (ancien) matching substring, §15.1-15.2 |
| 12 | Corrélations "intéressantes" (NPMI + dissimilarité) | *(dans results.json, section P1)* — `find_interesting_pairs` | `p1_interesting_correlations.json` | vs matrice NPMI brute seule (`p1_npmi.pt`, sans filtre) |
| 13 | Diffing cross-domaine (SAE natif, mails originaux vs augmentés) | *(déjà produit)* `slurm/baseline_diffing/run_baseline_full_v2.slurm` | `results_v11_baseline_objetfix/cache_baseline_full/diff_*.csv` | avant/après fix biais "Objet :" (§14.1) ; vs diffing energy/sports (P1, générique) |
| 14 | **Embedding backbone Pipeline 2 : F2LLM-80M vs F2LLM-330M (nouveau)** | `slurm/pipeline_runs/run_sae_v10_p2_f2llm330m.slurm` | `results_v10_p2_f2llm330m/results.json` | comparer NMSE/L0/`clf_acc_email_axes` contre `results_v10_emails_main/results.json` (section P2) |
| 15 | Retrieval BM25 sur vocabulaire latent (Latent Terms) | `scripts/retrieval_demo.py` / `src/sae/retrieval/latent_terms.py --mails ...` | résultats console | non comparé formellement à ce jour à `select_latents_by_similarity` (piste de suite) |
| 16 | Robustesse au biais de formatage du corpus augmenté | *(déjà produit)* fix `load_augmented` + rerun #13 | cf. §14.1 | avant/après, par axe/niveau |

## Limites connues de l'extrapolation à 500K tokens

Toutes les ablations méthodologiques listées ci-dessus (et la quasi-totalité de
`RESULTS_TESTS.md`) sont produites au budget `N_TOKENS_EXTRA_TRAIN=500000`. C'est une
contrainte matérielle/temps assumée, pas un choix neutre : `RESULTS_TESTS.md` §18.3
situe ce régime 50 à 100× en dessous du seuil où la littérature (SAE Boost) observe un
effet de volume sur la performance en domaine général. Les conclusions qui en dépendent
ne sont valables que dans ce régime réduit, jusqu'à confirmation à plus grande échelle :

- **§12/§18.3** — budget de tokens non-limitant : conclusion directement citée en
  "Conditions fixées" ci-dessus, non extrapolable sans la confirmation à 200M tokens
  (§23.5, job 41658, toujours `PENDING`, bloqué par un incident de maintenance
  cluster). Le seul signal positif observé en cours de route (+8,7 pt à 25M tokens,
  §23.4) ne réplique pas sur deux graines supplémentaires (§56) — traité comme du
  bruit d'échantillonnage, pas comme un effet de volume confirmé.
- **§28** — dose-réponse taille de modèle (1b/4b/12b, n=150×3) : facteur limitant à
  petite échelle explicitement signalé par la section elle-même, charge la décision
  d'architecture (Gemma-3-12b) sur seulement 3 points de mesure.
- **§33** — un seul jeu de labels connus (14 axes), un seul corpus, un seul SAE :
  conclusion non généralisée au-delà de cette configuration précise.
- **§51** — le balayage de couche qui trouve la couche 31 meilleure que la couche 24
  (configuration actuelle, cf. tableau ci-dessus) n'a pas été suivi d'un rerun de
  référence à la couche 31 pour vérifier si l'écart tient à plus grande échelle.

Prioriser un rerun à plus grande échelle sur la chaîne §12→§18.3→§23 (c'est la
décision la plus citée ailleurs dans le projet) avant §28 ou §33.

## Décalage avec `RESULTS_TESTS.md`

Ce document n'a pas été mis à jour en substance depuis sa création (liée à §16) — un
seul commit ultérieur, purement rédactionnel. Tout `RESULTS_TESTS.md` §17 à §72
(largement la série `docs/AUDIT_2026-08.md`, §57-72) est donc absent d'ici, y compris
des résultats qui contredisent ou nuancent des lignes déjà présentes dans les tableaux
ci-dessus : le balayage de couche (§51, ci-dessus), le point de hook mlp_out > attn_out
(§53), et surtout §60 (`INTENT_KEYWORDS_FR` sous-comptait sévèrement les labels réels
d'intention — "résiliation" passe de 1 à 864 mails détectés une fois le bug de regex
réellement corrigé, ×1,15 à ×3,37 sur les autres intentions). Ce correctif touche
directement la ligne #8 "Détection d'urgence/intention (réelle)" de l'inventaire
ci-dessus : toute lecture de `intent_urgency_probe_results.json` produite avant ce
correctif utilise des labels de référence faux, pas seulement bruités.

## Comment lire les résultats consolidés

`scripts/consolidate_evaluation_report.py` (nouveau) parcourt un `SAVE_DIR` donné et
assemble automatiquement tous les artefacts ci-dessus (quand présents) en un seul
rapport markdown + un résumé JSON, pour éviter d'ouvrir 15 fichiers séparément.
Le dashboard (`src/visualization/dashboard.py`) expose ce même résumé dans un onglet
dédié ("Vue d'ensemble complète").

```bash
PYTHONPATH=. .venv/bin/python scripts/consolidate_evaluation_report.py results_v10_emails_main
.venv/bin/python -m streamlit run src/visualization/dashboard.py
```

## Critères de décision avant de passer à la comparaison multi-modèles

Le passage à une évaluation sur plusieurs modèles/conditions (mise de côté pour
l'instant) est conditionné à l'absence de "problème majeur" sur cette passe
unique. Signaux à surveiller dans le rapport consolidé :

- Fidélité (#9) : le ratio chute top-K / chute random-K doit rester très supérieur à 1
  (déjà confirmé très largement le cas, cf. `RESULTS_TESTS.md`).
- Plausibilité (#10) : le taux de succès du choix forcé doit être significativement
  au-dessus de 50% (sinon l'explication produite n'a pas de valeur perçue au-delà du
  hasard, un problème majeur).
- Aucune régression du run F2LLM-330M (#14) par rapport à -80M sur NMSE/L0 (un embedding
  plus gros qui dégrade la reconstruction serait un signal à investiguer avant
  d'étendre le protocole).
- Cohérence qualitative dans le dashboard (UMAP, features, diffing) — inspection
  visuelle finale requise avant d'élargir la comparaison.
