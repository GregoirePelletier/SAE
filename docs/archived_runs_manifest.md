# Runs archivés (données brutes supprimées)

Répertoires `results_v*/` supprimés du disque (nettoyage 2026-08-23) — tous
produits avant le correctif B.3/B.5/B.6/B.11 du protocole juge (commit
`c1d73ef`, 2026-08-21) ou avant, donc non comparables aux chiffres
d'interprétabilité produits depuis. Les conclusions numériques restent dans
`RESULTS_TESTS.md` (append-only, jamais purgé) ; seuls les artefacts lourds
(activations, checkpoints, fragments token-level) sont supprimés. Chaque
ligne pointe vers le script qui reproduit le même type de run — relancer le
script régénère un résultat directement comparable au code actuel, pas une
reproduction bit-exacte de l'ancien chiffre.

| Répertoire supprimé | Type d'ablation | Script pour refaire ce type de run |
|---|---|---|
| `results_v9_full`, `results_v9_test` | Runs de bring-up initiaux, pipeline complet — scripts (`run_sae_full.slurm`, `run_sae.slurm`) supprimés aussi (nettoyage slurm), strictement supersédés par `run_sae_v10_emails.slurm` | `slurm/pipeline_runs/run_sae_v10_emails.slurm` |
| `results_v9_confirmatory_n150` | Confirmation domaine n=150 (C1) | `slurm/pipeline_runs/run_c1_confirmatory_domain_n150.slurm` |
| `results_v10_ablation_tok100k`, `results_v10_ablation_tok2M` | Ablation budget de tokens extension (100k, 2M) | `run_sae_v10_ablation_tok100k.slurm`, `run_sae_v10_ablation_tok2M.slurm` |
| `results_v10_p2_bgem3`, `results_v10_p2_f2llm160m`, `results_v10_p2_f2llm330m` | Comparaison embedding backbone Pipeline 2 | `run_sae_v10_p2_bgem3.slurm`, `run_sae_v10_p2_f2llm160m.slurm`, `run_sae_v10_p2_f2llm330m.slurm` |
| `results_v10_p2_mrl64/128/640` | Matryoshka SAE (Pipeline 2) — **idée abandonnée au profit de SAE Boost**, scripts supprimés aussi (nettoyage slurm), ne pas reproduire | — (abandonné, cf. `git log` si besoin de retrouver la config) |
| `results_v11_baseline_objetfix` | Diffing cross-domaine, baseline avant/après fix biais "Objet :" — versions pré-fix (`run_baseline_full.slurm`, `run_baseline.slurm`, ciblaient `results_v9_test/`) supprimées aussi (nettoyage slurm), strictement supersédées | `slurm/baseline_diffing/run_baseline_full_v2.slurm` |
| `results_v12_ablation_capacity_extra`, `results_v12_ablation_epochs_only`, `results_v12_ablation_width65k_only` | Ablations capacité extension (D_EXTRA, EPOCHS_EXTRA, largeur 65k) | `run_ablation_capacity_extra.slurm`, `run_ablation_epochs_only.slurm`, `run_ablation_width65k_only.slurm` |
| `results_v12_sanity_frozen_decoder` | Sanity check décodeur figé (Korznikov et al.) | `slurm/analysis/run_sanity_check_frozen_decoder.slurm` |
| `results_v12_scaled_65k` | Run à l'échelle largeur 65k | `run_sae_v12_scaled.slurm` |
| `results_v13_ablation_attn_out`, `results_v13_ablation_mlp_out` | Ablation hook-point (attn_out, mlp_out vs resid_post) | `run_ablation_attn_out.slurm`, `run_ablation_mlp_out.slurm` |
| `results_v13_ablation_d_extra2048_only` | Ablation D_EXTRA=2048 | `run_ablation_d_extra2048_only.slurm` |
| `results_v13_ablation_k_extra5`, `_seed7`, `_seed99` | Ablation K_EXTRA=5 (setup classique du papier) + variance inter-graine | `run_ablation_k_extra5.slurm`, `_seed7.slurm`, `_seed99.slurm` |
| `results_v13_ablation_layer12/31/41` | Balayage layer resid_post (12/31/41 vs 24 défaut) | `run_ablation_layer12.slurm`, `_layer31.slurm`, `_layer41.slurm` |
| `results_v13_ablation_model_scale_1b/4b` | Ablation taille du modèle extracteur/juge — anciens scripts supprimés aussi (nettoyage slurm), directement remplacés par le setup classique actuel (K_EXTRA=5, n=150), sweep étendu à 27B (§ ci-dessous) | `run_ablation_classic_setup_k5_25m_model_scale_1b.slurm`, `_4b.slurm` |
| `results_v13_ablation_seed123` | Variance inter-graine (seed=123) | `run_ablation_seed_variance.slurm` |
| `results_v13_ablation_volume25m`, `_seed7`, `_seed99` | Ablation volume 25M tokens (première génération, variance inter-graine) — **remplacé par le run 2026-08-23** (`results_v13c_ablation_volume25m_h100/`, RESULTS_TESTS.md) | `run_ablation_volume_25m_seed7.slurm`, `_seed99.slurm` (script principal remplacé par `run_ablation_volume_25m.slurm`/`_h100.slurm`) |
| `results_v13_ablation_width262k_only` | Ablation largeur dictionnaire 262k | `run_ablation_width262k_only.slurm` |
| `results_v14_validation_25m_layer24` | Run de validation 25M tokens, layer 24 (première génération) | `run_validation_25m_layer24.slurm` |
| `results_v15` à `v23` (`validation_100k_layer24_postfixes`, `v2` à `v9`) | Checkpoints de débogage intermédiaires, chacun corrigeant un bug trouvé par le précédent (reencode, extraction filler, batch size, TF32, sharding, corrections data-science, originaux seuls non contrôlé) — **tous strictement supersédés** par le run suivant de la même séquence, aucun n'est un point de comparaison scientifique en soi. Scripts supprimés aussi (nettoyage slurm) : ni le bug ni le correctif ne se "rejoue", ils sont dans `git log`/`RESULTS_TESTS.md`. | — (historique de débogage, `git log --all -- 'slurm/pipeline_runs/run_validation_100k_layer24_v*'` si besoin) |
| `results_v14c_ablation_volume5m_h100` | Doublon de course (a100/h100) de l'ablation volume 5M, annulé une fois le doublon a100 terminé (résultat déjà obtenu) | `run_ablation_volume_5m_h100.slurm` |

## Runs conservés (post-correctif B.3/B.5/B.6/B.11, 2026-08-21)

`results_v10_emails_main/` (référence principale), `results_v13c_ablation_volume25m_h100/`
et `results_v14b_ablation_volume5m/` (calibration volume), `results_v24` à `v26`
(validation post-correctif négatifs, B.1, confirmation n=150) — actifs, cités
directement dans `RESULTS_TESTS.md` §77-§81. `results_v28_validation_layer31_originals_1M2_mislabeled_100M/`
conservé mais mineur (nom explicite : réservoir non rempli, 1,23M tokens
réels sur les 100M ciblés, corpus trop petit — cf. correctif dans
`run_validation_100M_layer31_h100.slurm`).

## Sweep taille de modèle (setup classique K_EXTRA=5, D_EXTRA=1024, 25M tokens, n=150)

Remplace intégralement l'ancien sweep 1B/4B/12B (archivé ci-dessus) et l'étend
à 27B : `run_ablation_classic_setup_k5_25m_model_scale_1b.slurm`, `_4b.slurm`,
`run_ablation_classic_setup_k5_25m_layer31.slurm` (12B, référence), et
`run_ablation_classic_setup_k5_25m_layer40_27b.slurm` (27B, layer 40 = ~2/3
profondeur sur 62 couches, même logique que layer 31/48 pour 12B — cf.
`src/config.py::_PRESETS["27b"]`). Résultats 4B/12B/27B : `RESULTS_TESTS.md`
§82 (effet significatif 4B→12B, plateau 12B→27B) — 1B relancé après un bug
réel trouvé au premier essai (`Gemma3TextModel` sans niveau `.language_model`,
seul palier de taille sans tour de vision, corrigé dans `saev5.py`).

## Sweep layer (setup classique K_EXTRA=5, 12B), réplication de §51

`run_ablation_classic_setup_k5_25m_layer12.slurm`, `_layer41.slurm` (layer 31
déjà couvert par le sweep taille de modèle ci-dessus, layer 24 = référence
`results_v10_emails_main/`) — réplique le balayage layer de §51 (seul résultat
individuel significatif du dépôt avant ce sweep, jamais répliqué) sous le
setup classique et n=150 à pleine puissance.
