# Expériences — index structuré

Vue synthétique, organisée par question de recherche plutôt que par ordre
chronologique. Le détail complet (logs, jobs SLURM, bugs rencontrés et corrigés) vit
dans `RESULTS_TESTS.md` — ce document y renvoie systématiquement.

## 1. Le pipeline de bout en bout fonctionne-t-il ?

**Oui**, validé à deux échelles :
- Smoketest local (`Gemma-3-270M-it`, 6 Go VRAM). 8/8 tests `pytest` passants.
- Échelle complète (`Gemma-3-12B-it`, cluster GPU) — `slurm/pipeline_runs/run_sae.slurm`/`slurm/pipeline_runs/run_sae_full.slurm`,
  cf. `RESULTS_TESTS.md` §2, §10.

## 2. Le corpus complet peut-il être augmenté et traité dans un budget de calcul raisonnable ?

**Oui**, après parallélisation. Augmentation complète (3480 mails × 13 axes = 45 240
générations) : ~63h GPU séquentielles → ~7h30 en array SLURM 8 shards parallèles.
Baseline sur le corpus complet (43 423 textes) : 1h11min après correction d'un bug de
complexité quadratique dans le pooling (`RESULTS_TESTS.md` §0).

| Étape | Volume | Durée mur |
|---|---|---|
| Augmentation (8 shards parallèles) | 45 240 générations (39 949 acceptées, 88,3%) | ~7h27 |
| Baseline (SAE natif) | 43 423 textes | 1h11min20s |
| Run complet pipeline (corpus generic) | ~9 500 textes | 37min32s |

## 3. Le taux de détection de l'intrus (odd-one-out) est-il limité par le volume d'entraînement ou par autre chose ?

Réponse : **principalement par le contenu du corpus d'entraînement** (les emails
n'y entraient jamais), **pas** par le volume brut de tokens. Diagnostic complet
et 3 runs de validation dans `RESULTS_TESTS.md` §12.

### Design de l'expérience

1. **Diagnostic** (lecture de code + inspection des résultats existants, aucun calcul
   supplémentaire) : le corpus d'entraînement de l'extension (`train_texts`) était bâti
   uniquement depuis FineWeb-2/Wikipedia générique (energy/sports/support) —
   `email_texts` n'était chargé qu'après l'entraînement, pour de la visualisation
   post-hoc uniquement.
2. **Fix** : corpus principal = mails originaux + augmentés (group-aware split par mail
   d'origine), corpus generic réduit à un rôle post-hoc secondaire (diffing
   cross-domaine uniquement).
3. **Ablation contrôlée** : 3 runs à corpus strictement identique (emails-dominant),
   seul `N_TOKENS_EXTRA_TRAIN` varie (100k / 500k / 2M) — isole l'effet du volume de
   l'effet du domaine (déjà corrigé dans les 3 runs).

### Résultats (n=150 features jugées par run, `odd_one_out_judge`)

| Run | Corpus | `N_TOKENS_EXTRA_TRAIN` | dead_feature | Taux interp. | IC95% approx. |
|---|---|---|---|---|---|
| `results_v9_full` (avant fix) | generic (energy/sports/support) | 500k | 0/10 | 20,0% (2/10) | très large (n=10) |
| `results_v10_ablation_tok100k` | emails+augmentés | 100k | 0/150 | 40,7% (61/150) | ±7,9 pts |
| `results_v10_emails_main` | emails+augmentés | 500k | 0/150 | 45,3% (68/150) | ±8,0 pts |
| `results_v10_ablation_tok2M` | emails+augmentés | 2M | 0/150 | 44,7% (67/150) | ±8,0 pts |

*(IC95% approx. = ±1.96·√(p(1-p)/n), Wald — indicatif, n modeste)*

### Conclusion

- **Effet domaine (generic → emails)** : +20 à +25 points, à volume comparable (500k).
  C'est le facteur dominant.
- **Effet volume (100k → 2M), à domaine fixé** : aucun écart statistiquement
  distinguable entre les 3 valeurs testées. Le SAE d'extension n'est pas "starved" à
  100k tokens une fois le domaine correct ; le porter à 2M n'apporte rien de mesurable.
- **Reste à expliquer** : ~55-59% des features d'extension restent non
  interprétables même corpus corrigé. Hypothèses non testées (pistes pour la suite) :
  robustesse du protocole de jugement (décision greedy unique vs vote/ensemble),
  qualité du contrôle négatif, capacité architecturale (`D_EXTRA`/`K_EXTRA`).

### Effet de bord : séparabilité linéaire des axes d'augmentation

La sonde de classification logistique (`downstream_classification`, prenant en
charge le cas multi-classe — cf. `RESULTS_TESTS.md` §12) sur les 14 classes
d'axes d'augmentation (émotion, urgence, registre, orthographe, original)
donne **acc_SAE = 93,5%** (Pipeline 1) et **79,3%** (Pipeline 2) sur le run
`results_v10_ablation_tok100k`. Les codes latents séparent donc très bien ces axes de
façon linéaire — résultat directement pertinent pour les cas d'usage détection
d'urgence / détection d'intention visés par le projet, au-delà du seul
diagnostic odd-one-out.

## 4. Le SAE capture-t-il des différences réelles entre sous-populations de mails (diffing) ?

Oui — `scripts/baseline_gemmascope.py` (SAE natif GemmaScope, sans extension) sur
originaux vs augmentés par axe : features significatives (Fisher exact + BH) pour
chaque axe/niveau, cf. `RESULTS_TESTS.md` §0 et §6 pour les tables complètes. Point de
vigilance documenté : biais de formatage résiduel (17,5% des mails augmentés contiennent
encore une ligne "Objet :" que les originaux n'ont pas), qui peut faire remonter des
features de formatage au lieu de features sémantiques comme "top discriminant" — corrigé
partiellement (prompt système), pas éliminé.

## 5. Les labels Neuronpedia sont-ils exploitables sans accès réseau permanent ?

Oui, après mise en cache locale canonique (`local_data/neuronpedia_labels/`) :
13 535 labels disponibles pour `gemma-3-12b-it/24-gemmascope-2-res-16k` (82,6% de
couverture sur 16 384 features). Réutilisé par tous les scripts (`saev5.py`,
`baseline_gemmascope.py`) sans jamais retenter d'appel réseau une fois le fichier
présent. Les CSV de diffing produits hors-ligne avant cette mise en cache ont été
relabellisés offline (`scripts/relabel_diff_csvs.py`, CPU uniquement, sans job GPU) :
91 920 labels corrigés dans `results_v9_test/cache_baseline_full/`.
