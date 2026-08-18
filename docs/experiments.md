# Expériences — index structuré

Vue synthétique, organisée par question de recherche plutôt que par ordre
chronologique. Le détail complet (logs, jobs SLURM) vit dans `RESULTS_TESTS.md` — ce
document y renvoie systématiquement.

## 1. Le corpus complet peut-il être traité dans un budget de calcul raisonnable ?

Augmentation complète (3480 mails × 13 axes = 45 240 générations) : ~63h GPU
séquentielles → ~7h30 en array SLURM à 8 shards parallèles.

Chronométrage par étape (`stage_timer`, `src/sae/saev5.py`) : chargement du corpus,
Pipeline 1 et Pipeline 2 sont chronométrés séparément à chaque run. Aucun run n'a
encore été conduit au volume qui compte réellement pour trancher l'effet du volume de
tokens (100-200M, cf. `docs/evaluation_protocol.md`) — pas de table de temps à cette
échelle pour l'instant.

## 2. Le taux de détection de l'intrus (odd-one-out) est-il limité par le volume d'entraînement ou par autre chose ?

### Design de l'expérience

1. **Diagnostic** : le corpus d'entraînement de l'extension (`train_texts`) était bâti
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

| Run | Corpus | `N_TOKENS_EXTRA_TRAIN` | Taux interp. | IC95% approx. |
|---|---|---|---|---|
| `results_v9_full` (avant fix) | generic (energy/sports/support) | 500k | 20,0% (2/10) | très large (n=10) |
| `results_v10_ablation_tok100k` | emails+augmentés | 100k | 40,7% (61/150) | ±7,9 pts |
| `results_v10_emails_main` | emails+augmentés | 500k | 45,3% (68/150) | ±8,0 pts |
| `results_v10_ablation_tok2M` | emails+augmentés | 2M | 44,7% (67/150) | ±8,0 pts |

*(IC95% approx. = ±1.96·√(p(1-p)/n), Wald — indicatif, n modeste)*

### Effet domaine (generic → emails)

+20 à +25 points, à volume comparable (500k). C'est le facteur dominant sur cette
plage testée.

### Effet volume (100k → 2M), à domaine fixé

Aucun écart statistiquement distinguable entre les 3 valeurs testées, dans cette
plage. Ça ne démontre pas que le volume est sans effet en général — seulement que,
sous contrainte de calcul, retenir 500k plutôt que 2M ne coûte rien de mesurable dans
cette plage précise. Elle reste 50 à 100× en dessous du seuil où la littérature (SAE
Boost) documente un effet de volume (`docs/evaluation_protocol.md`, `RESULTS_TESTS.md`
§18.3) — pas une preuve d'absence d'effet à l'échelle de production.

Reste à expliquer : ~55-59% des features d'extension restent non interprétables même
corpus corrigé. Hypothèses non testées : robustesse du protocole de jugement (décision
greedy unique vs vote/ensemble), qualité du contrôle négatif, capacité architecturale
(`D_EXTRA`/`K_EXTRA`).

### Effet de bord : séparabilité linéaire des axes d'augmentation

La sonde de classification logistique (`downstream_classification`) sur les 14 classes
d'axes d'augmentation (émotion, urgence, registre, orthographe, original) donne
acc_SAE = 93,5% (Pipeline 1) et 79,3% (Pipeline 2) sur `results_v10_ablation_tok100k`.
À comparer, sur exactement le même protocole (StratifiedKFold 5 plis, mêmes classes,
même corpus), à un baseline TF-IDF du texte brut (aucune activation SAE, juste la
présence de mots/expressions) : **87,0%** (`RESULTS_TESTS.md` §37). 93% du signal
rapporté pour P1 est donc déjà présent dans le texte brut ; le gain réel du SAE est
modeste (+6,5 points), pas la démonstration forte que suggère le chiffre 93,5% cité
seul. Aucun baseline équivalent n'a été mesuré pour le chiffre P2 (79,3%). Cause
probable : le corpus augmenté est généré sous contrainte de style explicite par axe
(`src/data/augmentation.py::AXES`), et en garde des tics lexicaux répétés (jusqu'à
100% des documents d'une classe partageant un même trigramme) — pas testé sur des
mails réels non augmentés.

## 3. Le SAE capture-t-il des différences réelles entre sous-populations de mails (diffing) ?

`scripts/baseline_gemmascope.py` (SAE natif GemmaScope, sans extension) sur originaux
vs augmentés par axe : features significatives (Fisher exact + BH) pour chaque
axe/niveau, cf. `RESULTS_TESTS.md` §0 et §6 pour les tables complètes. Point de
vigilance : biais de formatage résiduel (17,5% des mails augmentés contiennent encore
une ligne "Objet :" que les originaux n'ont pas), qui peut faire remonter des features
de formatage au lieu de features sémantiques comme "top discriminant" — corrigé
partiellement (prompt système), pas éliminé.
