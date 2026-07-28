# SAE — Résultats des tests & état des jobs Slurm

_Dernière mise à jour : 2026-07-13 (corpus complet). Cluster : partitions `a100`
(dgx-a100, 8×GPU), `h100` / `h100-bis` (dgx-h100{,-bis}, 8×GPU chacun)._

## 0. Corpus complet — augmentation parallélisée (8 shards) + baseline

**Vérification du biais de formatage (§7 plus bas) avant le run complet** : confirmé
sur l'échantillon test (0,03% des mails originaux ont "Objet :" vs 25,6% des
augmentés) → prompt système corrigé (`src/data/augmentation.py`, contrainte n°5
ajoutée : interdiction explicite d'ajouter une ligne "Objet :"/mise en forme absente
de l'original). Ancien fichier partiel généré avec l'ancien prompt (job 38949,
1440 lignes) archivé sous `results_v9_test/augmented_mails_STALE_biased_prompt_partial.jsonl.bak`
pour ne pas contaminer le run complet.

**Parallélisation** : le corpus complet (3480 mails × 13 axes = 45 240 générations)
aurait pris ~63h GPU en séquentiel (extrapolé du rythme observé sur les jobs
38949/38987). Ajout d'un mode sharding à `scripts/run_augmentation.py`
(`AUGMENT_NUM_SHARDS`/`AUGMENT_SHARD_IDX`, découpage entrelacé `df.iloc[idx::N]`) et
d'un nouveau `slurm/augmentation/run_augmentation_full.slurm` en **array Slurm 8 tâches** (`--array=0-7`),
toutes lancées en parallèle sur les 8 GPUs du nœud a100 (idle au moment du
lancement) — ~7h15-7h30 de mur au lieu de 63h.

**Résultat — job array 39017 (8 shards)** : ✅ **tous COMPLETED**, aucune erreur/
traceback/OOM dans les 8 logs (`logs/augmentation_full_39017_{0..7}.log`) :

| Shard | Durée | Générations | Acceptées |
|---|---|---|---|
| 0 | 7:13:52 | 5655 | 4989 |
| 1 | 7:21:44 | 5655 | 5004 |
| 2 | 7:19:48 | 5655 | 4963 |
| 3 | 7:23:55 | 5655 | 4960 |
| 4 | 7:19:30 | 5655 | 5043 |
| 5 | 7:23:06 | 5655 | 4948 |
| 6 | 7:27:00 | 5655 | 5001 |
| 7 | 7:23:28 | 5655 | 5041 |
| **Total** | **~7h27 (mur)** | **45 240** | **39 949 (88,3%)** |

Vérifié : 45 240 `aug_id` tous uniques entre shards (aucune collision), fusion via
`slurm/augmentation/merge_augmentation_shards.sh` → `results_v9_test/augmented_mails.jsonl` (45 240 lignes,
39 949 acceptées).

**Biais "Objet :" résiduel après correction du prompt** : réduit de 25,6% → **17,5%**
des textes acceptés (6999/39949). Le modèle ne suit l'instruction que partiellement à
température 0.8 (sampling). Pas de nouveau run relancé pour ce point (coût : encore
~7h GPU pour un gain marginal) — traité comme limite connue du corpus généré, pas
bloquant pour la suite du pipeline baseline.

**Job baseline complet (39246)** sur `augmented_mails.jsonl` (45 240 lignes) →
❌ **TIMEOUT** après 4h, sans qu'aucun résultat ne soit produit (`cache_baseline_full/`
resté vide) — le job n'avait même pas terminé le calcul `encode_corpus` (première
étape, avant la boucle sur les 13 axes).

**Cause racine identifiée** : `maxpool_sae_docs` (`src/analysis/activations.py`)
appelait `scatter_maxpool` à **chaque batch/chunk**, qui réalloue et parcourt un
tenseur `[n_docs, d_sae]` **complet** (`torch.full` + `torch.where` + le
`torch.maximum` de fusion) à chaque appel — un coût `O(n_docs · d_sae)` payé pour
chaque batch de 8 documents, au lieu de `O(batch_size · d_sae)`. Sur le run test
(4168 docs), ce surcoût restait tolérable (~11 min au total) ; sur le corpus complet
(43 423 docs, ~10,4× plus), le temps par batch grossit proportionnellement au nombre
total de documents → comportement quadratique en pratique. Extrapolation : plusieurs
heures rien que pour `encode_corpus`, bien au-delà du budget de 4h.

**Fix appliqué** : `maxpool_sae_docs` scatter désormais **in-place** dans
l'accumulateur `doc_acts` persistant (`scatter_reduce_` avec `include_self=True`),
sans réallocation ni parcours complet à chaque batch — la conversion finale
`-inf → 0` ne se fait qu'une seule fois, à la fin. Coût ramené à
`O(batch_size · d_sae)` par batch, comme attendu. `scatter_maxpool` elle-même n'a pas
été touchée (reste correcte pour ses autres usages en appel unique dans
`sae_shared.py`/`phrase_sae.py`, qui ne bouclent pas dessus par batch). Tests
`pytest` re-passés (8/8 OK).

**Job 39492** : `slurm/baseline_diffing/run_baseline_full.slurm` resoumis avec le fix → ✅ **COMPLETED en
1h11min20s** (contre >4h de TIMEOUT sans rien produire avant le fix). Corpus complet
traité : 3474 mails originaux + 39 949 augmentés acceptés = 43 423 textes encodés.

| Axe / niveau | Features significatives | Top feature | LOR |
|---|---|---|---|
| emotion / colere_forte | 4208 | F15531 | 10.07 |
| emotion / frustration | 3897 | F13696 | 9.67 |
| emotion / impatience | 4158 | F7340 | -7.91 |
| emotion / satisfaction | 3172 | F2916 | -5.11 |
| registre / soutenu | 3375 | F9825 | -8.08 |
| registre / standard | 3622 | F3610 | -6.71 |
| registre / familier | 3969 | F12615 | 11.32 |
| orthographe / degrade_leger | 1244 | F4817 | 5.89 |
| orthographe / degrade_fort | 4183 | F7340 | -7.64 |
| orthographe / corrige | 1171 | F4817 | 6.14 |
| urgence / panique | 3845 | F12392 | 9.79 |
| urgence / menace_resiliation | 3284 | F2918 | 7.13 |
| urgence / calme | 3623 | F2916 | -6.22 |

Sorties : `results_v9_test/cache_baseline_full/diff_<axe>__<niveau>.csv` + `.html`
(26 fichiers) + `baseline_doc_acts_all.pt` (cache, 2.85 Go).

**Note** : contrairement au run test (qui avait par chance obtenu quelques labels
Neuronpedia), le run complet n'a pas pu joindre Neuronpedia (même erreur SSL réseau
qu'avant, cluster offline) → toutes les features top sont des identifiants bruts
`F{idx}` non labellisés. Dégradation gracieuse attendue (pas un crash), mais réduit
l'interprétabilité immédiate des résultats — labels à générer hors-cluster via
`fetch_neuronpedia_labels()` si besoin d'interprétation fine.

---

## 10. Run à l'échelle complète du pipeline principal (`slurm/pipeline_runs/run_sae_full.slurm`)

Après validation que la chaîne augmentation → baseline fonctionnait, question posée :
le smoketest `slurm/pipeline_runs/run_sae.slurm` (volumes réduits ~×12) suffit-il, ou faut-il investiguer
avant un run à pleine échelle ? Vérification faite :

- **`saev5.py` (pipeline P1/P2) n'est PAS vulnérable au bug quadratique de `maxpool_sae_docs`** :
  son pooling P1 utilise `doc_maxpool` (`src/storage/fragment_store.py`) sur des
  fragments **sparses par document** (CSR, O(nnz) par doc) — c'est justement
  l'architecture v9 pensée pour éviter les gros tenseurs `[n_docs × d]` denses. Pas de
  correctif nécessaire ici.
- **Mais `saev5.py:720`** (`llm(**inputs, output_hidden_states=True)`) avait le **même
  gaspillage** que celui identifié dans `activations.py` (logits calculés sur toute la
  séquence/le vocabulaire Gemma-3 sans `logits_to_keep`). Jamais déclenché jusqu'ici
  (H100 80GB + volumes réduits du smoketest = grande marge), mais risque latent avant
  un run complet (plus de documents, éventuellement sur a100). **Fix préventif
  appliqué** : `logits_to_keep=1` ajouté à cet appel. Tests `pytest` re-passés (8/8 OK).

**Nouveau `slurm/pipeline_runs/run_sae_full.slurm`** (le smoketest `slurm/pipeline_runs/run_sae.slurm` reste inchangé, gardé
comme référence validée) :
- Toutes les réductions de volume retirées → valeurs par défaut de `src/config.py`/
  `saev5.py` appliquées : `N_TOTAL_ENERGY/SPORTS/SUPPORT=2000` (vs 400 en smoketest),
  `N_TOKENS_EXTRA_TRAIN=500000` (vs 60000), `D_SAE=8192` (vs 2048), `EPOCHS=30`
  (vs 15), `N_FEATURES_TO_LABEL=10` (vs 8).
- `SAE_ID`/largeur 16k **conservée** (pas une réduction de volume mais un choix
  délibéré documenté dans `Context.md`/`src/config.py` : bien meilleure couverture
  Neuronpedia que 262k).
- **Nouveau `SAVE_DIR=./results_v9_full/`** (distinct de `results_v9_test/`) : évite
  toute contamination par du cache de tailles/dimensions incompatibles (le smoketest a
  par ex. `D_SAE=2048` alors que le run complet utilise `D_SAE=8192` — un cache
  partagé aurait causé des erreurs de shape ou, pire, une réutilisation silencieuse de
  résultats à la mauvaise échelle).
- Partition `a100` (h100/h100-bis saturés par d'autres utilisateurs au moment du
  lancement — 0 GPU libre sur les deux ; a100 avait de la marge). `--time=24:00:00`,
  `--mem=150G` (mémoire hôte disponible resserrée sur tous les nœuds par la charge
  d'autres utilisateurs au moment du lancement — dimensionné en conséquence).

**Job 39531** : parti en `PENDING (Resources)` puis démarré automatiquement dès qu'un
GPU s'est libéré sur a100 → ✅ **COMPLETED en seulement 37min32s** (largement sous le
budget de 24h prévu — le smoketest lui-même avait déjà pris jusqu'à 69 min sur un run
"froid" ; l'A100 encaisse bien la charge complète). Vérifié en détail dans le log
(`sae_v9_full_39531.log`) que ce n'est pas un raccourci de cache ou un échec
silencieux : tous les marqueurs de volume correspondent bien aux valeurs par défaut
(2000 chunks/classe → 5400 train/600 test, 500 000 tokens résidus pour l'extension
FrozenCore, D_SAE=8192 confirmé par "8174/8192 features actives", 30/30 epochs P2
complétées), aucune erreur/traceback, nouveau `SAVE_DIR=results_v9_full/` bien utilisé
(pas de contamination par le cache du smoketest).

### Bilan comparatif — run complet (`results_v9_full/`)

| Pipeline | NMSE | L0 | dead% | ρ_SAE | silhouette | acc_SAE | FVE_base | clusters |
|---|---|---|---|---|---|---|---|---|
| P1 Gemma-3 SAE (Max-Pool tokens) | n/a | 2121.5 | 47.0 | 0.9099 | -0.0045 | 0.4600 | 0.7404 | 4 |
| P2 F2LLM Phrase-SAE (Max-Pool phrases) | 0.3047 | 22.5 | 0.0 | 0.8066 | -0.0009 | 0.5675 | — | 0 |

Comparé au dernier smoketest de référence (job 38896, volumes ×12 réduits) : ρ_SAE P1
en hausse (0.91 vs 0.80), dead% en baisse (47% vs 58%), FVE_base comparable (0.74 vs
0.72) — cohérent avec un SAE mieux entraîné sur plus de données. P2 dead%=0.0 (aucune
feature morte sur les 8192, contre 0.24% en smoketest sur 2048) : bon signe de
capacité utilisée. **Point à noter** : `clusters=0` pour P2 (contre 2 en smoketest) —
le clustering n'a trouvé qu'un seul cluster homogène sur le run complet, à
creuser si l'analyse de clustering P2 est un livrable attendu.

Comme pour le baseline, les labels Neuronpedia n'ont pas pu être récupérés (réseau
cluster offline) → features core affichées en `F{idx}` brut ; les 10 features
d'extension ont été labellisées par le juge LLM local (ex: "Espoir/Attente", "Numéros
Téléphone").

Sorties disponibles : `results_v9_full/results.json`, `results_v9_full/umap_*.html`,
`results_v9_full/p1_frozen_core_d1024_k32.pt`, `results_v9_full/p2_sae_dim320_d8192_k16.pt`.

---

## 11. Bilan général de la session

Tous les pipelines du repo ont été testés, les bugs bloquants corrigés, et les runs à
pleine échelle exécutés avec succès :

| Pipeline | Statut final |
|---|---|
| `slurm/pipeline_runs/run_sae.slurm` (smoketest v9) | ✅ déjà validé (inchangé) |
| `slurm/pipeline_runs/run_sae_full.slurm` (échelle complète) | ✅ COMPLETED (job 39531, 37min32s) |
| `slurm/validation/run_test_massive.slurm` | ✅ COMPLETED (job 38948) |
| `slurm/augmentation/run_augmentation_full.slurm` (8 shards parallèles) | ✅ COMPLETED (job 39017, 45 240 générations) |
| `slurm/baseline_diffing/run_baseline_full.slurm` | ✅ COMPLETED (job 39492, 1h11min, après fix perf) |
| Tests `pytest` | ✅ 8/8 (après chaque correctif) |

Bugs de code corrigés (persistants, indépendants des jobs individuels) :
1. `src/analysis/activations.py::extract_residual_acts` — `logits_to_keep=1` (évite
   un calcul de logits pleine séquence/vocabulaire inutile).
2. `src/analysis/activations.py::extract_residual_acts` — `@torch.no_grad()` sur
   fonction génératrice remplacé par `with torch.no_grad():` explicite (l'autograd
   tournait réellement actif, cause du premier OOM récurrent).
3. `src/analysis/activations.py::maxpool_sae_docs` — scatter in-place au lieu de
   réallouer un tenseur `[n_docs, d_sae]` complet à chaque batch (comportement
   quadratique avec la taille du corpus, cause du TIMEOUT sur le run complet).
4. `src/data/augmentation.py` (`_SYSTEM`) — contrainte anti-biais de formatage
   ("Objet :", markdown) ajoutée après détection sur l'échantillon test.
5. `src/sae/saev5.py:720` — même fix `logits_to_keep=1` appliqué préventivement
   (risque latent identique, jamais déclenché grâce au H100/volumes réduits du
   smoketest, mais applicable avant un run complet).

Aucune action supplémentaire requise de votre côté pour l'instant — tous les jobs
demandés ont terminé avec succès. Reste en suspens (mentionné mais non traité, à
votre discrétion) : le biais résiduel "Objet :" (17,5% des mails augmentés, cf. §0) et
le `clusters=0` de P2 sur le run complet (§10).

## 1. Ce qui a été audité

Tous les scripts `.slurm` du repo ont été relus, comparés aux scripts Python qu'ils
appellent, et testés (localement en syntaxe/imports, puis réellement soumis sur le
cluster quand c'était possible). Détail par script ci-dessous.

## 2. `slurm/pipeline_runs/run_sae.slurm` (pipeline principal v9, smoketest) — ✅ déjà validé, inchangé

Aucune modification : ce script fonctionne déjà de façon reproductible. Historique des
runs (`sae_v9_test_*.log`) :

| Job | Statut | Notes |
|---|---|---|
| 38557 | ✅ terminé | premier run propre après le fix `input_scale` (voir plus bas) |
| 38569 | ✅ terminé | reproductibilité confirmée |
| 38627 | ❌ crash | `RuntimeError: Missing key "input_scale"` — checkpoint `ExtendedSAE` d'une version de code antérieure au champ `input_scale`, incompatible avec le code actuel. Corrigé en régénérant le checkpoint (runs suivants OK). |
| 38862 | ❌ crash | `Release gemma-scope-2-12b-it not found` — mauvais `LOCAL_SAE_DIR`/release à ce moment-là |
| 38865, 38869 | ❌ crash | `LocalEntryNotFoundError` — tentative de fetch HF en ligne alors que le nœud de calcul n'a pas d'accès Internet direct (offline forcé) |
| 38867 | ⛔ annulé | `CANCELLED` manuellement pendant le chargement du modèle |
| 38896 | ✅ terminé | run de référence actuel (voir tableau métriques ci-dessous) |

### Métriques Pipeline 1 (Gemma-3 12B → GemmaScope 16k, max-pool token) — évolution

| Job | L0 | dead% | ρ_SAE | silhouette | acc_SAE | FVE_base | clusters |
|---|---|---|---|---|---|---|---|
| 38557 | 3880.7 | 84.6 | nan | -0.0236 | 0.5389 | nan | 5 |
| 38569 | 3895.2 | 84.6 | 0.7959 | -0.0245 | 0.5500 | 0.3133 | 3 |
| **38896** | **1761.8** | **58.0** | **0.7987** | **-0.0120** | **0.5556** | **0.7177** | 5 |

Amélioration nette entre 38569 et 38896 : L0 divisé par ~2.2, `dead%` en forte baisse,
FVE (variance expliquée du SAE étendu vs baseline pretrained) passe de 0.31 à 0.72 —
cohérent avec les correctifs successifs (AuxK actif, θ adaptatif, cache sparse) décrits
dans les commentaires du script.

### Métriques Pipeline 2 (F2LLM-v2-80M phrase-level SAE) — run 38896

- NMSE = 0.4536, L0 ≈ 19.6, dead% = 0.24 (quasi aucune feature morte), ρ_SAE = 0.724,
  acc_SAE = 0.6375 (vs acc_raw non calculé ici mais `clf_delta` positif observé sur
  d'autres runs comparables).
- Entraînement stable sur 15 epochs, NMSE décroît de 0.676 → 0.298 sans divergence.

**Conclusion : le pipeline v9 (`slurm/pipeline_runs/run_sae.slurm`) est fonctionnel et reproductible. Pas
besoin de le relancer pour ce tour de validation.**

## 3. Tests unitaires (`pytest`, CPU seulement) — ✅ passent tous

```
$ .venv/bin/python -m pytest tests/ -q
........                                                                 [100%]
8 passed, 3 warnings in 43s
```

Aucune modification nécessaire côté code testé.

## 4. Bugs trouvés dans les 3 autres `.slurm` (jamais fonctionnels tels quels)

Ces trois scripts n'avaient **jamais réussi à s'exécuter** (logs `augmentation_38743.log`,
`logs/test_massive_38845.log` : échec immédiat). Cause commune + bugs spécifiques :

### Bug commun : `uv run python ...`
Le nœud de calcul n'a pas d'accès Internet direct (proxy EDF bloquant). `uv run`
tente de re-résoudre/valider l'environnement contre le lockfile et essaie de
télécharger torch depuis `download.pytorch.org` → timeout après 3 retries, échec
systématique. `slurm/pipeline_runs/run_sae.slurm` (le seul qui marchait) contournait déjà ce problème en
appelant directement `.venv/bin/python` (l'environnement est déjà provisionné sur
disque). **Fix appliqué aux 3 scripts : `uv run python X` → `.venv/bin/python X`.**

### `slurm/validation/run_test_massive.slurm`
- Chemin faux : `test_massive_acts.py` n'existe qu'à `scripts/test_massive_acts.py`.
- `LOCAL_SAE_DIR` par défaut (`./local_data/saes/gemma-scope-2-12b-it`) pointe vers un
  répertoire **vide** — les poids réels sont dans
  `/home/h21486/SAE/saes/gemma-scope-2-12b-it-res/`. Ajouté en export explicite.
- **Statut : job 38948 soumis et TERMINÉ avec succès (exit 0:0)** après correctifs —
  confirme que les checkpoints `results_v9_test/p1_frozen_core_d1024_k32.pt` et
  `p1_raw_residuals.pt` sont bien lisibles et que le diagnostic de corrélation
  Pearson(activation extra, ||token||) s'exécute sans erreur.

### `slurm/augmentation/run_augmentation.slurm`
- Chemin faux : `run_augmentation.py` n'existe qu'à `scripts/run_augmentation.py`.
- `SAVE_DIR` non défini → tombait sur le défaut `./results/` au lieu de
  `./results_v9_test/`, incohérent avec le reste du pipeline v9. Fixé.
- **Statut : job 38949 soumis, EN COURS** (budget 2h, génère les variantes augmentées
  du vrai `Mails.tsv` via Gemma-3-12B-it — chargement modèle confirmé OK dans le log).

### `slurm/baseline_diffing/run_baseline.slurm`
- **Bug bloquant** : `scripts/baseline_gemmascope.py` a une fonction `main(mails_tsv,
  augmented_jsonl)` qui exige 2 arguments positionnels — le script slurm ne passait
  **aucun argument** → `IndexError` immédiat garanti. Fixé en passant
  `"$LOCAL_MAILS_PATH"` et `"${SAVE_DIR}augmented_mails.jsonl"`.
- Même bug `LOCAL_SAE_DIR` vide que `slurm/validation/run_test_massive.slurm` — fixé pareil.
- `SAVE_DIR`/`CACHE_DIR` non alignés avec `slurm/augmentation/run_augmentation.slurm` (le fichier
  `augmented_mails.jsonl` que ce script consomme est produit par l'augmentation) — les
  deux scripts utilisent maintenant le même `SAVE_DIR="./results_v9_test/"`.
- **Statut : job 38950 soumis avec `--dependency=afterok:38949`** — démarrera
  automatiquement quand l'augmentation aura réussi (sinon restera `PENDING` puis sera
  annulé si l'augmentation échoue).

## 5. Le run complet d'augmentation ne tient pas dans le budget de temps

**Job 38949** (augmentation complète, 3480 mails × 13 axes ≈ 45 240 générations,
`--time=2:00:00`) → **TIMEOUT** après 2h pile, tué par Slurm en plein calcul.
Progrès réel sauvegardé avant la coupure : 1440 générations écrites (1245 acceptées),
sur les 45 240 attendues. Au rythme observé (~12 générations/min), couvrir le corpus
complet prendrait **~63h de calcul GPU cumulées**. Le script reprend automatiquement
là où il s'est arrêté (skip des `aug_id` déjà présents dans le `.jsonl`), donc rien
n'est perdu, mais il faut plusieurs relances (ou un `--time` largement augmenté) pour
aller au bout — décision à prendre avec vous avant d'engager ce volume de calcul.

**Job 38950** (baseline, dépendait de 38949) est resté `PENDING
(DependencyNeverSatisfied)` puisque 38949 n'a pas fini en `COMPLETED` → annulé.

**Décision prise avec l'utilisateur : valider d'abord tout le pipeline baseline sur un
sous-échantillon rapide**, avant de décider si l'augmentation complète (63h GPU) vaut
le coût. Implémenté :
- `scripts/run_augmentation.py` : nouvelle option `AUGMENT_SAMPLE_N` (sous-échantillonnage
  déterministe, `random_state=SEED`) et `AUGMENT_OUT_NAME` (fichier de sortie séparé).
- `slurm/augmentation/run_augmentation.slurm` : run de test sur **60 mails** (`AUGMENT_SAMPLE_N=60`),
  sortie dans `augmented_mails_test.jsonl`, `--time=1:30:00`.
- `slurm/baseline_diffing/run_baseline.slurm` : pointe sur `augmented_mails_test.jsonl` pour ce test (à
  remplacer par `augmented_mails.jsonl` une fois le corpus complet généré).

## 6. Bug supplémentaire trouvé et corrigé : OOM CUDA dans `slurm/baseline_diffing/run_baseline.slurm`

**Job 38987** (augmentation test, 60 mails × 13 = 780 générations) → ✅ **COMPLETED**.
780/780 générations produites, 694 acceptées (89%), en ~1h.

**Job 38988** (baseline test, suite à 38987) → ❌ **FAILED** :
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 39.49 GiB ... 38.64 GiB memory in use.
```
Cause racine identifiée dans `src/analysis/activations.py::extract_residual_acts` :
l'appel `model(..., output_hidden_states=True)` sur un `AutoModelForCausalLM` calcule
par défaut (`logits_to_keep=0`) les **logits sur toute la séquence et tout le
vocabulaire** (Gemma-3 : ~262k tokens), alors que seul `hidden_states` est utilisé en
aval — rien qu'un batch de 8 × 512 tokens alloue ~2 GiB de logits totalement inutiles.
Ça passait inaperçu sur le run principal (`slurm/pipeline_runs/run_sae.slurm`, nœud H100 85 GB VRAM) mais
sature un A100 40 GB (`slurm/baseline_diffing/run_baseline.slurm`/`slurm/augmentation/run_augmentation.slurm` tournent sur la
partition `a100`).

**Fix appliqué** : `logits_to_keep=1` ajouté à l'appel du modèle (ne calcule les
logits que pour la dernière position, mémoire négligeable). Bénéficie aussi à
`src/sae/compare/pipeline.py` qui réutilise la même fonction. Tests `pytest`
re-passés après coup (8 passed).

**Job 38999** (retry avec `logits_to_keep=1`) → ❌ **FAILED**, toujours OOM, mais
**plus tard** dans le calcul (37.32/39.49 GiB déjà alloués) : le fix a bien éliminé le
gaspillage du `lm_head`, mais une seconde cause plus profonde restait active.

**Investigation approfondie + fix racine réel** : ajout de `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` + purge périodique `torch.cuda.empty_cache()` (hypothèse
fragmentation) → **job 39000, toujours FAILED**, avec cette fois très peu de mémoire
"reserved but unallocated" (489 MiB, contre 1.59 GiB avant) : la fragmentation n'était
**pas** la vraie cause, l'allocation elle-même grossissait légitimement.

Cause racine réelle trouvée : `extract_residual_acts` (`src/analysis/activations.py`)
était décorée `@torch.no_grad()` **sur une fonction génératrice** (elle contient
`yield`). Piège Python classique : ce décorateur n'entoure que l'appel qui *crée*
l'objet générateur (quasi instantané, le corps ne s'exécute pas encore à ce moment),
pas les itérations réelles faites ensuite via `next()`/`for`. Résultat : chaque forward
du modèle tournait en réalité **avec l'autograd actif**, retenant le graphe de calcul
des ~48 couches (`output_hidden_states=True`) alors qu'aucun `.backward()` n'est
jamais appelé — mémoire largement supérieure au nécessaire, et l'OOM survient sur le
batch dont le contenu (mails les plus longs) fait dépasser la capacité de la carte.

**Fix appliqué** : `with torch.no_grad():` explicite entourant tout le corps de la
boucle (au lieu du décorateur), qui couvre bien toutes les itérations du générateur.
Bénéficie aussi à `src/sae/compare/pipeline.py` (même fonction réutilisée).

**Job 39003** : `slurm/baseline_diffing/run_baseline.slurm` resoumis avec le vrai correctif → ✅ **COMPLETED**
en 11min24s (contre 3 échecs OOM successifs). Pipeline baseline validé de bout en bout
sur l'échantillon de 60 mails augmentés :

| Axe / niveau | Features significatives | Top feature (label) | LOR |
|---|---|---|---|
| emotion / colere_forte | 1202 | "roman numeral lists" | 11.63 |
| emotion / frustration | 1302 | "again, followed by comma" | 8.65 |
| emotion / impatience | 962 | "Subject: followed by email subject lines" | 9.19 |
| emotion / satisfaction | 519 | "Subject: followed by email subject lines" | 6.91 |
| registre / soutenu | 588 | "`:` or `**` followed by The" | 6.96 |
| registre / standard | 459 | "`:` or `**` followed by The" | 7.20 |
| registre / familier | 909 | "informal colloquial language" | 9.21 |
| orthographe / degrade_leger | 116 | "Subject: followed by email subject lines" | 6.86 |
| orthographe / degrade_fort | 852 | "F3527" (non labellisée) | 8.51 |
| orthographe / corrige | 104 | "Subject: followed by email subject lines" | 6.86 |
| urgence / panique | 1070 | "Subject: followed by email subject lines" | 12.51 |
| urgence / menace_resiliation | 1140 | "Subject: followed by email subject lines" | 8.28 |
| urgence / calme | 594 | "Subject: followed by email subject lines" | 6.80 |

Sorties : `results_v9_test/cache_baseline/diff_<axe>__<niveau>.csv` + `.html` (26
fichiers) + `baseline_doc_acts_all.pt` (cache réutilisable, 273 Mo).

**⚠ Observation d'interprétabilité (pas un bug) à noter pour vous** : le feature le
plus discriminant est très souvent un label Neuronpedia générique/structurel
("Subject: followed by email subject lines", "roman numeral lists", ponctuation) plutôt
qu'un concept sémantique lié à l'axe de perturbation (émotion, urgence...). Piste
probable : les mails augmentés générés par Gemma-3 gardent presque tous une ligne
"Subject:" ou une mise en forme (listes, `**`) que les mails originaux n'ont pas au
même degré → le SAE capture surtout un artefact de formatage du LLM générateur, pas
(seulement) le contenu de l'axe demandé. À vérifier en inspectant quelques mails
augmentés bruts (`results_v9_test/augmented_mails_test.jsonl`) avant de lancer le
corpus complet — un biais de génération qui se retrouverait à l'identique sur 45k
générations.

## 7. Jobs actuellement sur le cluster / historique de cette session

| Job ID | Script | But | Résultat |
|---|---|---|---|
| 38948 | `slurm/validation/run_test_massive.slurm` | Diagnostic massive activations | ✅ COMPLETED |
| 38949 | `slurm/augmentation/run_augmentation.slurm` (corpus complet) | Génère `augmented_mails.jsonl` | ⏱ TIMEOUT (2h) — 1440/45240 générations, reprise possible |
| 38950 | `slurm/baseline_diffing/run_baseline.slurm` (dépendait de 38949) | Baseline SAE natif | ⛔ annulé (dépendance jamais satisfaite) |
| 38987 | `slurm/augmentation/run_augmentation.slurm` (test, 60 mails) | Génère `augmented_mails_test.jsonl` | ✅ COMPLETED (780/780, 694 acceptées) |
| 38988 | `slurm/baseline_diffing/run_baseline.slurm` (test) | Baseline SAE natif sur échantillon test | ❌ FAILED (OOM, `lm_head` sur tout le vocab) |
| 38999 | `slurm/baseline_diffing/run_baseline.slurm` (test, retry 1) | Idem, avec fix `logits_to_keep=1` | ❌ FAILED (OOM plus tardif) |
| 39000 | `slurm/baseline_diffing/run_baseline.slurm` (test, retry 2) | Idem, avec fix fragmentation (`expandable_segments`) | ❌ FAILED (pas de la fragmentation) |
| 39003 | `slurm/baseline_diffing/run_baseline.slurm` (test, retry 3) | Idem, avec le vrai fix (`no_grad` sur générateur) | ✅ **COMPLETED** (11min24s) |

## 8. Bilan des correctifs de code appliqués (persistants, indépendants des jobs)

- `scripts/test_massive_acts.py` / `scripts/run_augmentation.py` / `scripts/baseline_gemmascope.py`
  appelés via `.venv/bin/python` au lieu de `uv run python` dans les 3 `.slurm`
  (contournement réseau bloqué sur le cluster).
- Chemins corrigés (`scripts/` manquant) dans `slurm/augmentation/run_augmentation.slurm` et `slurm/validation/run_test_massive.slurm`.
- `LOCAL_SAE_DIR` explicite (le défaut `local_data/saes/...` est vide) dans
  `slurm/validation/run_test_massive.slurm` et `slurm/baseline_diffing/run_baseline.slurm`.
- `slurm/baseline_diffing/run_baseline.slurm` : arguments positionnels manquants (`IndexError` garanti) ajoutés.
- `scripts/run_augmentation.py` : ajout `AUGMENT_SAMPLE_N` / `AUGMENT_OUT_NAME` pour
  permettre un run de validation sous-échantillonné.
- `src/analysis/activations.py::extract_residual_acts` :
  1. `logits_to_keep=1` (évitait un calcul de logits pleine séquence/vocabulaire inutile) ;
  2. `del out` après extraction du hidden state utile ;
  3. **fix principal** : `@torch.no_grad()` sur fonction génératrice remplacé par
     `with torch.no_grad():` explicite dans le corps — l'autograd tournait réellement
     actif sur tous les forwards précédents. Bénéficie aussi à `src/sae/compare/pipeline.py`.

## 9. Action à faire de votre côté

Le pipeline baseline est maintenant validé de bout en bout sur un échantillon. Deux
décisions restent à prendre avec vous :

1. **Lancer l'augmentation complète ?** (3480 mails × 13 axes ≈ 45 240 générations,
   ~63h GPU cumulées d'après le rythme observé). À faire en plusieurs jobs successifs
   (le script reprend automatiquement grâce au skip des `aug_id` déjà écrits) : relancer
   `slurm/augmentation/run_augmentation.slurm` avec `AUGMENT_SAMPLE_N` retiré (ou mis à 0) et `--time`
   augmenté (ex. 8h par job, relancer ~8 fois), puis repointer `slurm/baseline_diffing/run_baseline.slurm`
   sur `${SAVE_DIR}augmented_mails.jsonl` (au lieu de `..._test.jsonl`).
2. **Vérifier le biais de formatage observé** (§7) sur `augmented_mails_test.jsonl`
   avant d'engager le calcul complet — si confirmé, envisager de retirer les
   instructions de formatage (listes, gras) du prompt système dans
   `src/data/augmentation.py` (`_SYSTEM`) ou de nettoyer les lignes "Subject:" en
   post-traitement, pour que le SAE discrimine sur le contenu plutôt que la mise en forme.

Fichiers de résultats disponibles dès maintenant : `results_v9_test/cache_baseline/diff_*.csv`
(et `.html` pour visualisation) par axe/niveau.

---

## 12. Diagnostic et correction du faible taux de détection de l'intrus (odd-one-out)

**Question posée par l'utilisateur** : le taux de labellisation des features d'extension
(protocole odd-one-out, `src/sae/judge.py::odd_one_out_judge`) est faible — est-ce un
problème de volume d'entraînement (`N_TOKENS_EXTRA_TRAIN`) ou autre chose ?

### Diagnostic

Relecture de `src/sae/saev5.py` (bloc `MAIN`) + inspection de
`results_v9_full/cache/p1_judge_labels_extended.json` (dernier run complet
disponible, job 39531) :

- **0/10 features "dead_feature"** sur ce run → *pas* un problème de volume brut
  au sens strict (les features s'activent bel et bien).
- **Seulement 2/10 passaient le test odd-one-out** (20%).
- Les `pos_examples` retournés pour les features non interprétables étaient des
  extraits **FineWeb-2/Wikipedia génériques et sans rapport entre eux** (ex. un
  extrait sur un rappel produit iPad, un extrait sur les prisons norvégiennes, un
  extrait de recette de cuisine — présentés comme "exemples positifs" d'une même
  feature). Aucun concept cohérent commun → le juge LLM ne pouvait objectivement
  pas trouver l'intrus, l'odd-one-out échouant par construction, pas par manque
  de capacité du juge.
- **Cause racine** : `train_texts` (le corpus utilisé pour échantillonner le
  réservoir de résidus `N_TOKENS_EXTRA_TRAIN` et entraîner `ExtendedSAE`) était
  bâti **uniquement** à partir de `energy_texts + sports_texts + support_texts`
  (FineWeb-2/Wikipedia filtré par mots-clés). Les emails originaux et augmentés
  (`email_texts`) n'étaient chargés qu'**après** l'entraînement, uniquement pour
  une visualisation UMAP post-hoc (`analyze_with_umap`) — jamais vus par le SAE
  pendant l'entraînement. Le SAE d'extension apprenait donc des concepts
  Wikipedia génériques, jamais des concepts liés aux emails.

### Correction appliquée

- `src/data/preparation.py::build_email_train_test_corpus()` — nouveau corpus
  principal : mails originaux (`Mails.tsv`) + variantes augmentées acceptées
  (`augmented_mails.jsonl`, 39 949 lignes). Split **group-aware par mail
  d'origine** (`parent_id`) : un mail et toutes ses variantes tombent du même
  côté train/test (sinon une variante augmentée d'un mail de test fuiterait dans
  le train — quasi-duplicata sémantique, biais classique de leakage).
- `src/sae/saev5.py` : `train_texts`/`test_texts` = ce nouveau corpus
  (41 176 train / 2 177 test, contre ~5 400/600 FineWeb-2 avant — et surtout
  100% email au lieu de 0%). `energy_texts`/`sports_texts`/`support_texts`
  réduits (`N_TOTAL_*` 2000→300) et repositionnés en corpus **secondaire**
  (`diff_texts`/`diff_labels`), encodés post-hoc par le SAE déjà entraîné,
  gardés uniquement pour la démonstration de diffing cross-domaine existante
  (jamais utilisés pour l'entraînement).
- Labels Neuronpedia déplacés vers un cache canonique partagé
  (`local_data/neuronpedia_labels/`, cf. `src/config.py::NEURONPEDIA_LABELS_PATH`) :
  réutilisé par tous les runs sans jamais retenter d'appel réseau.
- `N_FEATURES_TO_LABEL` relevé de 10 à **150** pour une puissance statistique
  correcte sur le taux d'interprétabilité observé.

### 3 runs de validation (corpus emails+augmentés, N_FEATURES_TO_LABEL=150)

Objectif : confirmer le fix ET isoler l'effet du volume de celui du domaine, en
faisant varier `N_TOKENS_EXTRA_TRAIN` à corpus strictement identique.

| Run (SLURM job) | `N_TOKENS_EXTRA_TRAIN` | Durée | dead_feature | Taux interp. (odd-one-out) | ρ_interp moyen (interp=1) |
|---|---|---|---|---|---|
| **Baseline (avant fix)** — `results_v9_full`, job 39531 | 500 000 (corpus generic) | 37min32s | 0/10 | **2/10 = 20,0%** | n/a |
| `slurm/pipeline_runs/run_sae_v10_ablation_tok100k.slurm`, job 39661 | 100 000 | 3h11min37s | 0/150 | **61/150 = 40,7%** | 0,362 |
| `slurm/pipeline_runs/run_sae_v10_emails.slurm`, job 39660 (**run principal**) | 500 000 (défaut) | 3h01min53s | 0/150 | **68/150 = 45,3%** | 0,241 |
| `slurm/pipeline_runs/run_sae_v10_ablation_tok2M.slurm`, job 39662 | 2 000 000 | 2h21min01s | 0/150 | **67/150 = 44,7%** | 0,336 |

**Lecture** : corriger le domaine du corpus (emails dominants au lieu de
Wikipedia générique) **plus que double le taux d'interprétabilité** (20% → ~41-45%),
à volume de tokens comparable (500k) — c'est le principal levier. Une fois le
domaine corrigé, faire varier le budget de tokens sur un facteur **20×**
(100k → 2M) ne produit aucun écart significatif (40,7% / 45,3% / 44,7% — à
comparer à l'écart-type binomial attendu sur n=150, ≈ 4,1 points : les trois
valeurs sont dans le bruit les unes des autres). **Conclusion : le problème
observé était principalement un problème de contenu/domaine du corpus
d'entraînement (les emails n'entraient jamais dans le train), pas de volume
brut de tokens — une fois le domaine corrigé, le SAE d'extension n'est déjà
plus starved de volume à 100k tokens, et le porter à 2M n'apporte aucun gain
supplémentaire mesurable.**

Résultat additionnel obtenu au passage (bug `downstream_classification` corrigé
entre les jobs 39662 et 39661, cf. section suivante) : sur le job 100k (premier
run avec le fix), la sonde logistique sur les 14 classes d'axes email donne
**acc_SAE = 93,5%** (P1, Gemma-3/GemmaScope) et **79,3%** (P2, F2LLM/PhraseSAE)
— les codes latents du SAE séparent donc très bien les axes de perturbation
(émotion, urgence, registre, orthographe, original) de façon linéaire, un
résultat encourageant pour les cas d'usage visés (détection d'urgence,
détection d'intention) au-delà du seul diagnostic odd-one-out.

Qualité des labels obtenus (contraste direct avec les extraits Wikipedia du run
avant fix) : `Réclamations Clients`, `Litiges Factures`, `Résiliation Énergie`,
`Menace Résiliation`, `Demande Urgente`, `Problèmes énergie` — des concepts
directement alignés avec les objectifs métier (détection d'urgence, d'intention,
réclamations) au lieu d'artefacts Wikipedia sans rapport.

### Bug additionnel trouvé et corrigé pendant ces runs

`src/analysis/metrics.py::downstream_classification` : la nouvelle sonde de
classification multi-classe sur les axes d'augmentation email (14 classes)
échouait silencieusement (exception attrapée, métrique `nan`) — `liblinear` ne
supporte que la classification binaire dans les versions récentes de
scikit-learn. Fix : `lbfgs` (multinomial natif) au-delà de 2 classes,
`liblinear` conservé pour le probe binaire energy/sports préexistant (ne change
aucun résultat déjà validé). Les jobs 39660 et 39662 ont tourné avant que le fix
ne soit sur disque (métrique `acc_axes_email` = `nan` dans leurs `results.json`,
non bloquant pour le diagnostic principal odd-one-out, indépendant de cette
métrique annexe) ; le job 39661 (soumis avant le fix mais resté `PENDING` en
file d'attente jusqu'après le commit du correctif) a démarré son exécution
avec le code corrigé et donne les valeurs `acc_axes_email` exploitables
ci-dessus.

---

## 13. Suites données au diagnostic §12 : robustesse du juge et validation métier sur mails originaux

Deux analyses complémentaires, lancées après le diagnostic §12, pour répondre plus
largement aux objectifs du stage (`Context.md`, section "Projet" : détection
d'urgence, détection d'intentions) et à la piste ouverte sur le résidu de ~55-59% de
features non interprétées.

### 13.1. Le résidu non-interprété est-il dû au biais de position du protocole de jugement ?

`scripts/judge_robustness_check.py` (nouveau) + `slurm/analysis/run_judge_robustness.slurm`, job
40672 — **aucune réextraction Gemma-3** : réutilise les activations et fragments déjà
en cache (`results_v10_emails_main/`), recharge seulement le modèle comme juge.
Pour chacune des 150 features déjà jugées, répète la question odd-one-out **5 fois**
avec un ordre de mélange différent à chaque fois (mêmes exemples), et calcule le vote
majoritaire.

| Métrique | Valeur |
|---|---|
| Taux interp. single-shot (déjà connu) | 45,3% (68/150) |
| Taux interp. vote majoritaire (5 répétitions) | 48,7% (73/150) |
| Features dont la décision change (0→1 ou 1→0) | 47/150 (31,3%) — 26 vers interprétable, 21 vers non-interprétable |
| Taux d'accord moyen entre les 5 répétitions | 80,3% |
| Distribution des décisions (n_correct/5) | 0/5: 22 · 1/5: 30 · 2/5: 25 · 3/5: 19 · 4/5: 30 · 5/5: 24 |

**Lecture** : seulement 46/150 features (30,7%) obtiennent une décision **unanime**
sur les 5 répétitions (0/5 ou 5/5) — les 69,3% restantes montrent un désaccord partiel
purement dû à l'ordre de présentation des mêmes exemples. Le taux agrégé change peu
(45,3%→48,7%, les bascules dans les deux sens se compensant globalement), mais la
fiabilité de la décision **par feature individuelle** est clairement insuffisante à
une seule question greedy. **Conclusion : une partie du résidu non-interprété est due
au protocole de jugement lui-même (bruit de position), pas nécessairement à un défaut
réel des features.** Un vote majoritaire sur plusieurs répétitions (ou une
température non nulle) est recommandé avant de conclure qu'une feature spécifique
n'est "pas interprétable".

### 13.2. Le SAE prédit-il l'urgence et l'intention sur des mails originaux (pas augmentés) ?

`scripts/intent_urgency_probe.py` (nouveau) — **zéro calcul GPU**, réutilise les
activations déjà en cache (`p1_all_doc_acts_ext_d1024.pt`) et les labels faibles par
regex déjà calculés dans `src/data/dataset.py::load_mails_tsv`
(`INTENT_KEYWORDS_FR` : réclamation, résiliation, remboursement, information,
urgence), appliqués aux 3300 mails **réels** (non augmentés) du split train.
Répond directement aux objectifs du stage "détection d'urgence"/"détection
d'intentions" avec un test indépendant du corpus augmenté synthétique.

| Intention | Prévalence | acc_SAE (sonde logistique, 5-fold) | Baseline (classe majoritaire) | Δ |
|---|---|---|---|---|
| `intent_urgence` | 29,3% (968/3300) | **97,7%** | 70,7% | **+27,0 pts** |
| `intent_reclamation` | 55,1% (1819/3300) | **97,7%** | 55,1% | **+42,6 pts** |
| `intent_information` | 18,2% (599/3300) | 87,8% | 81,8% | +6,0 pts |
| `intent_remboursement` | 14,5% (479/3300) | 84,5% | 85,5% | −1,0 pt |
| `intent_resiliation` | 0,03% (1/3480) | — | — | ignoré (classe dégénérée) |

**Lecture** : les codes latents du SAE (Pipeline 1, 17 408 dimensions) séparent très
nettement l'**urgence** (+27 points au-dessus de la baseline) et la **réclamation**
(+42,6 points) sur des mails originaux non augmentés — validation directe et à moindre
coût (aucun calcul GPU supplémentaire) des deux objectifs "détection d'urgence" et
"détection d'intentions" du projet. L'**information** est modérément mieux détectée
que la baseline ; le **remboursement** ne l'est pas mieux qu'une baseline déjà forte
(la classe est déséquilibrée à 85,5% de négatifs, la baseline est donc déjà un
prédicteur difficile à battre) — pas un échec du SAE en soi, plutôt un signal que
"remboursement" au sens de la regex n'est peut-être pas une catégorie linéairement
séparable dans cet espace, ou que le label faible par regex est trop bruité pour ce
cas précis.

---

## 14. Suites supplémentaires : biais Objet: (mesure avant/après), dashboard, comparaison SAELens chiffrée

### 14.1. Effet mesuré du fix Objet:/Subject: sur le diffing complet

Suite au fix `load_augmented` (commit "Corrige le biais résiduel Objet:/Subject:..."),
`slurm/baseline_diffing/run_baseline_full_v2.slurm` (job 40674, ✅ COMPLETED en 1h06min05s) relance
`scripts/baseline_gemmascope.py` sur le corpus complet (43 423 textes) avec le texte
augmenté nettoyé, dans un nouveau répertoire (`results_v11_baseline_objetfix/`, pas de
partage de cache avec `results_v9_test/cache_baseline_full` pour éviter toute
contamination). Comparaison directe, mêmes 13 combinaisons axe/niveau :

| Axe / niveau | Features significatives (avant) | (après) | Δ |
|---|---|---|---|
| orthographe / corrigé | 1171 | 413 | **−64,7%** |
| orthographe / dégradé léger | 1244 | 635 | **−49,0%** |
| urgence / panique | 3845 | 3556 | **−7,5%** |
| urgence / menace_résiliation | 3284 | 3183 | −3,1% |
| registre / soutenu | 3375 | 3323 | −1,5% |
| orthographe / dégradé fort | 4183 | 4170 | −0,3% |
| emotion / impatience | 4158 | 4149 | −0,2% |
| registre / standard | 3622 | 3614 | −0,2% |
| urgence / calme | 3623 | 3615 | −0,2% |
| registre / familier | 3969 | 3966 | −0,1% |
| emotion / colère forte | 4208 | 4210 | +0,0% |
| emotion / satisfaction | 3172 | 3173 | +0,0% |
| emotion / frustration | 3897 | 3912 | +0,4% |

**Lecture** : le fix réduit fortement le nombre de features "significatives" pour les
deux axes **orthographe** (−65% et −49%) — cohérent, ces axes (dégradation/correction
de l'orthographe) sont exactement ceux où une ligne "Objet :" ajoutée par erreur est la
plus susceptible d'être confondue avec la variation orthographique réellement visée
(tous deux sont des perturbations de surface au niveau du texte brut). Effet modéré sur
**urgence/panique** (−7,5%) et **urgence/menace_résiliation** (−3,1%) ; effet
négligeable (<2%) sur les axes émotion et registre. Le TOP-1 feature change pour 2 des
13 combinaisons (orthographe/corrigé : "Subject: followed by email subject lines" →
"fine-tuning, answer" ; orthographe/dégradé léger : idem → "conjunctions in Slavic
languages"). Le proportion globale de features significatives labellisées "Subject:"/
"Objet:" reste faible dans les deux cas (0,21%→0,20%) : sur le corpus complet (43k
textes, forte puissance statistique), l'artefact ne dominait déjà pas la majorité des
features significatives (contrairement à l'échantillon test à 60 mails, §6, où il
apparaissait en position 1 pour 8/13 axes) — mais il gonflait spécifiquement le
DÉNOMBREMENT de features significatives sur les axes orthographiques, un effet
maintenant corrigé.

### 14.2. Dashboard Streamlit

`src/visualization/dashboard.py` (fonctionnalité listée dès l'énoncé initial du
projet, `Context.md`) : lit uniquement des artefacts déjà sur disque (JSON, parquet,
CSV) -- aucun modèle chargé, démarre en quelques secondes. Pages : vue d'ensemble
(métriques par run), UMAP interactif (Plotly, coloration par label/cluster), features
(core Neuronpedia + extension + phrase-level, avec exemples positifs ET négatifs --
`neg_example` maintenant persisté dans `src/sae/judge.py` pour les runs futurs),
diffing (parcours des `diff_*.csv`), recherche par mot-clé sur les labels, et une page
dédiée aux résultats urgence/intention + robustesse du juge (§13). Testé avec
`streamlit.testing.v1.AppTest` sur toutes les pages × tous les runs disponibles
(`results_v7` à `results_v11_baseline_objetfix`) : 0 exception.

```bash
.venv/bin/python -m streamlit run src/visualization/dashboard.py
```

### 14.3. Comparaison SAELens chiffrée

Cf. `docs/references.md` (section dédiée) et `scripts/saelens_numeric_comparison.py` :
désaccord numérique important (0,41 / 0,83 / 1,00 selon la formule) entre notre FVE et
les deux formules de variance expliquée maintenues par `sae_lens.evals`, sur le même
SAE natif et les mêmes activations réelles -- causé par les activations massives de
Gemma-3 (une dimension atteint une magnitude ~74752 contre une moyenne ~53).

---

## 15. Relecture du papier de référence (interp_embed) : 4 corrections concrètes

Suite à une relecture détaillée de `pdf/InterpretableSAE_Embeddings.pdf` (Jiang, Sun
et al. 2025 — référence n°3 de `Context.md`, comparaison "systématique" demandée par
la règle n°2), comparaison ligne à ligne avec notre code. Quatre écarts concrets
trouvés et corrigés, au-delà de la comparaison de formule déjà faite pour SAELens
(§13/docs/references.md).

### 15.1. Bug : `property_based_retrieval`/`targeted_clustering_by_axis` (matching par sous-chaîne)

Le papier (§4.4, Appendix F.1) sélectionne les latents pertinents pour une requête par
**similarité d'embedding dense** entre le label et la requête. Notre code faisait un
matching par **sous-chaîne littérale** (`any(word in lbl.lower() for word in
query_words)`) — vérifié empiriquement : la requête "urgence réclamation client" ne
retournait quasiment que des labels contenant le mot "client" au sens large, y compris
des faux positifs absurdes ("client/server architecture", "Client secret",
"spacecraft client"), en ratant tout label sémantiquement lié mais formulé
différemment. De plus, dans `property_based_retrieval`, le poids de pondération
"température" (`exp(-(rank/k)/temperature)`) utilisait `rank = position dans
matched_latents`, c'est-à-dire l'**ordre d'itération du dict** `feature_labels` — sans
rapport avec la pertinence réelle à la requête, rendant la pondération arbitraire.

**Fix** : nouvelle fonction `select_latents_by_similarity` (`src/sae/saev5.py`),
câblée dans les deux fonctions, qui trie les latents par similarité cosinus
décroissante entre leur label et la requête — le rang de pondération devient donc le
rang de pertinence réel.

### 15.2. Choix du modèle d'embedding : F2LLM insuffisant sur des labels courts, bge-m3 validé

Premier essai avec F2LLM-v2-80M (déjà utilisé partout ailleurs dans le projet,
Pipeline 2) : bons résultats sur "urgence réclamation client" (labels pertinents
remontés : `[EXT] Réclamation Urgente`, `emergency call 911`, `[EXT] Réclamations
Clients`...) mais résultats **sans rapport** sur "facturation résiliation panne"
(`fact statement`, `unlock loss cash`, `disclaimersobviousexaggerate`, `end of
sentence`...) — aucun des labels `[EXT] Résiliation*` pourtant présents dans le jeu
n'apparaissait dans le top 15. Diagnostic : F2LLM utilise un pooling **dernier-token**
(optimisé pour des phrases complètes en contexte de recherche documentaire), mal
adapté à la similarité sémantique de labels courts (2-5 mots) en contexte
cross-lingue (requête FR, labels mêlés FR/EN).

Testé en alternative : **bge-m3** (pooling `[CLS]`, multilingue, déjà présent
localement sous `models/bge-m3`, conçu pour la similarité sémantique/retrieval) —
bons résultats sur les **deux** requêtes :

| Requête | bge-m3 — top résultats |
|---|---|
| "urgence réclamation client" | `urgent requests and invoices` (0.745), `[EXT] Réclamation Urgente` (0.743×2), `[EXT] Réclamation Client` (0.705×3), `[EXT] Réclamations Clients` (0.696×4) |
| "facturation résiliation panne" | `[EXT] Facture contestée` (0.605×2), `[EXT] Résiliation contrat` (0.604), `[EXT] Réclamations Factures` (0.585), `[EXT] Contestation Facture` (0.572×2), `[EXT] Litige Facture` (0.568), `[EXT] Colère Facture` (0.567) |

**Fix retenu** : `select_latents_by_similarity` (et `find_interesting_pairs`, §15.3)
utilisent bge-m3 (`src/config.py::LATENT_LABEL_EMB_MODEL`), pas F2LLM. Vérifié
bout-en-bout via `saev5mod.select_latents_by_similarity` directement (pas seulement le
script de diagnostic autonome) : résultats identiques aux tests isolés.

### 15.3. Corrélations "intéressantes" : filtre manquant + pipeline jamais branché

Constat additionnel en creusant ce point : `cooccurrence_graph` (NPMI + communautés
Louvain) n'était **même pas appelée** dans `saev5.py` — seule la matrice NPMI brute
était calculée et cachée (`p1_npmi.pt`), sans aucune analyse en sortie. Le papier
(§4.2/Appendix E.1) filtre en plus les paires à **NPMI élevé ET similarité sémantique
des labels faible** pour isoler les corrélations non-triviales (biais/artefacts) des
évidentes (labels quasi-synonymes). Nouvelle fonction `find_interesting_pairs`
(`src/analysis/cooccurrence.py`), câblée dans le pipeline principal (sortie
`p1_interesting_correlations.json`), embeddings bge-m3 pour la similarité des labels.

### 15.4. Protocole de labellisation : gate odd-one-out vs génération contrastive directe

Le papier (Appendix C) ne fait **jamais** de gate avant de labelliser : il présente
toujours 10 positifs + 10 négatifs à un LLM et demande directement le label par
contraste, avec des instructions détaillées (contexte avant les marqueurs, ignorer
les tokens spéciaux, une propriété unifiée). Notre `odd_one_out_judge` ne génère un
label QUE si le test odd-one-out réussit (déjà mesuré bruyant, §13.1 : 30,7% de
décisions unanimes seulement sur 5 réordonnancements), et le prompt de labellisation
n'utilisait que les 9 positifs (aucun négatif).

**Test** (`scripts/contrastive_labeling_test.py`, réutilise les activations/fragments
déjà en cache, aucune réextraction) sur les mêmes 150 features déjà jugées :

- **Bug trouvé en cours d'écriture** : nos négatifs (comme ceux du protocole
  existant, `build_feature_examples_with_control`) portent un marqueur `<<...>>` sur
  un token arbitraire (milieu de document) — le papier est explicite ("NEGATIVE
  samples... no << >> markers"). Corrigé (marqueurs retirés des négatifs).
- **1er run** : prompt de labellisation contenant un exemple de valeur JSON
  plausible (`"phrase courte en français (<=4 mots)"`) — le LLM l'a recopié
  littéralement comme label pour 48/82 features originellement non-interprétables
  (bug de prompt, pas un résultat). Taux de récupération apparent avant correction :
  100% (biaisé par cet artefact).
- **2e run** (prompt corrigé, placeholders non-ambigus) : plus aucun echo de
  template. Sur les 82 features originellement non-interprétables (odd-one-out
  échoué), tous obtiennent un label spécifique et distinct — inspection qualitative
  d'un large échantillon : labels cohérents et directement pertinents pour le
  domaine EDF (`Mise en service énergie`, `Numéro de contrat`, `Demande de
  résiliation`, `Informations bancaires`, `Réclamation de facture`, `Sentiment
  d'urgence`, `Nom de famille en fin`...), pas de contenu générique ou incohérent
  détecté par échantillonnage manuel.
- **Limite méthodologique confirmée** : le champ `confident` auto-rapporté par le
  LLM (censé permettre de refuser un label quand aucune propriété cohérente n'existe)
  est resté `true` pour 150/150 features dans les deux runs, y compris pour les
  quelques cas encore incertains à l'inspection manuelle — l'auto-évaluation de
  confiance du LLM n'est **pas un signal fiable en l'état** (cohérent avec le biais
  de complaisance documenté pour les LLM en général, et avec le résultat de §13.1 sur
  la fiabilité limitée du jugement LLM sur ce protocole).

**Conclusion et recommandation** : la génération de labels ne devrait **pas** être
gatée par le test odd-one-out — celui-ci est plus fiable comme **score
d'interprétabilité diagnostique indépendant** (avec vote majoritaire, §13.1) que
comme filtre de labellisation. Recommandation retenue pour un futur refactor de
`src/sae/judge.py` : (1) toujours générer un label par contraste direct (10 positifs
+ 10 négatifs non marqués, instructions détaillées façon Appendix C), (2) évaluer
l'interprétabilité séparément par vote majoritaire odd-one-out (déjà validé, §13.1),
(3) ne **pas** se fier au champ `confident` auto-rapporté comme filtre de qualité —
lui substituer soit une validation croisée (ex. ρ_interp, déjà implémenté), soit une
revue humaine sur échantillon. **Ce changement n'a pas été appliqué au pipeline de
production dans cette session** (implique de refaire tourner les 3 runs de
validation §12 pour un nombre comparable avant de remplacer le chiffre 45,3% déjà
publié dans le rapport) — documenté ici comme piste well-evidenced pour la suite du
stage, pas comme correction déjà intégrée.

---

## 16. Qualité de l'explication document-level + protocole de test complet du repo

Suite directe de la question utilisateur "comment tester ma pipeline de bout en bout
pour estimer la qualité de notre solution d'explication des documents ?". Deux tests
complémentaires (fidélité causale + plausibilité perçue), puis un protocole
d'évaluation consolidé couvrant l'ensemble des méthodes du dépôt, sous conditions
fixées (`docs/evaluation_protocol.md`).

### 16.1. Fidélité de l'explication (ablation)

`scripts/explanation_fidelity_test.py` (CPU uniquement, réutilise les activations déjà
en cache) : ajuste une sonde logistique finale par intention (urgence, réclamation,
information, remboursement) sur les mails originaux, identifie pour 200 documents
positifs bien classés (proba > 0.7) les 10 features dont la contribution
(`coef × activation`) est la plus forte, puis ablate (met à zéro) ce top-10 et mesure
la chute de probabilité prédite, comparée à l'ablation de 10 features actives
aléatoires et de 10 features actives les moins contributives (bottom-10).

| Intention | n docs testés | Chute top-10 | Chute random-10 | Chute bottom-10 | Ratio top/random |
|---|---|---|---|---|---|
| Réclamation | 200 | 0,576 | ~0,000 | ~0,000 | 576 225× |
| Remboursement | 200 | 0,9997 | 0,0009 | ~0,000 | 1 058× |
| Information | 200 | 0,9998 | 0,0040 | ~0,000 | 251× |
| Urgence | 200 | 0,612 | ~0,000 | ~0,000 | 42 837× |

**Lecture** : l'ablation du top-10 des features "citées" comme explication fait
s'effondrer la probabilité prédite (58 à 100 points), alors que l'ablation de features
actives aléatoires ou peu contributives ne change quasiment rien (<0,4 point). Les
features désignées comme explication ne sont donc pas de simples labels plausibles a
posteriori : elles portent réellement la décision de la sonde. Résultat sans ambiguïté
sur cette architecture (sonde linéaire directement sur les codes SAE).

### 16.2. Plausibilité de l'explication (choix forcé, juge LLM)

`scripts/explanation_plausibility_test.py` (GPU, juge Gemma-3-12B-it, réutilise les
activations déjà en cache) : sur 60 mails originaux échantillonnés, présente au juge le
mail + deux ensembles de labels (le top-8 réel des features les plus actives pour ce
document, et un décoy de 8 labels tirés au hasard parmi les features labellisées mais
non actives pour ce document), ordre A/B mélangé aléatoirement, et demande lequel
explique le mieux le mail.

| Métrique | Valeur |
|---|---|
| Taux de succès (choix de l'ensemble réel) | **71,7%** (43/60) |
| Niveau du hasard | 50% |
| Signification | z ≈ 3,4 (approximation normale), p < 0,001 |

**Lecture** : le juge préfère l'explication réelle à un décoy aléatoire nettement plus
souvent que le hasard — l'explication a une valeur perçue réelle pour un lecteur (LLM),
sans être parfaite (28,3% des cas où le décoy est jugé au moins aussi bon, cohérent
avec le taux d'interprétabilité ~45% mesuré par ailleurs : une partie substantielle des
features "expliquant" un document ne sont elles-mêmes pas clairement monosémantiques).

### 16.3. Corrélation "intéressantes" — calculée rétroactivement

`scripts/compute_interesting_correlations_retro.py` (GPU léger, bge-m3 uniquement,
aucune réextraction Gemma-3) : `find_interesting_pairs` (§15.3) a été ajoutée au
pipeline principal APRÈS la production de `results_v10_emails_main/` — recalculée ici
directement depuis `test_doc_acts` déjà en cache. Résultat : seulement **3 paires**
retenues sur 26 579 arêtes du graphe de cooccurrence (3 395 nœuds), et 2 des 3 paires
impliquent une feature non labellisée (`F17402`, `F17315`, `F43`) — résultat honnête
mais peu exploitable en l'état (impossible de juger si la corrélation est un artefact
réel sans label sur au moins un des deux côtés). Piste de suite : élargir la plage de
fréquence (`min_freq`/`max_freq` de `cooccurrence_graph`) ou prioriser les paires où
les deux features sont labellisées.

### 16.4. Protocole d'évaluation complet du dépôt (conditions fixées)

`docs/evaluation_protocol.md` (nouveau) : recense les 16 capacités/méthodes du dépôt,
leur commande de reproduction, leur artefact de résultat, et l'alternative à laquelle
chacune est comparée, sous un jeu de conditions **fixé** (Gemma-3-12B-it, GemmaScope
16k+FrozenCore par défaut, corpus emails-dominant, F2LLM-v2-**330M** pour le backbone
Pipeline 2 — "assez grand", décision de cette session —, bge-m3 pour la similarité de
labels). `scripts/consolidate_evaluation_report.py` (nouveau) assemble automatiquement
tous les artefacts disponibles d'un `SAVE_DIR` donné en un rapport markdown unique
(`EVALUATION_REPORT.md`) + un résumé JSON, exposés dans le dashboard (nouvel onglet
"Rapport consolidé"). Nouvel onglet dashboard "Explication (fidélité/plausibilité)"
pour les résultats §16.1-16.2, avec inspection des exemples individuels (documents,
features citées, cas où le juge s'est trompé).

### 16.5. Comparaison du backbone Pipeline 2 : F2LLM-v2-80M vs -330M

`slurm/pipeline_runs/run_sae_v10_p2_f2llm330m.slurm` (Pipeline 2 seul, corpus identique à
`results_v10_emails_main`, aucun réentraînement Gemma-3/GemmaScope nécessaire) :
MATRYOSHKA_DIM=320 reste une simple troncature des 320 premiers dims de l'embedding
dernier-token, mécaniquement compatible avec n'importe quelle taille de backbone
(vérifié : `hidden_size=896` pour F2LLM-330M ≥ 320).

Deux tentatives nécessaires : la première a échoué à la toute dernière étape
(`MODEL_ID` non défini explicitement pour le juge P2 -- même bug que §16.2 avant son
premier correctif), mais l'entraînement complet du `PhraseLevelSAE` sur les
embeddings F2LLM-330M était déjà en cache, la reprise (job 40745) a donc pu sauter
directement à l'étape manquante (quelques secondes au lieu de ~10 minutes).

| Métrique | F2LLM-v2-80M | F2LLM-v2-330M | Δ |
|---|---|---|---|
| NMSE | 0,0745 (`results_v10_emails_main`) | **0,0689** | −7,5% (meilleur) |
| L0 | 16,14 | 15,94 | ≈ identique |
| dead% | 0,63% | 0,70% | ≈ identique |
| ρ_SAE | 0,9597 | 0,9574 | ≈ identique |
| silhouette | 0,0212 | 0,0183 | légèrement inférieur |
| clusters | 4 | 4 | identique |
| acc_SAE (energy/sports, corpus diffing) | 0,6650-0,6717 | **0,6867** | +2 points (meilleur) |
| acc_SAE (axes email, 14 classes) | 0,7933 (`results_v10_ablation_tok100k`, seul run 80M avec le fix `downstream_classification`) | 0,7717 | **−2,2 points (légèrement moins bon)** |

**Lecture** : résultat **mixte**, pas de gain net et homogène. F2LLM-330M reconstruit
légèrement mieux (NMSE, cohérent avec un backbone plus grand) et sépare légèrement
mieux le corpus diffing generic (energy/sports), mais sépare légèrement MOINS bien
les axes email (la métrique la plus directement liée aux objectifs métier du
projet). Aucun de ces écarts n'est de l'ordre de grandeur d'un problème majeur (tous
< 3 points ou < 8% relatif) : conclusion retenue pour cette passe -- **pas de
justification claire pour préférer -330M à -80M sur ce projet** ; le choix n'est pas
un facteur bloquant avant de considérer une comparaison multi-modèles plus large.

### 16.6. Complète le balayage : F2LLM-v2-160M (taille intermédiaire, jamais testée)

`slurm/pipeline_runs/run_sae_v10_p2_f2llm160m.slurm` (job 41495, terminé en 8min51s
-- Pipeline 2 seul, nouveau `SAVE_DIR` pour éviter la contamination de cache déjà
documentée en §16.5) :

| Métrique | F2LLM-v2-80M | F2LLM-v2-160M | F2LLM-v2-330M |
|---|---|---|---|
| NMSE | 0,0745 | **0,0727** | 0,0689 |
| L0 | 16,14 | 15,94 | 15,94 |
| dead% | 0,63% | 0,77% | 0,70% |
| ρ_SAE | 0,9597 | 0,9531 | 0,9574 |
| silhouette | 0,0212 | 0,0193 | 0,0183 |
| clusters | 4 | 4 | 4 |
| acc_SAE (energy/sports, corpus diffing) | 0,6650-0,6717 | 0,6700 | **0,6867** |
| acc_SAE (axes email, 14 classes) | n/a (bug antérieur au fix) | 0,7685 | 0,7717 |

**Lecture** : NMSE suit une tendance monotone avec la taille (0,0745 → 0,0727 →
0,0689, plus grand = meilleure reconstruction, cohérent), mais ρ_SAE ne suit PAS
cette tendance (160M légèrement en dessous des deux autres tailles) et les deux
métriques de classification placent 160M au même niveau que 80M, pas entre 80M et
330M comme la tendance NMSE le suggérerait. Confirme le constat déjà établi en
§16.5 : **aucune taille de backbone Pipeline 2 ne domine clairement les autres**
sur l'ensemble des métriques -- la taille du backbone d'embedding n'est pas le
facteur limitant de ce pipeline.

### 16.7. bge-m3 comme backbone Pipeline 2 (perspective #10, jamais fait avant ce fix)

bge-m3 est déjà utilisé pour la similarité de labels (§15.2, pooling [CLS]) mais
n'avait jamais été câblé comme backbone d'ENTRAÎNEMENT du `PhraseLevelSAE` --
`extract_f2llm_embeddings` faisait un pooling dernier-token EN DUR, incorrect
pour un backbone encodeur bidirectionnel entraîné pour le pooling [CLS]. Corrigé
via un nouveau `EMB_POOLING` (`src/config.py`, défaut `"last_token"` -- aucun run
F2LLM existant affecté), vérifié sur CPU avant de lancer le job GPU
(`slurm/pipeline_runs/run_sae_v10_p2_bgem3.slurm`, job 41539, terminé en 30min51s).

| Métrique | F2LLM-v2-80M | F2LLM-v2-160M | F2LLM-v2-330M | **bge-m3** |
|---|---|---|---|---|
| NMSE | 0,0745 | 0,0727 | 0,0689 | **0,0559** |
| L0 | 16,14 | 15,94 | 15,94 | 16,20 |
| dead% | 0,63% | 0,77% | 0,70% | **0,13%** |
| ρ_SAE | 0,9597 | 0,9531 | 0,9574 | 0,9303 |
| silhouette | 0,0212 | 0,0193 | 0,0183 | **0,0228** |
| clusters | 4 | 4 | 4 | 4 |
| acc_SAE (energy/sports, corpus diffing) | 0,6650-0,6717 | 0,6700 | 0,6867 | **0,6967** |
| acc_SAE (axes email, 14 classes) | n/a | 0,7685 | 0,7717 | 0,7680 |

**Résultat net, contrairement au balayage F2LLM seul** : bge-m3 domine clairement
sur NMSE (−18,8% vs le meilleur F2LLM, 330M), dead% (4-5x moins de features
mortes), silhouette, et acc_SAE diffing (meilleur des 4 backbones) -- seul
ρ_SAE est légèrement inférieur, et acc_SAE axes email reste dans la même
fourchette que 160M/330M (pas de gain ni de perte notable). Backbone d'embedding
le plus prometteur testé à ce jour pour Pipeline 2, à retenir comme candidat par
défaut pour une suite de stage plutôt que F2LLM à n'importe quelle taille.

## 17. Ablation de mise à l'échelle (v12) : largeur du SAE core, époques, N_FEATURES_TO_LABEL

### 17.0. Correction préalable des largeurs de SAE disponibles pour 12b

Couverture Neuronpedia mesurée empiriquement pour `gemma-3-12b-it`/layer 24, sur les
4 largeurs de SAE core disponibles (`sae_lens.loading.pretrained_saes_directory`) :

| Largeur | Features labellisées | Total | Couverture |
|---|---|---|---|
| 16k (choix initial du projet) | 13 535 | 16 384 | 82,6% |
| **65k** | **57 551** | 65 536 | **87,8%** |
| 262k | 13 851 | 262 144 | 5,3% (confirme le ~10 000 estimé manuellement) |
| 1m | — | — | absent côté Neuronpedia (HTTP 404) |

65k est donc la largeur retenue (meilleure couverture proportionnelle ET ~4,3x plus
de features labellisées en absolu que 16k) — jamais vérifiée spécifiquement pour ce
modèle avant cette passe (seule une couverture ~98% pour 65k était documentée, mais
mesurée sur `gemma-3-270m-it`, un modèle différent). Poids SAE 65k téléchargés via
`download_sae.py --sae-only` (`SAE_ID=layer_24_width_65k_l0_medium`) : absents du
premier essai de lancement du run v12 (jobs 40833/40827 en échec, poids jamais
téléchargés — seuls les LABELS avaient été vérifiés, pas les poids du SAE
lui-même), corrigé avant resoumission (jobs 40844/40845).

### 17.1. Run combiné (`results_v12_scaled_65k`, job 40844+40845+40846-40850)

Trois leviers augmentés SIMULTANÉMENT par rapport au run principal (`results_v10_emails_main`) :

| Paramètre | Run principal | Run v12 | Facteur |
|---|---|---|---|
| Largeur SAE core | 16k | 65k | — |
| `EPOCHS_EXTRA` | 10 | 40 | ×4 |
| `EPOCHS` (P2) | 30 | 100 | ×3,3 |
| `N_FEATURES_TO_LABEL` | 150 | 600 | ×4 |
| Backbone P2 | F2LLM-v2-80M | F2LLM-v2-330M | — |

**Résultats** :

| Métrique | Run principal (16k) | Run v12 (65k, échelle) |
|---|---|---|
| Taux d'interprétabilité (odd-one-out, n features jugées) | 45,3% (68/150) | **53,7% (322/600)** |
| Taux d'interprétabilité, vote majoritaire 5 répétitions | 48,7% | 55,5% |
| Accord moyen entre 5 répétitions (robustesse du juge) | 80,3% | 79,3% (stable) |
| `clf_acc_email_axes` (P1, 14 classes) | 93,5% (`results_v10_ablation_tok100k`) | 91,9% |
| `clf_acc_email_axes` (P2, 14 classes) | 79,3%/77,2% (80M/330M) | 76,7% |
| NMSE P2 | 0,0745 (80M) / 0,0689 (330M) | 0,0667 |
| `fve_pretrained` (P1 core, reconstruction GemmaScope) | non mesuré à ce niveau de détail avant | 0,823 |
| `dead_pct` extension P1 | 0% (3 runs précédents) | 55,9% *(voir §17.4 -- lecture nuancée)* |

### 17.2. Analyse gratuite : le rang par magnitude n'est PAS un bon proxy de l'interprétabilité

Question posée par la hausse du taux global (45,3%→53,7%) avec `N_FEATURES_TO_LABEL`
relevé à 600 : les 450 features supplémentaires jugées sont-elles simplement du
"bruit" qui dilue une statistique, ou apportent-elles un signal réel ? Analyse
gratuite (aucun calcul GPU, relecture de `p1_judge_labels_extended.json`, l'ordre
d'insertion du dict préservant l'ordre de sélection par magnitude décroissante,
`feature_selection_by_magnitude`) :

| Sous-ensemble (rang par magnitude) | n | Interprétables | Taux |
|---|---|---|---|
| Rang 1-150 (équivalent au budget du run principal) | 150 | 66 | **44,0%** |
| Rang 151-600 (features supplémentaires apportées par le scale-up) | 450 | 256 | **56,9%** |
| Total | 600 | 322 | 53,7% |

**Résultat contre-intuitif et important** : le sous-ensemble rang 1-150 obtient un
taux (44,0%) statistiquement indistinguable du run principal à 16k (45,3%, écart
1,3pt << écart-type binomial ≈4,1pt à n=150) — cohérent avec un contrôle correct
(même feature de tête, largeur différente, effet nul comme attendu). Mais les
features de rang 151-600, plus faibles en magnitude, sont en réalité **plus**
interprétables (56,9%) que les features de tête (44,0%). **La magnitude d'activation
moyenne n'est donc pas un proxy fiable de l'interprétabilité** : un budget de
labellisation restreint au sommet du classement par magnitude exclut
systématiquement des features au moins aussi (ici plus) interprétables plus bas dans
le classement. Élargir `N_FEATURES_TO_LABEL` n'est donc pas qu'un gain de puissance
statistique : cela change la composition qualitative de l'échantillon labellisé.

### 17.3. Bug trouvé lors de l'analyse du run v12 : chemin de labels figé sur 16k

Le test de plausibilité (`scripts/explanation_plausibility_test.py`) donnait un
résultat très dégradé sur ce run : **56,7% (34/60)** contre 71,7% (43/60) sur
`results_v10_emails_main` -- une chute surprenante alors que le taux
d'interprétabilité sous-jacent s'améliore (§17.1). Inspection des exemples
sauvegardés (`cache/explanation_plausibility_results.json`) : certains "vrais"
labels affichés au juge comme référence étaient en réalité des transcriptions
BRUTES de raisonnement d'auto-interprétation (plusieurs milliers de caractères,
"**Analysis:** 1. `MAX_ACTIVATING_TOKENS`...") au lieu d'un label court. Cause
racine : `load_label_map()` dans CE script (et 3 autres :
`explanation_fidelity_test.py`, `embedding_model_comparison_test.py`,
`compute_interesting_correlations_retro.py`) charge le fichier de labels Neuronpedia
**16k en dur**, indépendamment du SAE réellement utilisé par le run (ici 65k). Pour
tout index de feature < 16 384, le script associait donc silencieusement le label
d'une feature **16k totalement différente** (le core 65k et le core 16k sont deux
dictionnaires de features indépendants), parfois une des 47 entrées Neuronpedia 16k
elles-mêmes corrompues (0,35% du fichier 16k, contre 0,02% pour le fichier 65k --
un défaut de qualité connu du jeu de données Neuronpedia source, présent aux deux
largeurs). **Corrigé** : les 4 scripts utilisent désormais `NEURONPEDIA_LABELS_PATH`
(dérivé de `SAE_ID`, `src/config.py`) au lieu du chemin figé. Le test de fidélité
n'était affecté que de façon cosmétique (le calcul d'ablation opère par index, pas
par texte de label) ; le test de plausibilité, lui, juge directement le texte du
label -- son résultat a donc été invalidé et recalculé après correction (jobs 40946
plausibilité, 40947 fidélité -- résultat définitif en §17.5).

### 17.4. `dead_pct` P1 à 55,9% : lecture

Métrique brute inquiétante en apparence (0% sur les 3 runs de validation
précédents). Ce chiffre porte sur la plage CORE (65 536 features GemmaScope, jamais
réentraînées) et non sur l'extension (`ExtendedSAE`, la seule dont le taux de mort
est un signal de qualité d'entraînement pertinent -- cf. §5.1, toujours 0% sur ce
run). Un SAE core pré-entraîné généraliste (GemmaScope) a une proportion normale et
attendue de features jamais ou très rarement activées sur un corpus spécifique et
plus restreint (emails EDF) que son corpus d'entraînement d'origine ; ce chiffre
n'était simplement jamais mesuré/reporté à ce grain avant ce run. Non comparable
directement au 0% historique (qui ne concernait que l'extension).

### 17.5. Ablations isolées : décomposition largeur / époques / capacité / N_FEATURES_TO_LABEL

Le run combiné (§17.1) fait varier 4 leviers à la fois (largeur, époques, capacité
implicitement inchangée, `N_FEATURES_TO_LABEL`). Trois runs à facteur UNIQUE
(jobs 40952 largeur seule, 40950 époques seules, 40953 capacité seule), chacun
comparé au run principal (`results_v10_emails_main`, 45,3%) et à la tranche
rang 1-150 du run combiné (44,0%, §17.2) :

| Run | Largeur | Époques (extra/P2) | `D_EXTRA`/`K_EXTRA` | `N_FEAT` | Interprétabilité |
|---|---|---|---|---|---|
| **Run principal** (référence) | 16k | 10/30 | 1024/32 | 150 | **45,3%** (68/150) |
| Largeur seule (job 40952) | 65k | 10/30 | 1024/32 | 150 | 43,3% (65/150) |
| Époques seules (job 40950) | 16k | 40/100 | 1024/32 | 150 | 41,3% (62/150) |
| Capacité seule (job 40953) | 16k | 10 (P1 seul) | 2048/64 | 150 | 40,0% (60/150) |
| Combiné, tranche rang 1-150 (§17.2) | 65k | 40/100 | 1024/32 | 600 (dont 150 analysés) | 44,0% (66/150) |
| **Combiné, total** (§17.1) | 65k | 40/100 | 1024/32 | 600 | **53,7%** (322/600) |

**Lecture** : les trois ablations à facteur unique donnent des taux (43,3% / 41,3% /
40,0%) tous statistiquement indistinguables du run principal (45,3%) compte tenu de
l'écart-type binomial attendu à n=150 (≈4,1 points) -- **aucun des trois leviers pris
isolément (largeur du SAE core, nombre d'époques, capacité de l'extension) n'améliore
le taux d'interprétabilité**, cohérent avec le résultat déjà établi pour le volume de
tokens (§5.2) : une fois le corpus corrigé, ce protocole d'évaluation semble plafonné
par autre chose que ces paramètres d'échelle. La hausse observée sur le run combiné
(53,7%) s'explique donc presque entièrement par l'effet de composition déjà identifié
au §17.2 (les features de rang 151-600 sont mieux interprétées que celles de rang
1-150), **pas** par une meilleure qualité d'entraînement du SAE due à la largeur, aux
époques ou à la capacité. Conclusion pour la question posée en tête de ce chapitre
("scaler la pipeline améliore-t-il les résultats ?") : **oui pour la plausibilité de
l'explication (§17.3, +16,6 points après correction du bug) et pour le volume/la
richesse du catalogue de features labellisées, mais pas pour le taux brut
d'interprétabilité odd-one-out**, qui reste gouverné par le domaine du corpus (§3-5)
et par le protocole de jugement lui-même (§7.1), pas par le volume d'aucun des
paramètres testés à ce jour (tokens, largeur, époques, capacité).

### 17.6. Résultat définitif : plausibilité après correction du bug de labels

Une fois le bug §17.3 corrigé (labels dérivés de `SAE_ID` au lieu du chemin 16k figé),
les tests de fidélité et de plausibilité ont été recalculés sur `results_v12_scaled_65k`
(jobs 40947 fidélité, 40946 plausibilité) :

| Métrique | Run principal (v10, 16k) | Run v12 buggé (16k chargé à tort) | **Run v12 corrigé (65k)** |
|---|---|---|---|
| Plausibilité (choix forcé, 60 documents) | 71,7% (43/60) | 56,7% (34/60) | **88,3% (53/60)** |
| Fidélité (ratio top/random, moyenne 4 intentions) | 250×-576 000× | *(non affecté, cosmétique)* | 251×-32 992× (même ordre de grandeur) |

**La plausibilité progresse réellement et fortement** une fois le bug corrigé
(71,7%→88,3%, +16,6 points) — contrairement au taux d'interprétabilité odd-one-out
(§17.5, stable), la plausibilité bénéficie directement d'un catalogue de features
labellisées beaucoup plus riche pour construire les ensembles "réels" présentés au
juge (65k core à 87,8% de couverture + 600 features d'extension jugées, contre 16k à
82,6% + 150 seulement) : la feature réellement la plus active d'un document a
beaucoup plus de chances de disposer d'un label exploitable, donc d'un ensemble
"réel" complet et cohérent à opposer au leurre. La fidélité, elle, reste dans le même
ordre de grandeur (aucune conclusion de changement, les deux mesures restant très
supérieures au hasard) : cohérent avec le fait qu'elle mesure une propriété causale
du classifieur indépendante du volume de labels disponibles.

## 18. Relecture littérature (session pdf/) : `FrozenCoreResidualSAE` est une implémentation de SAE Boost

Lecture de `pdf/teacholdsaes.pdf` (*Teach Old SAEs New Domain Tricks with Boosting*,
Koriagin et al., COLM 2025) — quasi certainement la référence "SAE Boost" du cadrage
initial du stage (`Context.md`, objectif n°4, marquée "non fait" dans
`docs/references.md` depuis le début du stage).

### 18.1. Constat : architecture identique, jamais identifiée comme telle

Leur méthode ("SAE Boost") : un SAE secondaire, entraîné à reconstruire le résidu
`e = x - x̂` d'un SAE core **gelé**, sommé au SAE core à l'inférence
(`x ≈ x̂ + ê`). C'est exactement `FrozenCoreResidualSAE`/`ExtendedSAE`
(`src/sae/frozen_core.py`) : `core_sae` gelé (`requires_grad_(False)`), branche
"extra" entraînée sur `residual = x - core_out`, sortie `core_out + extra_out`.
Coïncidence notable : leur dictionnaire résiduel fait 1024 features dans toutes
leurs expériences — exactement notre `D_EXTRA` par défaut. Le projet a donc, sans
le savoir/le documenter, déjà implémenté et validé à l'échelle la méthode listée
comme objectif optionnel du cadrage initial. Corrigé dans `docs/references.md`
et `report/01_etat_de_lart.md`.

### 18.2. Écart de sensibilité `K_EXTRA` (top-k du résiduel)

Leur étude de sensibilité (Table 12 du papier) teste top-k ∈ {5, 10, 20, 50} pour
le SAE résiduel : k=5 est retenu comme meilleur compromis interprétabilité/EV
domaine (k plus élevé améliore légèrement l'EV domaine mais dégrade la parcimonie/
interprétabilité). Notre `K_EXTRA` par défaut est **32**, jamais testé en dessous
de cette valeur (notre ablation capacité, §17.5, a testé K_EXTRA=64, dans la
direction opposée). Piste non testée à ce jour : un `K_EXTRA` plus faible
(5-10) pourrait améliorer le taux d'interprétabilité odd-one-out à budget
d'entraînement égal.

### 18.3. Écart critique : budget de tokens du SAE résiduel

Leur Figure 4 montre qu'un SAE résiduel entraîné sur **moins de 100M tokens**
dégrade la performance en domaine général de jusqu'à **-31% d'EV** ; la
convergence sans dégradation nécessite de dépasser ~200M tokens. Notre ablation de
volume (§5.2/§12, `N_TOKENS_EXTRA_TRAIN` ∈ {100k, 500k, 2M}) reste **50 à 100 fois**
en dessous de ce seuil. La conclusion tirée dans ce projet ("le volume de tokens ne
change rien au-delà de 100k, une fois le domaine corrigé") est donc établie
uniquement dans un régime que leur étude qualifie explicitement d'insuffisant pour
observer un effet de convergence — **elle ne peut pas être extrapolée** au régime
100M-200M sans le tester directement. Aucun run à ce volume n'a été lancé dans ce
stage (coût GPU largement supérieur aux runs existants : leur budget est calibré
sur un unique domaine à la fois, quand ce projet réextrait aussi les activations
Gemma-3-12B à chaque configuration). Reste une limite explicitement documentée
(`report/04_limites_et_perspectives.md`) plutôt qu'un résultat.

### 18.4. Baselines alternatives jamais comparées

Le papier compare SAE Boost à quatre alternatives de domain adaptation :
Extended SAE (init sur features les plus actives), Extended SAE (init aléatoire),
SAE Stitching (fine-tuning complet + greffe des features les plus changées), et
full fine-tuning. Leurs résultats : SAE Boost (= notre architecture) offre le
meilleur compromis performance domaine/généraliste ; le fine-tuning complet
souffre d'oubli catastrophique (EV générale -28% à -36% selon le domaine) ; les
approches "Extended SAE" sont compétitives mais moins efficientes en parcimonie.
Ce projet n'a jamais comparé son choix architectural (`FrozenCoreResidualSAE`) à
ces alternatives plus simples sur SON corpus — le choix a toujours reposé sur la
littérature générale (Context.md, règle n°3 : "Conserver FrozenCoreResidualSAE"),
jamais sur un test empirique comparatif propre à ce projet. Piste de poursuite.

## 19. Sanity check (Korznikov et al. 2026) : le taux d'interprétabilité bat-il un décodeur aléatoire ?

Lecture de `pdf/sanitychecks.pdf` (*Sanity Checks for Sparse Autoencoders: Do SAEs
Beat Random Baselines?*, Korznikov et al., 2026, preprint). Résultat central du
papier : sur des SAE conventionnels (BatchTopK, JumpReLU, ReLU), une baseline
**"Frozen Decoder"** (décodeur figé à une initialisation aléatoire, jamais
entraîné — seul l'encodeur apprend à projeter sur ces directions fixes) égale un
SAE réellement entraîné sur interprétabilité automatique (AutoInterp, 0,87 vs
0,90), sparse probing (0,69 vs 0,72) et édition causale RAVEL (0,73 vs 0,72).
Leur conclusion : ces métriques, prises isolément, ne suffisent pas à prouver
qu'un SAE a appris une décomposition en features réellement significative — un
ajustement de l'encodeur seul à des directions arbitraires peut produire les
mêmes scores.

### 19.1. Protocole appliqué à ce projet

Nouvelle classe `FrozenDecoderExtendedSAE` (`src/sae/frozen_core.py`), sous-classe
DIRECTE de `FrozenCoreResidualSAE` (pas d'`ExtendedSAE`, dont l'initialisation PCA
sur le résidu serait déjà informée par les données — le test doit partir d'un
décodeur purement aléatoire, fidèle à leur protocole) : `W_dec_extra.requires_grad_
(False)` + `normalize_decoder()` neutralisée (no-op, pour éviter toute dérive
flottante cumulative du décodeur "figé" sur des milliers de pas -- vérifié par un
test unitaire dédié, `tests/test_frozen_core.py::test_frozen_decoder_stays_frozen_during_training`,
qui entraîne 5 pas et vérifie `W_dec_extra` bit-à-bit inchangé tandis que
`W_enc_extra` bouge normalement). Sélectionnable via
`SANITY_CHECK_FROZEN_DECODER=1` (`src/config.py`), sans toucher au pipeline de
production (défaut `0`).

Run `results_v12_sanity_frozen_decoder` (job 41060,
`slurm/analysis/run_sanity_check_frozen_decoder.slurm`) : toutes conditions
IDENTIQUES au run principal (`results_v10_emails_main` — SAE_ID=16k,
`N_FEATURES_TO_LABEL=150`, `EPOCHS_EXTRA=10`, `D_EXTRA=1024`/`K_EXTRA=32`, même
corpus), seule la classe change (`FrozenDecoderExtendedSAE` au lieu
d'`ExtendedSAE`) — comparaison à facteur unique directe.

### 19.2. Résultat (job 41082, COMPLETED)

| Métrique | Run principal (`ExtendedSAE`, décodeur entraîné) | Frozen Decoder (décodeur figé aléatoire) | Écart | Significativité |
|---|---|---|---|---|
| Interprétabilité odd-one-out (n=150) | 45,3% (68/150) | **29,3%** (44/150) | −16,0 points | z=2,91 (p<0,01) |
| `clf_acc_email_axes` (14 classes, n=2177 test) | 93,5% | **91,2%** | −2,3 points | z=2,86 (p<0,01) |
| Features mortes (extension) | 0 | 0 | — | — |

**Lecture — résultat nuancé, ni réplication à l'identique du papier, ni réfutation
complète** :

- **Interprétabilité odd-one-out : écart réel et substantiel** (16 points, très
  au-delà de l'écart-type binomial ≈4,1pt à n=150, z=2,91). Contrairement au
  résultat du papier (leur baseline "Frozen Decoder" égalait leur SAE entraîné sur
  AutoInterp, 0,87 vs 0,90), **notre protocole odd-one-out distingue clairement un
  décodeur entraîné d'un décodeur aléatoire** sur ce projet -- un signal rassurant
  que l'entraînement de `ExtendedSAE` apprend une structure réelle, pas seulement un
  ajustement de l'encodeur à des directions arbitraires. Le décodeur figé reste
  cependant loin d'être inerte (29,3%, bien au-dessus d'un score nul), signe que la
  seule diversité de 1024 directions aléatoires suffit à produire un nombre non
  négligeable de "features" fortuitement monosémantiques par pur volume
  combinatoire -- cohérent avec l'argument du papier sur le rôle du hasard à grande
  échelle, mais ne suffisant pas à expliquer tout le taux mesuré.
- **Classification en aval : écart réel mais faible en proportion du signal total**
  (2,3 points, statistiquement significatif à n=2177 mais très petit dans l'absolu).
  **Ce volet réplique largement le constat du papier pour le sparse probing** : un
  décodeur aléatoire de 1024 directions capture déjà 91,2% de précision de
  classification -- la quasi-totalité du signal exploité par la sonde en aval ne
  dépend donc pas d'un apprentissage réel du décodeur, seuls les 2,3 derniers points
  y sont attribuables. Cohérent avec l'hypothèse du papier : à haute dimension, un
  nombre suffisant de directions aléatoires corrèle déjà avec la plupart des
  concepts par hasard, rendant le sparse probing peu discriminant pour juger de la
  qualité d'un SAE.

**Conclusion retenue pour ce projet** : le protocole d'auto-interprétation
odd-one-out (métrique centrale du chapitre 3) résiste bien au sanity check --
l'écart avec un décodeur aléatoire est net et significatif. La sonde de
classification en aval (`clf_acc_sae`/`clf_acc_email_axes`), en revanche, doit être
interprétée avec prudence : un score élevé n'est qu'une preuve faible d'un
apprentissage de features réellement significatif, la majorité du signal étant déjà
disponible avec un décodeur aléatoire de taille comparable. Ceci nuance
(sans l'invalider) le résultat de séparabilité linéaire du chapitre 3 §5.4 (93,5%
Pipeline 1) : la valeur ajoutée de l'entraînement pour CETTE métrique spécifique est
modeste, même si les métriques d'auto-interprétation et de fidélité/plausibilité
(chapitre 3 §8, non retestées ici mais causales par construction) restent, elles,
des preuves plus solides de features significatives.

## 20. Erreur de nettoyage disque : suppression d'un lien symbolique confondu avec un doublon

Lors du nettoyage du dépôt (réorganisation `slurm/`/`logs/`, cf. commits précédents),
le dossier racine `saes/` (30 Go) a été identifié comme un "doublon legacy" de
`local_data/saes/` (ancienne convention de nommage `-res`, cf. `Context.md`) et
supprimé après confirmation utilisateur. **Erreur** : `local_data/saes/
gemma-scope-2-12b-it` n'était pas un dossier réel mais un **lien symbolique**
pointant vers `saes/gemma-scope-2-12b-it-res` — la suppression du dossier racine a
donc supprimé la seule copie physique des poids SAE GemmaScope (16k/65k/262k),
laissant un lien symbolique cassé. Non détecté avant de relancer un nouveau job
(`results_v12_sanity_frozen_decoder`, job 41060) : `ls -la` sur le dossier parent
n'avait pas été utilisé pour vérifier si l'entrée `gemma-scope-2-12b-it` était un
lien avant suppression — seul l'espace disque et le nom du dossier ("legacy",
suffixe `-res`) avaient motivé la décision.

**Impact réel** : nul sur les résultats déjà produits (tous les runs `results_v12_*`
avaient déjà terminé et leurs artefacts — activations, checkpoints, labels — étaient
déjà écrits sur disque indépendamment des poids SAE sources). Le seul job affecté
était `results_v12_sanity_frozen_decoder` (échec immédiat, `ValueError` au
chargement du SAE, cf. log `logs/analysis/sanity_check_frozen_decoder_41060.log`).

**Corrigé** : lien symbolique cassé supprimé, poids des 3 largeurs (16k/65k/262k)
retéléchargés via `download_sae.py --sae-only` directement vers le chemin canonique
`local_data/saes/gemma-scope-2-12b-it` (dossier réel désormais, plus de lien
symbolique vers un autre chemin — élimine la source de confusion pour l'avenir).
Job relancé (41082).

**Leçon retenue** : avant toute suppression présentée comme un "nettoyage de
doublon", vérifier avec `ls -la`/`readlink` si le chemin candidat est un lien
symbolique référencé ailleurs, pas seulement sa taille ou son nom -- particulièrement
quand deux chemins portent des noms proches (`saes/` vs `local_data/saes/`,
`gemma-scope-2-12b-it` vs `gemma-scope-2-12b-it-res`).

## 21. Ablation de variance de seed (Unstable Features, Reproducible Subspaces)

Question posée par *Unstable Features, Reproducible Subspaces* (arXiv:2606.12138)
et *Toward Identifiable Sparse Autoencoders* (arXiv:2605.31245) : les features
individuelles d'un SAE varient-elles substantiellement d'un seed d'entraînement à
l'autre, même quand tout le reste (corpus, split, hyperparamètres) est identique ?
Toutes les ablations précédentes de ce projet utilisent `SEED=42` -- cet axe n'avait
jamais été testé. `SEED` a été découplé du split train/test du corpus
(`CORPUS_SPLIT_SEED`, nouveau, `src/config.py`, défaut 42 inchangé) pour isoler
proprement la variance d'entraînement du SAE (init des poids, échantillonnage de
`feature_selection_by_magnitude`) de toute variance de corpus. Run
`results_v13_ablation_seed123` (job 41118) : `SEED=123`, `CORPUS_SPLIT_SEED=42`
(inchangé), toutes les autres conditions identiques au run principal (16k, 150
features, `EPOCHS_EXTRA=10`, `D_EXTRA=1024`/`K_EXTRA=32`).

| Métrique | SEED=42 (run principal) | SEED=123 | Écart |
|---|---|---|---|
| Interprétabilité odd-one-out | 45,3% (68/150) | 47,3% (71/150) | +2,0 points (z=0,35, non significatif) |
| `clf_acc_email_axes` | 93,5% | 91,3% | −2,2 points |
| Recouvrement EXACT des labels interprétables (chaîne de caractères) | — | **22/78 = 28,2%** | |

**Lecture** : au niveau AGRÉGÉ, le taux d'interprétabilité est parfaitement stable
d'un seed à l'autre (45,3% vs 47,3%, écart non significatif) -- rassurant, ce
n'est pas un artefact d'un seed particulièrement favorable. En revanche, au niveau
INDIVIDUEL, seulement 28,2% des labels obtenus sont des chaînes de caractères
identiques entre les deux seeds -- confirmant précisément le résultat de la
littérature (*Unstable Features, Reproducible Subspaces*) : les features
individuelles ne sont PAS reproductibles à l'identique, seule la performance
agrégée et la thématique générale le sont (les deux seeds recouvrent des thèmes
similaires -- adresses, contrats énergie, réclamations, coupures, urgence -- mais
rarement avec le même libellé exact ni la même sélection de features). **Implication
pour la lecture de ce rapport** : toute feature individuelle citée comme exemple
(chapitre 3) doit être comprise comme un exemple représentatif d'une catégorie de
concepts récurrente, pas comme un atome stable et reproductible du dictionnaire —
seul le taux agrégé d'interprétabilité constitue une mesure fiable de la qualité
du SAE.

## 22. Test de biais multilingue du juge (FR original vs EN traduit)

Question posée par la littérature sur l'interprétabilité multilingue (Resck et al.
2025 ; *Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse
Languages*, arXiv:2507.11230) : le juge (Gemma-3-12B-it) interprète-t-il MIEUX les
mêmes features quand les exemples et le prompt sont en anglais plutôt qu'en
français (hypothèse : représentation interne des concepts dominée par l'anglais) ?
`scripts/multilingual_judge_bias_test.py` (nouveau, job 41119) : réutilise les 150
features déjà jugées de `results_v10_emails_main` (mêmes activations/fragments,
aucune réextraction), traduit les exemples odd-one-out en anglais (un appel Gemma-3
par feature, JSON in/JSON out, marqueurs `<<mot>>` préservés) puis rejoue le
protocole odd-one-out intégralement en anglais (prompt traduit aussi).

| Métrique | Français (original) | Anglais (traduit) | Écart |
|---|---|---|---|
| Interprétabilité odd-one-out (n=145, 5 échecs de traduction exclus) | 46,9% (68/145) | 45,5% (66/145) | −1,4 point (z=0,24, non significatif) |
| Features changeant de statut FR→EN | — | 27 (non-interprétable→interprétable) | |
| Features changeant de statut EN→FR | — | 29 (interprétable→non-interprétable) | |
| **Total features changeant de statut** | — | **56/145 = 38,6%** | |

**Lecture — résultat nul sur l'hypothèse testée, mais confirmation d'un bruit de
décision substantiel** : aucune différence significative entre le taux
d'interprétabilité en français et en anglais traduit (46,9% vs 45,5%, écart bien en
deçà du bruit binomial attendu) -- **pas de preuve d'un biais systématique
favorisant l'anglais** sur ce protocole et ce corpus. Cependant, 38,6% des features
changent de statut individuellement selon la langue de présentation -- un taux de
retournement même supérieur à celui déjà mesuré pour le simple réordonnancement des
exemples (§13.1, 31,3%). Puisque les retournements sont globalement symétriques
(27 dans un sens, 29 dans l'autre) plutôt que systématiquement orientés vers
l'anglais, l'interprétation la plus probable est que la traduction introduit son
propre bruit de perturbation (comparable en ampleur à un simple réordonnancement),
pas un déficit structurel de l'auto-interprétation en français. **Limite assumée** :
le test traduit les exemples via LE MÊME modèle juge (pas de réentraînement sur
corpus anglais natif, qui testerait une hypothèses différente -- cf.
`docs/references.md`) ; la traduction elle-même peut introduire des artefacts
(perte de nuance, changement de longueur) indépendants de la question posée.
**Conclusion retenue** : renforce (sans le remettre en cause) le résultat déjà
établi en §13.1 -- le protocole odd-one-out à décision greedy unique reste bruyant
face à toute perturbation de surface (ordre OU langue), justifiant l'adoption du
vote majoritaire comme protocole par défaut plutôt qu'une preuve d'un problème
spécifiquement multilingue.

## 23. Ablation volume à grande échelle (~100-120M tokens, SAE Boost)

Suite directe du §18.3 : le papier de référence (SAE Boost, Koriagin et al., COLM
2025) montre qu'un SAE résiduel a besoin de 100-200M tokens pour converger sans
dégrader la performance générale — 50 à 100x au-dessus du volume testé dans notre
ablation initiale (§5/§12, jusqu'à 2M). Ce chapitre documente l'investigation de
faisabilité et le run lancé pour tester directement ce seuil sur ce projet.

### 23.1. Recherche d'une source de données française pour le volume manquant

Le corpus emails+augmentés (~6M tokens) est très en dessous de 100M. Options
étudiées :

- **SignalConso** (réclamations consommateurs officielles françaises, DGCCRF,
  data.gouv.fr, ~500k signalements) : écarté après vérification empirique directe
  du schéma de l'export public (`data.economie.gouv.fr/api/explore/v2.1/catalog/
  datasets/signalconso`) — 15 champs, tous catégoriels/géographiques
  (`category`, `subcategories`, `tags`, `dep_name`, etc.), **aucun champ de texte
  libre** contenant la réclamation rédigée par le consommateur (anonymisation
  RGPD probable). Inutilisable comme source de texte.
- **FineWeb2-fr filtré par mots-clés**, retenu, avec trois configurations testées
  empiriquement (échantillon de 200-300k documents du shard local
  `000_00000.parquet`, `datasets.load_dataset(..., streaming=True)`) :

| Configuration de mots-clés | Hit rate | Précision qualitative (échantillon manuel) |
|---|---|---|
| Union `ENERGY_KEYWORDS` + `SUPPORT_KEYWORDS` (déjà existants) | 70,8% | Quasi nulle -- mots trop génériques ("bonjour", "urgence", "cordialement") matchent presque toute page web |
| Phrases composées spécifiques (`UTILITY_COMPLAINT_KEYWORDS`, nouveau) | 0,275% | ~15-20% -- majorité de pages e-commerce/télécom génériques partageant le registre "réclamation/résiliation" sans rapport avec l'énergie |
| Ancre énergie (EDF/Enedis/kWh) ET terme de plainte (substring court) | 3,7% | Quasi nulle -- "edf"/"eni" comme substrings courts matchent des mots sans rapport (page mémorial Auschwitz, biographie du XIXe, forum Ubuntu, extrait de roman) ; vocabulaire juridique générique ("litige", "contentieux", "risque") traverse tous les secteurs |

**Conclusion retenue** : aucune configuration de mots-clés ne donne un corpus
"réclamations énergie" pur sur un corpus web générique français -- le registre
"client mécontent" est partagé par de nombreux secteurs (télécom, e-commerce,
assurance), pas spécifique à l'énergie. `UTILITY_COMPLAINT_KEYWORDS`
(`src/data/keywords.py`, 33 phrases composées) retenu comme meilleur compromis
disponible, avec sa limite de précision documentée explicitement plutôt que
présentée comme un filtre propre.

### 23.2. Protocole du run

`run_llm_max_pool_pipeline` (`src/sae/saev5.py`) étendu avec un paramètre
`volume_filler_texts` (optionnel, défaut `None` → comportement 100% inchangé pour
tous les runs existants) : le filler est ajouté **uniquement** au réservoir de
tokens résiduels (échantillonnage par réservoir de Vitter, déjà en place), jamais à
`train_texts` lui-même — la sélection des features à labelliser
(`feature_selection_by_magnitude`, restreinte à `range(n_train)`) et la sonde de
classification email restent calculées sur les emails+augmentés SEULS, pour ne pas
réintroduire le biais de domaine diagnostiqué et corrigé au §12.

Run `results_v13_ablation_volume100m` (job 41176,
`slurm/pipeline_runs/run_ablation_volume_100m.slurm`) : 3 shards FineWeb2-fr locaux
(`000_00000/1/2.parquet`, ~14,5 Go, 2 shards supplémentaires téléchargés pour ce
run — `HuggingFaceFW/fineweb-2`, config `fra_Latn`, 135 shards disponibles au
total), `N_VOLUME_FILLER_TARGET_CHUNKS=540000` (~114M tokens estimés),
`N_TOKENS_EXTRA_TRAIN=100000000`. Sinon identique au run principal (16k, 150
features jugées, `EPOCHS_EXTRA=10`, `D_EXTRA=1024`/`K_EXTRA=32`) — Pipeline 1
seul (le levier testé, SAE résiduel, n'existe pas pour le `PhraseLevelSAE`
from-scratch de Pipeline 2). Coût attendu multi-jours (extraction Gemma-3-12B sur
~584 000 textes, ~13x le volume du run principal) — job soumis avec
`--time=120:00:00`.

### 23.3. Incident : OOM du run à 100M tokens, correction et relance à 25M

Job 41176 tué par le gestionnaire OOM du nœud après 2h39 d'exécution, à 24% de
l'extraction (19 720/82 643 batches, MaxRSS 187,5 Go pour un `--mem=180G` demandé
— `sacct -j 41176` : état `OUT_OF_MEMORY`).

**Diagnostic** : la cause n'est pas un sous-dimensionnement anodin de `--mem`, mais
un problème de conception à cette échelle. Le réservoir de résidus bruts
(`raw_residuals_list`/`reservoir`, `saev5.py` ~L801-896, échantillonnage de Vitter)
alloue en RAM **hôte** (pas VRAM) un buffer de taille
`N_TOKENS_EXTRA_TRAIN × hidden_size × 2 octets` (bf16), indépendamment de la taille
du corpus ou du filler. Pour Gemma-3-12B (`hidden_size=3840`) et
`N_TOKENS_EXTRA_TRAIN=100000000`, cela représente **768 Go** pour ce seul tenseur —
largement au-delà des 180 Go demandés, et proche de la RAM totale du nœud a100
(1 To, partagé avec d'autres jobs GPU). Le buffer double transitoirement
(`raw_residuals_list` + `reservoir` coexistent brièvement au moment de la création
du réservoir, L880-882), aggravant le pic mémoire réel au-delà même de 768 Go.

Point notable : le run était par ailleurs sain — l'extraction progressait
normalement (~3,5 batches/s, ~14 docs/s), le filler FineWeb2-fr avait produit
286 316 chunks (sur les 540 000 visés, faute de matches suffisants sur les 3 shards
locaux — hit rate mesuré de 0,275%, cf. §23.1), et à ce débit l'extraction complète
du corpus (~330 000 textes au total) aurait pris ~6-7h, bien en-deçà du
`--time=120:00:00` alloué. Seule la mémoire était en cause.

**Options considérées** : (a) réduire `N_TOKENS_EXTRA_TRAIN` pour rester en RAM à
un volume plus modeste, (b) pousser à ~90M tokens avec `--mem=900G` (le nœud,
quasi inactif au moment du diagnostic, le permettait en théorie, avec un risque de
contention si d'autres jobs démarrent sur le même nœud partagé), (c) ré-architecturer
le réservoir en memmap disque pour atteindre malgré tout 100-200M tokens sans
changer le compromis mémoire. Option (a) retenue pour sa simplicité et son
absence de risque de contention : `N_TOKENS_EXTRA_TRAIN=25000000` (buffer ≈ 192 Go),
`--mem=500G` (marge confortable sur un nœud à 1 To). Ce choix ne permet pas
d'atteindre le seuil exact de 100-200M du papier SAE Boost, mais teste tout de même
un volume ~12x supérieur à l'ablation initiale (2M tokens, §5/§12) — suffisant pour
détecter un éventuel effet de volume s'il se manifeste de façon monotone entre 2M
et 100M.

Relancé sous `results_v13_ablation_volume25m`
(`slurm/pipeline_runs/run_ablation_volume_25m.slurm`, job 41375), mêmes 3 shards
FineWeb2-fr, `N_VOLUME_FILLER_TARGET_CHUNKS=540000` inchangé (la RAM n'en dépend
plus, seul le plafond `N_TOKENS_EXTRA_TRAIN` compte désormais).

### 23.3. Résultats (job 41375, terminé en 18h17min21s)

| Run | `N_TOKENS_EXTRA_TRAIN` | Filler FineWeb2-fr | Taux interp. (odd-one-out) | `clf_acc_email_axes` | `fve_pretrained` |
|---|---|---|---|---|---|
| Run principal, `results_v10_emails_main` | 500 000 | Non | **68/150 = 45,3%** | 93,5% | — |
| Ablation initiale (2M), job 39662 | 2 000 000 | Non | 67/150 = 44,7% | — | — |
| **Ce run**, `results_v13_ablation_volume25m`, job 41375 | 25 000 000 | Oui (~114M tokens visés, 3 shards) | **81/150 = 54,0%** | 91,3% | 0,831 |

Comparaison statistique au run principal (test z sur deux proportions,
n=150 de part et d'autre) : z=-1,50, **non significatif** au seuil conventionnel
(\|z\|>1,96) malgré un écart numérique de +8,7 points. `clf_acc_email_axes` recule
légèrement (93,5% → 91,3%, comme systématiquement observé dans les ablations qui
ajoutent du contenu hors du bloc emails+augmentés au réservoir résiduel — cf.
§17.5).

**Lecture** : le volume porté à 25M tokens (12x l'ablation initiale, toujours
50-100x en dessous du seuil 100-200M du papier SAE Boost) ne change pas la
conclusion qualitative de §5/§12 — le taux d'interprétabilité reste dans une
fourchette statistiquement indiscernable du run principal. L'écart numérique
(+8,7 points, non significatif seul) va dans le **même sens et est du même ordre
de grandeur** que celui observé indépendamment pour l'ablation K_EXTRA=5 (+9,4
points, §25, également non significatif seule) — une coïncidence entre deux
ablations testant des leviers complètement différents (volume de tokens vs
capacité/parcimonie de l'extension) serait surprenante, mais rester prudent :
deux résultats non significatifs qui pointent dans la même direction ne
constituent pas une preuve combinée sans un test dédié (ex. répliquer sur
plusieurs seeds). Piste à explorer si le temps du stage le permet : une
réplication à seed multiple pour trancher si cet écart directionnel reflète un
effet réel de petite taille ou une coïncidence entre deux ablations.

Aucun signal de dégradation liée au filler générique (FineWeb2-fr, ~15-20% de
précision qualitative sur le thème énergie, cf. §23.1) : le taux d'interprétabilité
et `clf_acc_email_axes` restent dans la même fourchette que les runs sans filler,
cohérent avec le fait que le filler n'alimente que le réservoir de tokens résiduels
et jamais `train_texts`/la sélection de features (cf. §23.2) — le garde-fou conçu
pour éviter de réintroduire le biais de domaine original semble avoir fonctionné
comme prévu.

## 24. Fidélité du steering (`steer_and_decode`) : jamais testé, résultat très hétérogène par intention

`steer_activations`/`steer_and_decode` (`src/sae/sae_shared.py`) existent dans le
dépôt depuis le début mais n'étaient jamais réellement exercés : seule
`run_steering_demo` (`saev5.py`) les utilise, et uniquement pour une vérification
géométrique superficielle (cosinus avant/après suppression/amplification d'UNE
feature, sans tâche en aval) — signalé comme piste non exploitée dans
`docs/references.md` (entrée "A Survey on Sparse Autoencoders"). `explanation_
fidelity_test.py` (§16) ablate déjà des features par intention, mais **directement
dans l'espace des codes SAE, sans jamais appeler `decode()`**.

**Question testée** : si on utilise réellement `steer_and_decode` — décoder le code
stimulé vers l'espace résidu, puis RÉ-ENCODER ce résidu décodé — l'intervention
(suppression des top-10 features explicatives d'une intention) tient-elle à travers
cet aller-retour, ou le décodeur/encodeur du SAE la dilue-t-il ?

**Limite méthodologique assumée** : les vecteurs utilisés
(`p1_all_doc_acts_ext_d1024.pt`) sont des codes SAE poolés par MAX sur tous les
tokens d'un document (comme dans `run_steering_demo` déjà) — pas le code d'un token
réel. Décoder un pooling ne reconstruit donc pas un résidu de token authentique,
mais une direction résidu synthétique représentant le mélange de concepts du
document.

**Protocole** (`scripts/steering_fidelity_test.py`, zéro calcul LLM — réutilise les
activations en cache + le checkpoint `p1_frozen_core_d1024_k32.pt` déjà entraîné de
`results_v10_emails_main`) : pour chaque intention testée en §16, sur les mêmes
documents/top-10 features explicatives, compare la chute de probabilité prédite (a)
par ablation en place dans l'espace des codes (témoin, identique à §16) et (b) par
`steer_and_decode` (decode → `ext_sae.encode()` → re-score), ainsi que la fraction
d'activation résiduelle des features "supprimées" après l'aller-retour.

| Intention | Chute en place (témoin) | Chute steer_and_decode | Ratio | Fuite résiduelle moyenne |
|---|---|---|---|---|
| réclamation | 0,576 | 1,000 | **1,74×** | 0,213 |
| remboursement | 1,000 | 0,016 | **0,02×** | 0,049 |
| information | 1,000 | 0,004 | **0,00×** | 0,246 |
| urgence | 0,646 | 0,584 | 0,90× | 0,455 |

**Résultat, hétérogène et contre-intuitif** : le comportement du round-trip
decode/encode varie du tout au tout selon l'intention — quasi neutralisé pour
`remboursement`/`information` (ratio 0,00-0,02×, l'intervention ne "tient" pas du
tout : le décodeur puis ré-encodeur régénère une prédiction quasiment identique à
l'original malgré la suppression), globalement préservé pour `urgence` (0,90×), et
même **amplifié** pour `réclamation` (1,74×, la chute de probabilité est plus forte
qu'en ablation directe). La "fuite résiduelle" (fraction de l'activation d'origine
des features ciblées qui réapparaît après le round-trip) reste partout < 0,5, sans
corrélation évidente avec le ratio de chute de probabilité — suggérant que ce n'est
pas la feature ciblée elle-même qui "revient", mais que d'autres features (actives
et corrélées) compensent différemment selon l'intention lors de la reconstruction
puis du ré-encodage.

**Conclusion** : `steer_and_decode`, tel qu'il existe dans le dépôt, n'est **pas**
un mécanisme d'intervention causale fiable et prévisible à partir du simple test
d'ablation en place (§16) — son effet dépend fortement de la structure de
corrélation entre features propre à chaque intention, et peut aussi bien annuler
que renforcer l'intervention voulue. Toute utilisation future du steering comme
méthode d'explication devrait mesurer cet effet par intention/concept plutôt que de
supposer qu'une ablation en place et un steering décodé sont équivalents.

## 25. Ablation `K_EXTRA=5` (SAE Boost, piste flaguée non testée en §18.2)

Le papier SAE Boost trouve k=5 optimal dans son étude de sensibilité pour un SAE
résiduel — notre `K_EXTRA=32` par défaut n'avait jamais été testé en dessous de
cette valeur (seulement testé plus haut, k=64, §17.5, direction opposée). Run
`results_v13_ablation_k_extra5` (job 41404, terminé en 3h51min20s, pipeline
complet rejoué depuis un `SAVE_DIR` neuf — mêmes conditions que le run principal
sauf `K_EXTRA=5`, cf. `slurm/pipeline_runs/run_ablation_k_extra5.slurm`).

| Run | `K_EXTRA` | Taux interp. (odd-one-out) | `clf_acc_email_axes` | `rho_sae` |
|---|---|---|---|---|
| Run principal, `results_v10_emails_main` | 32 | **68/150 = 45,3%** | 93,5% | 0,922 |
| Ablation capacité, §17.5 | 64 | — | — | — |
| **Ce run**, `results_v13_ablation_k_extra5`, job 41404 | 5 | **82/150 = 54,7%** | 91,2% | 0,849 |

Comparaison statistique au run principal : z=-1,62, **non significatif** au seuil
conventionnel (\|z\|>1,96) malgré un écart numérique de +9,4 points — le plus
proche du seuil de significativité (p≈0,106 bilatéral) de toutes les ablations
testées dans ce chapitre. `rho_sae` (proxy de fidélité de reconstruction du
résidu) recule sensiblement (0,922 → 0,849), cohérent avec le fait qu'un budget de
capacité par token beaucoup plus faible (k=5 vs 32) réduit mécaniquement la
fraction du résidu reconstructible — attendu, et conforme à la logique du papier
(k plus faible = code plus parcimonieux, quitte à moins bien reconstruire, en
échange d'une meilleure interprétabilité par feature active).

**Lecture** : direction cohérente avec l'hypothèse du papier (k plus faible →
meilleure interprétabilité), mais non significatif isolément sur n=150. Voir
§23.3 pour la discussion de la coïncidence directionnelle avec l'ablation volume
(les deux ablations, indépendantes, pointent vers un gain d'interprétabilité du
même ordre de grandeur — +9,4 et +8,7 points — sans qu'aucune des deux
n'atteigne la significativité seule).

## 26. Évaluation quantitative du retrieval Latent Terms (jamais faite jusqu'ici)

`src/sae/retrieval/latent_terms.py` (BM25 sur le vocabulaire latent d'un SAE
entraîné par pure reconstruction, Clavié et al. 2026, arXiv:2605.29384) n'était
exercé que via `scripts/retrieval_demo.py` (1-2 requêtes inspectées à l'œil, sur
des données de substitution FineWeb2/Wikipedia — écrit sur une machine sans
`Mails.tsv`) et le dashboard (parcours interactif, pas de métrique).

**Protocole** (`scripts/latent_retrieval_precision_eval.py`, job 41484, terminé en
1min12s — F2LLM-v2-80M + PhraseLevelSAE dédié dim320/8192/k16 entraîné sur les
3480 mails originaux, pas de Gemma-3-12B) : pour 4 intentions déjà validées comme
équilibrées (§16), une requête en langage naturel **paraphrasant** (pas copiant)
le motif regex de l'intention (`INTENT_KEYWORDS_FR`), Precision@10/@20 contre le
label faible d'intention comme vérité terrain, comparé à une baseline TF-IDF
classique sur le même corpus.

| Intention | Taux de base | P@10 Latent Terms | P@10 TF-IDF | P@20 Latent Terms | P@20 TF-IDF |
|---|---|---|---|---|---|
| réclamation | 54,7% | **1,00** | 1,00 | **1,00** | 1,00 |
| remboursement | 14,3% | **1,00** | 0,00 | **1,00** | 0,20 |
| information | 18,0% | **1,00** | 0,20 | **1,00** | 0,30 |
| urgence | 29,4% | **0,00** | 0,80 | **0,00** | 0,60 |

**Résultat, très hétérogène** : précision parfaite (1,00) pour 3 intentions sur 4,
et nettement supérieure à TF-IDF sur remboursement/information malgré des requêtes
qui ne reprennent PAS les mots exacts du motif regex (ex. "je souhaite être
remboursé du montant que j'ai payé en trop" ne contient ni "trop-perçu" ni "avoir")
— TF-IDF, qui ne peut matcher que des mots littéraux, échoue là où le retrieval
latent généralise sémantiquement. Sur réclamation (taux de base déjà élevé, 54,7%),
les deux méthodes atteignent le plafond.

**Échec complet sur "urgence"** (0,00 aux deux k, contre 0,80/0,60 pour TF-IDF) —
diagnostiqué en détail plutôt que pris tel quel : la requête active bien 31
features latentes non nulles (pas un vecteur dégénéré), mais **un seul document
sur 3480** dans tout le corpus partage une intersection non nulle avec ces 31
features précises (`LatentTermsIndex.search` ne retourne que les documents à
score `>0`) — et ce document n'est pas étiqueté "urgence". Ce n'est pas un bug de
calcul ni un raté sémantique profond, mais une **limite structurelle du BM25 sur
vocabulaire latent très parcimonieux** (k=16 activations par phrase) : si la
combinaison précise de features activées par une requête est rare/idiosyncratique
dans le corpus, le retrieval peut ne retourner presque rien, alors que TF-IDF
dégrade en douceur (chevauchement partiel de mots toujours possible même sans
intersection exacte).

**Conclusion** : le retrieval Latent Terms généralise mieux que TF-IDF pour 2
intentions sur 4 testées ici, mais son comportement n'est pas uniformément
fiable — sa sensibilité à l'intersection exacte du "vocabulaire" latent (plutôt
qu'à une similarité graduée) peut produire un échec complet et silencieux sur des
requêtes dont le code latent ne recoupe presque aucun document, sans qu'aucun
signal d'alerte n'indique que ce cas s'est produit (l'index retourne simplement
peu ou pas de résultats). Toute utilisation en production devrait vérifier le
nombre de documents à score non nul avant de faire confiance au classement.

## 27. Ablation `D_EXTRA=2048` seul (dictionnaire plus large, MÊME budget de parcimonie)

L'ablation "capacité" déjà faite (job 40953, §17.5) double `D_EXTRA` ET `K_EXTRA`
ensemble (1024/32 → 2048/64, ratio K/D=1/32 préservé). Jamais testé : élargir
SEULEMENT le dictionnaire (`D_EXTRA=2048`) à budget de parcimonie identique
(`K_EXTRA=32` inchangé, ratio K/D=1/64, deux fois plus sélectif). Run
`results_v13_ablation_d_extra2048_only` (job 41488, terminé en 3h35min56s), tout
identique au run principal sauf `D_EXTRA`.

| Run | `D_EXTRA`/`K_EXTRA` | Taux interp. | `rho_sae` | `clf_acc_email_axes` |
|---|---|---|---|---|
| Run principal | 1024/32 | 68/150 = 45,3% | 0,922 | 93,5% |
| Capacité doublée (§17.5, job 40953) | 2048/64 | 60/150 = 40,0% | — | — |
| **Ce run**, D_EXTRA seul | 2048/32 | **69/150 = 46,0%** | 0,925 | 91,2% |

Comparaison statistique au run principal : z=-0,12, aucun écart mesurable. `rho_sae`
s'améliore légèrement (0,922 → 0,925), cohérent avec plus d'atomes disponibles pour
reconstruire le résidu à budget de parcimonie identique.

**Lecture** : élargir le dictionnaire seul, sans toucher au budget de parcimonie,
ne change rien à l'interprétabilité (contrairement à l'hypothèse qu'un dictionnaire
plus sélectif -- ratio K/D plus faible -- produirait des features plus
spécialisées). Complète la conclusion déjà établie au §17.5 : ni la largeur du SAE
core, ni les époques, ni la capacité de l'extension (isolée OU combinée à un ratio
K/D constant OU variable) ne changent le taux d'interprétabilité une fois le
domaine du corpus corrigé -- ce protocole d'évaluation semble structurellement
plafonné par autre chose que les paramètres d'échelle testés jusqu'ici.

## 28. Ablation "échelle du modèle" : gemma-3-1b-it et -4b-it à la place de -12b-it

Toutes les ablations précédentes (largeur, époques, capacité, volume, seed,
K_EXTRA) gardent le modèle extracteur/juge fixé à gemma-3-12b-it. Jamais testé :
le taux d'interprétabilité dépend-il de l'échelle du modèle lui-même ? Runs
`results_v13_ablation_model_scale_1b`/`_4b` (jobs 41494/41493), gemma-3-1b-it et
gemma-3-4b-it + leurs GemmaScope dédiés (`gemma-scope-2-1b-it` layer 13/16k,
`gemma-scope-2-4b-it` layer 17/16k -- téléchargés et validés par chargement CPU
avant les jobs GPU), sinon identique au run principal. Le juge d'auto-
interprétation change AUSSI de modèle en même temps que l'extracteur (`MODEL_ID`
partagé par les deux rôles) -- confond structurellement "qualité des features
extraites" et "qualité du juge", comme pour toute paire modèle+SAE GemmaScope.

| Run | Modèle | `d_model` | Taux interp. | `clf_acc_email_axes` | `fve_pretrained` |
|---|---|---|---|---|---|
| Run principal | gemma-3-12b-it | 4096 | **68/150 = 45,3%** | 93,5% | — |
| `results_v13_ablation_model_scale_1b`, job 41494 (terminé, 2h46min23s) | gemma-3-1b-it | 1152 | **18/150 = 12,0%** | 88,2% | 0,565 |
| `results_v13_ablation_model_scale_4b`, job 41493 | gemma-3-4b-it | 2560 | *[en cours]* | *[en cours]* | *[en cours]* |

**Résultat 1b, de très loin le plus important de tout ce chapitre** : z=6,38,
**hautement significatif** (|z|>1,96 dépassé de plus de 3x) -- un effondrement de
45,3% à 12,0% (−33,3 points), sans commune mesure avec les écarts de 1 à 9 points,
tous non significatifs, observés sur TOUTES les autres ablations de ce chapitre
(largeur, époques, capacité, volume, seed, K_EXTRA). C'est le premier levier testé
dans ce projet qui déplace réellement le taux d'interprétabilité.

Fait notable en faveur d'une origine "qualité du juge" plutôt que "qualité des
features" : `clf_acc_email_axes` (séparabilité LINÉAIRE des axes de perturbation
dans l'espace des codes SAE, une mesure indépendante du juge LLM) ne s'effondre
PAS de la même façon (93,5% → 88,2%, seulement −5,3 points) -- les features
elles-mêmes restent en grande partie utilisables pour la classification, alors
que l'évaluation qualitative (odd-one-out) par le modèle 1B lui-même s'effondre
beaucoup plus fortement. `diff_hypothesis` (texte libre généré par le modèle 1B)
est qualitativement confus par rapport aux hypothèses cohérentes produites par le
12B (ex. inversion illogique cause/conséquence entre "énergie" et "sport" dans le
texte généré). `fve_pretrained` (0,565, très en dessous de 0,83 typique pour le
12B) suggère aussi une fraction de variance expliquée par le core GemmaScope
lui-même nettement dégradée à cette échelle.

**Conclusion (partielle -- 4b en cours)** : le plus fort effet mesuré dans tout ce
projet vient du choix du modèle lui-même, pas des hyperparamètres du SAE. Reste à
déterminer si 4b (échelle intermédiaire) montre une dégradation progressive ou un
effet de seuil -- complété dans la prochaine mise à jour de cette section.