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
  (FineWeb-2/Wikipedia filtré par mots-clés). Les emails réels et augmentés
  (`email_texts`) n'étaient chargés qu'**après** l'entraînement, uniquement pour
  une visualisation UMAP post-hoc (`analyze_with_umap`) — jamais vus par le SAE
  pendant l'entraînement. Le SAE d'extension apprenait donc des concepts
  Wikipedia génériques, jamais des concepts liés aux emails.

### Correction appliquée

- `src/data/preparation.py::build_email_train_test_corpus()` — nouveau corpus
  principal : mails réels (`Mails.tsv`) + variantes augmentées acceptées
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

## 13. Suites données au diagnostic §12 : robustesse du juge et validation métier sur mails réels

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

### 13.2. Le SAE prédit-il l'urgence et l'intention sur des mails réels (pas augmentés) ?

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
(+42,6 points) sur des mails réels non augmentés — validation directe et à moindre
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
information, remboursement) sur les mails réels, identifie pour 200 documents
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
activations déjà en cache) : sur 60 mails réels échantillonnés, présente au juge le
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
les axes email réels (la métrique la plus directement liée aux objectifs métier du
projet). Aucun de ces écarts n'est de l'ordre de grandeur d'un problème majeur (tous
< 3 points ou < 8% relatif) : conclusion retenue pour cette passe -- **pas de
justification claire pour préférer -330M à -80M sur ce projet** ; le choix n'est pas
un facteur bloquant avant de considérer une comparaison multi-modèles plus large.

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
