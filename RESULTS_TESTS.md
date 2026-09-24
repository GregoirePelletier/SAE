# SAE — Résultats des tests & état des jobs Slurm

_Cluster : partitions `a100` (dgx-a100, 8×GPU), `h100` / `h100-bis`
(dgx-h100{,-bis}, 8×GPU chacun)._

Fichier append-only : les identifiants `§N` sont cités depuis le rapport et
ne sont jamais renumérotés (`§35` n'existe pas — aucun contenu n'a été
retiré à cette position, cf. note à cet endroit). Chaque nouvelle section
suit le format : Question / Écart à la configuration de référence
(`docs/archive/evaluation_protocol.md`) / Méthode statistique / n / Résultat /
Conclusion / Limite connue.

## Index

Date : non renseigné quand la section elle-même ne la mentionne pas (aucune
n'est datée en prose dans ce fichier).

| § | Question posée | n | Résultat | Statut |
|---|---|---|---|---|
| 0 | Augmentation + baseline à l'échelle complète, tiennent-elles dans le budget de calcul ? | 43 423 textes | 8 shards parallèles, ~7h27 mur | confirmé |
| 1-3 | Audit initial : pipeline smoketest, tests unitaires | 8/8 tests | validés, inchangés | confirmé |
| 4-9 | Incidents d'infrastructure (scripts `.slurm`, OOM, budget de temps) | — | — | infra, voir `docs/ops_journal.md` |
| 10 | Run à l'échelle complète du pipeline principal | — | voir section | confirmé |
| 11 | Bilan général de la session d'audit initiale | — | voir section | infra |
| 12 | Le taux de détection de l'intrus est-il limité par le volume ou par le corpus ? | 10 puis 150 | domaine : 20%→45,3% ; volume : sans effet | confirmé |
| 13 | Robustesse du juge + validation métier sur mails originaux | 150 | voir section | confirmé |
| 14 | Biais "Objet :", dashboard, comparaison SAELens chiffrée | — | voir section | confirmé |
| 15 | Relecture interp_embed : 4 corrections méthodologiques | — | 4 écarts corrigés | confirmé |
| 16 | Qualité de l'explication document-level (fidélité/plausibilité) | 60 | fidélité sans ambiguïté, plausibilité 71,7% | confirmé |
| 17 | Ablation de mise à l'échelle : largeur SAE core, époques, N_FEATURES_TO_LABEL | 600 | 65k > 16k en couverture | confirmé |
| 18 | `FrozenCoreResidualSAE` est-elle une implémentation de SAE Boost ? | — | architecture identique | confirmé |
| 19 | Sanity check décodeur aléatoire (Korznikov et al. 2026) | 150 | 45,3% vs 29,3%, significatif ; classification résiste moins | confirmé, nuancé |
| 20 | Incident disque (lien symbolique) | — | — | infra, voir `docs/ops_journal.md` |
| 21 | Ablation de variance de seed | 150 vs 150 | 45,3% vs 47,3%, non significatif ; 28,2% recouvrement labels | confirmé |
| 22 | Biais multilingue du juge (FR vs EN traduit) | 150 | 46,9% vs 45,5%, non significatif ; 38,6% instabilité individuelle | confirmé |
| 23 | Ablation volume à grande échelle (~100-120M tokens) | 150 | pas d'effet significatif au volume testé | confirmé |
| 24 | Fidélité du steering (`steer_and_decode`) | 4 intentions | très hétérogène (0,00× à 1,74×) | confirmé |
| 25 | Ablation `K_EXTRA=5` | 150 | 54,7% vs 45,3%, non significatif | confirmé |
| 26 | Évaluation quantitative du retrieval Latent Terms | 4 requêtes | précision parfaite sur 3/4 | confirmé |
| 27 | Ablation `D_EXTRA=2048` seul | 150 | pas d'écart | confirmé |
| 28 | Ablation échelle du modèle (1b/4b/12b) | 150×3 | effet dose-réponse net et significatif | confirmé |
| 29 | Ablation largeur 262k | 150 | complète le balayage 16k/65k/262k | confirmé |
| 30 | Audit de méthodologie statistique | — | 3 lacunes trouvées et corrigées | confirmé |
| 31 | Balayage `MATRYOSHKA_DIM` | — | dégradation graduelle, pas abrupte | confirmé |
| 32 | Audit qualité 16k vs 65k (monosémanticité) | — | voir section | confirmé |
| 33 | Audit HDBSCAN : UMAP-2D vs cosine vs UMAP-nD vs PCA-nD | — | UMAP-10D retenu | confirmé |
| 34 | Reproductibilité par groupe de features vs feature individuelle | 78 labels | voir section | confirmé |
| 35 | — | — | — | absent (non-attribué) |
| 36 | GemmaScope-2 12b-it publie plus que resid_post layer 24 | — | audit de disponibilité | confirmé |
| 37 | Fuite lexicale dans le corpus augmenté (`clf_acc_email_axes`) | 14 classes | TF-IDF seul : 87,0% sur 93,5% rapporté | confirmé |
| 38 | Garde-fou qualité de l'augmentation : rejet massif sur 2 classes/13 | 45 240 | voir section | confirmé |
| 39 | Garde-fou `facts_lost` : faux positifs sur reformatage pur | — | corrigé | confirmé |
| 40 | Validation de `find_interesting_pairs` par injection synthétique | — | résultat positif | confirmé |
| 41 | Erreur juge vs erreur SAE sur 30 features non interprétées | 30 | voir section | confirmé |
| 42 | Juge par échantillonnage (temp=0,7) | — | stable | confirmé |
| 43 | Séparation juge/extraction (gemma-3-4b-it juge) | 150 | juge 2x moins de features interprétables | confirmé |
| 44 | Revue externe multi-perspective (avocat du diable) | — | Major Revision, 2 CRITICAL adjugés | confirmé |
| 45 | Réplication multi-seed `K_EXTRA=5` | 450 (3 seeds) | direction confirmée, significativité non atteinte | confirmé |
| 46 | Test confirmatoire C1 : effet domaine à n apparié | 150 vs 150 | 45,3% vs 30,0%, z=2,74, p≈0,006 | confirmé |
| 47 | Échecs d'infrastructure sur 5 jobs | — | — | infra, voir `docs/ops_journal.md` |
| 48 | Test C2 gratuit : origine des exemples positifs vs interprétabilité | 150 | pas de corrélation | confirmé |
| 49 | Bug hook attn_out/mlp_out : mauvais chemin d'attribut | — | corrigé | confirmé |
| 50 | C2 (suite) : re-jugement faisable, contrôle core | 150 | faisabilité 100% | confirmé |
| 51 | Balayage layer resid_post (12/31/41) | 150×4 | layer 31 significativement meilleur (z=2,20) | confirmé, à répliquer |
| 52 | C2 résolu : re-jugement original-only | 150 | 44,7% vs 45,3%, non significatif | confirmé |
| 53 | Hook-point resid_post vs attn_out vs mlp_out | 150×3 | mlp_out > attn_out significatif (z=3,02) | confirmé |
| 54 | Infrastructure de diagnostic + fix du réservoir mémoire | — | memmap disque, courbes, playbook | confirmé |
| 55 | Le SAE core seul égale-t-il core+extension sur les métriques en aval ? | 150 (probes) | aucune différence significative | confirmé |
| 56 | Réplication seed multiple de l'ablation volume 25M (§23.4) | 150×2 | +0,7pt et -0,6pt vs référence, ne réplique pas le seed original | confirmé, piste abandonnée |

## 0. Corpus complet — augmentation parallélisée (8 shards) + baseline

**Vérification du biais de formatage (`docs/ops_journal.md` §6) avant le run
complet** : confirmé sur l'échantillon test (0,03% des mails originaux ont
"Objet :" vs 25,6% des augmentés) → prompt système corrigé
(`src/data/augmentation.py`, contrainte explicite : interdiction d'ajouter
une ligne "Objet :"/mise en forme absente de l'original). Fichier partiel
généré avec l'ancien prompt archivé séparément pour ne pas contaminer le run
complet.

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
`pytest` re-passés (8/8 OK -- décompte historique à ce stade du projet ; la suite
compte 9 tests aujourd'hui après l'ajout ultérieur de `tests/test_interp_embed_diff.py`,
toutes les mentions "8/8"/"8 passed" de ce document reflètent des instantanés
antérieurs à cet ajout, pas une régression).

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

**Correction (vérifié en relisant les CSV directement)** : l'affirmation initiale
ici ("le run complet n'a pas pu joindre Neuronpedia, toutes les features top sont
des identifiants bruts non labellisés") était **fausse** -- les labels utilisés
proviennent du cache local (`local_data/neuronpedia_labels/`), peuplé hors-ligne
avant ce run via `fetch_neuronpedia_labels()`, donc indépendants de toute
connectivité réseau au moment de l'exécution du pipeline. En relisant directement
les CSV de sortie (`results_v9_test/cache_baseline_full/diff_*.csv`) : **82,4%
(36044/43751) des features significatives portent un vrai label Neuronpedia**,
pas un identifiant brut `F{idx}` -- certaines features du tableau ci-dessus
(F15531 = "roman numeral lists", F13696 = "repetition or repeating", etc.) ont
d'ailleurs un label réel malgré leur affichage sous forme `F{idx}` dans ce
tableau de synthèse (choix de présentation compact, indépendant de la présence
effective d'un label dans les données).

---

## 1. Ce qui a été audité

Tous les scripts `.slurm` du repo ont été relus, comparés aux scripts Python qu'ils
appellent, et testés (localement en syntaxe/imports, puis réellement soumis sur le
cluster quand c'était possible). Détail par script ci-dessous.

## 2. `slurm/pipeline_runs/run_sae.slurm` (smoketest du pipeline principal) — ✅ déjà validé, inchangé

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

**Conclusion : le pipeline (`slurm/pipeline_runs/run_sae.slurm`) est fonctionnel et reproductible. Pas
besoin de le relancer pour ce tour de validation.**

## 3. Tests unitaires (`pytest`, CPU seulement) — ✅ passent tous

```
$ .venv/bin/python -m pytest tests/ -q
........                                                                 [100%]
8 passed, 3 warnings in 43s
```

Aucune modification nécessaire côté code testé.

## 4. Bugs trouvés dans les 3 autres `.slurm` (jamais fonctionnels tels quels)

Incident d'infrastructure, détail : `docs/ops_journal.md` §4.

## 5. Budget de temps de l'augmentation complète

Incident d'infrastructure, détail : `docs/ops_journal.md` §5.

## 6. OOM CUDA dans `slurm/baseline_diffing/run_baseline.slurm`

Incident d'infrastructure, détail : `docs/ops_journal.md` §6. Observation
méthodologique retenue : première mise en évidence du biais de formatage
résiduel ("Objet :"/"Subject :"), mesuré et corrigé par la suite (§14.1).

## 7. Suivi des jobs (session d'audit initiale)

Incident d'infrastructure, détail : `docs/ops_journal.md` §7.

## 8. Correctifs de code de cette passe

Incident d'infrastructure, détail : `docs/ops_journal.md` §8.

## 9. Suivi (session d'audit initiale)

Incident d'infrastructure, détail : `docs/ops_journal.md` §9.

---

## 10. Run à l'échelle complète du pipeline principal (`slurm/pipeline_runs/run_sae_full.slurm`)

Après validation que la chaîne augmentation → baseline fonctionnait, question posée :
le smoketest `slurm/pipeline_runs/run_sae.slurm` (volumes réduits ~×12) suffit-il, ou faut-il investiguer
avant un run à pleine échelle ? Vérification faite :

- **`saev5.py` (pipeline P1/P2) n'est PAS vulnérable au bug quadratique de `maxpool_sae_docs`** :
  son pooling P1 utilise `doc_maxpool` (`src/storage/fragment_store.py`) sur des
  fragments **sparses par document** (CSR, O(nnz) par doc) — c'est justement
  une architecture pensée pour éviter les gros tenseurs `[n_docs × d]` denses. Pas de
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
  délibéré documenté dans `src/config.py` : bien meilleure couverture
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
0.72) — cohérent avec un SAE mieux entraîné sur plus de données. P2 dead%=0,049%
(4 features mortes sur les 8192, arrondi à 0,0 dans le tableau ci-dessus ; contre
0,24% en smoketest sur 2048) : bon signe de
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
| `slurm/pipeline_runs/run_sae.slurm` (smoketest) | ✅ déjà validé (inchangé) |
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


---

## 12. Diagnostic et correction du faible taux de détection de l'intrus (odd-one-out)

**Ma question** : le taux de labellisation des features d'extension
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
largement aux objectifs du projet (détection d'urgence, détection d'intentions)
et à la piste ouverte sur le résidu de ~55-59% de features non interprétées.

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
| `intent_resiliation` | 0,03% (1/3300) | — | — | ignoré (classe dégénérée) |

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
projet) : lit uniquement des artefacts déjà sur disque (JSON, parquet,
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
et al. 2025), comparaison ligne à ligne avec notre code. Quatre écarts concrets
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
- **2e run** (prompt corrigé) : plus aucun echo de template. Les 82 features
  originellement non-interprétables obtiennent chacune un label -- mais
  **"chacune obtient un label" ne veut pas dire "82 labels distincts"**, et la
  lecture initiale ("labels cohérents et directement pertinents, spécifiques et
  distincts") était fausse. Vérification systématique (pas un échantillon à
  l'œil) sur les 82 labels : **58 chaînes distinctes seulement -- 13 labels
  dupliqués, 37/82 features (45%) partagent leur label avec au moins une
  autre**. `"Demande d'action"` apparaît 7 fois, `"Expression de
  mécontentement"` 6 fois, `"Demande de résiliation"`/`"Coordonnées de
  contact"` 3 fois chacun. Deux paires (`16720`/`16949` : 8/9 exemples
  positifs identiques ; `16720`/`16852` : 7/9) ont d'abord semblé être des
  doublons de dictionnaire (*feature splitting*) -- **vérifié directement sur
  les activations et infirmé** : corrélation ≈0 entre ces paires, pas de
  quasi-redondance. Le vrai bug : `F16949` a une fréquence d'activation de
  **0,0000%** sur les 41176 documents train (feature quasi-morte) mais reçoit
  quand même un label spécifique et confiant avec des exemples positifs
  identiques à ceux de `F16720` -- la sélection d'exemples de
  `contrastive_labeling_test.py` ne reflète pas l'activation réelle de la
  feature pour ce cas, pas encore diagnostiqué au niveau code.
- `confident` auto-rapporté reste `true` pour 150/150 features dans les deux
  runs, y compris les doublons ci-dessus -- signal inutilisable, cohérent
  avec le biais de complaisance connu des LLM juges.

**Verdict** : le "100% de récupération" n'est pas une preuve que le protocole
contrastif retrouve du signal réel sur les features non-interprétées -- c'est
en bonne partie l'artefact attendu d'un LLM complaisant qui produit toujours
une étiquette plausible piochée dans un vocabulaire étroit de tropes email
client (urgence, résiliation, coordonnées, mécontentement), sans gate pour
refuser. Le protocole odd-one-out reste le seul chiffre publiable en l'état ;
la labellisation contrastive n'est pas prête à le remplacer sans (1) un gate
de similarité inter-labels/inter-exemples pour éliminer les doublons avant
comptage, (2) une revue humaine sur échantillon plutôt que `confident`.
**Aucun changement appliqué au pipeline de production** -- le chiffre 45,3%
publié reste la référence.

---

## 16. Qualité de l'explication document-level + protocole de test complet du repo

Suite directe de ma question : "comment tester ma pipeline de bout en bout
pour estimer la qualité de notre solution d'explication des documents ?". Deux tests
complémentaires (fidélité causale + plausibilité perçue), puis un protocole
d'évaluation consolidé couvrant l'ensemble des méthodes du dépôt, sous conditions
fixées (`docs/archive/evaluation_protocol.md`).

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
retenues sur 26 579 arêtes du graphe de cooccurrence (3 395 nœuds), et les **3 des 3
paires** impliquent une feature non labellisée (respectivement `F43`, `F17402`,
`F17315`, chacune appariée à la feature 1873 = "gem mining history") — résultat honnête
mais peu exploitable en l'état (impossible de juger si la corrélation est un artefact
réel sans label sur au moins un des deux côtés). Piste de suite : élargir la plage de
fréquence (`min_freq`/`max_freq` de `cooccurrence_graph`) ou prioriser les paires où
les deux features sont labellisées.

### 16.4. Protocole d'évaluation complet du dépôt (conditions fixées)

`docs/archive/evaluation_protocol.md` (nouveau) : recense les 16 capacités/méthodes du dépôt,
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
directement à l'étape manquante (2min20s selon `sacct`, au lieu de ~10 minutes).

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

## 17. Ablation de mise à l'échelle : largeur du SAE core, époques, N_FEATURES_TO_LABEL

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
premier essai de lancement de ce run (jobs 40833/40827 en échec, poids jamais
téléchargés — seuls les LABELS avaient été vérifiés, pas les poids du SAE
lui-même), corrigé avant resoumission (jobs 40844/40845).

### 17.1. Run combiné (`results_v12_scaled_65k`, job 40844+40845+40846-40850)

Trois leviers augmentés SIMULTANÉMENT par rapport au run principal (`results_v10_emails_main`) :

| Paramètre | Run principal | Run à l'échelle | Facteur |
|---|---|---|---|
| Largeur SAE core | 16k | 65k | — |
| `EPOCHS_EXTRA` | 10 | 40 | ×4 |
| `EPOCHS` (P2) | 30 | 100 | ×3,3 |
| `N_FEATURES_TO_LABEL` | 150 | 600 | ×4 |
| Backbone P2 | F2LLM-v2-80M | F2LLM-v2-330M | — |

**Résultats** :

| Métrique | Run principal (16k) | Run à l'échelle (65k) |
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

### 17.3. Alignement label/feature : chemin de labels figé sur 16k

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

Calculé systématiquement depuis (`dead_pct_core`/`dead_pct_extension`,
`src/analysis/metrics.py::dead_pct_core_extension`, `results.json` de tout run
`USE_FROZEN_CORE=True`) plutôt que de dépendre d'un rappel manuel de cette
section à chaque lecture d'un `dead_pct` élevé.

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

| Métrique | Run principal (16k) | Run à l'échelle, labels 16k (chargés à tort) | **Run à l'échelle, labels corrigés (65k)** |
|---|---|---|---|
| Plausibilité (choix forcé, 60 documents) | 71,7% (43/60) | 56,7% (34/60) | **88,3% (53/60)** |
| Fidélité (ratio top/random, moyenne 4 intentions) | 250×-576 000× | *(non affecté, cosmétique)* | 890,9×-32 992,3× (même ordre de grandeur) |

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
initial du projet, marquée "non fait" dans `docs/references.md` jusqu'ici.

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
et le rapport de stage.

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
100M-200M sans le tester directement.

**Mise à jour** : un run à 25M tokens a depuis été mené (§23.4, +12x l'ablation
initiale, toujours sans effet significatif) et un run visant 200M tokens (borne
haute exacte du seuil du papier) a été lancé (§23.4, job 41658) une fois les
deux obstacles bloquants résolus (mémoire hôte, volume de filler disponible) --
en attente de démarrage au moment de la rédaction (cluster en incident de
maintenance). Reste, en l'état, une limite partiellement levée plutôt
qu'un résultat définitif tant que le job 200M n'a pas produit de résultat.

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
littérature générale, jamais sur un test empirique comparatif propre à ce
projet. Piste de poursuite.

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
| Interprétabilité odd-one-out (n=150) | 45,3% (68/150) | **29,3%** (44/150) | −16,0 points | z=2,86 (p<0,01) |
| `clf_acc_email_axes` (14 classes, n=2177 test) | 93,5% | **91,2%** | −2,3 points | z=2,86 (p<0,01) |
| Features mortes (extension) | 0 | 0 | — | — |

**Lecture — résultat nuancé, ni réplication à l'identique du papier, ni réfutation
complète** :

- **Interprétabilité odd-one-out : écart réel et substantiel** (16 points, très
  au-delà de l'écart-type binomial ≈4,1pt à n=150, z=2,86). Contrairement au
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

## 20. Suppression accidentelle d'un lien symbolique lors d'un nettoyage disque

Incident d'infrastructure, détail : `docs/ops_journal.md` §20.

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
| Recouvrement EXACT des labels interprétables (chaîne de caractères,
normalisée : préfixe `[EXT]` retiré, insensible à la casse) | — | **22/78 = 28,2%** | |

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
(`src/data/keywords.py`, 34 phrases composées) retenu comme meilleur compromis
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

**Mise à jour** : ce doublement transitoire a depuis été éliminé (commit `18cc204`,
*"Élimine le doublement transitoire de mémoire du réservoir de résidus"*) : le
buffer `reservoir` est désormais préalloué une seule fois à sa taille finale dès
le début de l'extraction et rempli directement par tranches (phase 1) puis par
écriture indexée (phase 2, Vitter), sans jamais coexister avec une structure
intermédiaire de même taille. `raw_residuals_list` n'existe plus dans le code
actuel. Cette correction, restée non documentée ici jusqu'à cette mise à jour, a
été validée par un test isolé (tenseurs factices, corpus plus grand ET plus
petit que la cible) avant d'être exploitée pour le run à 200M tokens (§23.4).

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

### 23.4. Résultats (job 41375, terminé en 18h17min21s)

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
plusieurs seeds). **Réplication faite au §56 : ne tient pas** — les deux
seeds supplémentaires retombent sur la référence (46,0% et 44,7%), le pool à
3 seeds (48,2%) est loin du 54,0% mesuré ici.

Aucun signal de dégradation liée au filler générique (FineWeb2-fr, ~15-20% de
précision qualitative sur le thème énergie, cf. §23.1) : le taux d'interprétabilité
et `clf_acc_email_axes` restent dans la même fourchette que les runs sans filler,
cohérent avec le fait que le filler n'alimente que le réservoir de tokens résiduels
et jamais `train_texts`/la sélection de features (cf. §23.2) — le garde-fou conçu
pour éviter de réintroduire le biais de domaine original semble avoir fonctionné
comme prévu.

### 23.5. Run à 200M tokens (borne haute exacte du seuil SAE Boost)

Avec la soutenance repoussée d'environ deux mois, le temps disponible permet de
viser directement la borne haute (200M) de la fourchette 100-200M du papier,
plutôt que le compromis à 25M. Deux obstacles bloquants ont été levés avant ce
lancement :

- **Mémoire** : `N_TOKENS_EXTRA_TRAIN=200000000 × 3840 (hidden_size) × 2 octets
  (bf16) ≈ 1,40 To` pour le seul buffer de résidus (préalloué, sans
  doublement transitoire depuis §23.3 mise à jour ci-dessus). Toutes les
  partitions utilisées jusqu'ici (a100, ~1 To de RAM total) sont insuffisantes
  -- ce run cible `h100` (~2 To de RAM par nœud, `--mem=1800G`).
- **Volume de filler disponible** : les 3 shards FineWeb2-fr existants (§23.1)
  ne donnaient qu'environ 65-70M tokens exploitables -- insuffisant même pour
  100M avec marge. 15 shards supplémentaires ont été téléchargés (18 au total,
  ~87 Go), portant le pool estimé à ~350-400M tokens bruts de filler.

Job **41658** (`slurm/pipeline_runs/run_ablation_volume_200m.slurm`,
`results_v13_ablation_volume200m`) soumis : à ce jour, l'état est **PENDING**
-- les trois nœuds du cluster (a100, h100, h100-bis) sont en incident de
maintenance simultané (`Reason=Kill task failed`, non lié à ce projet), tous
en état `drain`/`draining`. Le job démarrera automatiquement une fois le
cluster rétabli, sans action supplémentaire requise. *[Résultats à compléter
une fois le job terminé.]*

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
| Run principal, `results_v10_emails_main` | 32 | **68/150 = 45,3%** | 93,5% | 0,906 |
| Ablation capacité, §17.5 | 64 | — | — | — |
| **Ce run**, `results_v13_ablation_k_extra5`, job 41404 | 5 | **82/150 = 54,7%** | 91,2% | 0,849 |

Comparaison statistique au run principal : z=-1,62, **non significatif** au seuil
conventionnel (\|z\|>1,96) malgré un écart numérique de +9,4 points — le plus
proche du seuil de significativité (p≈0,106 bilatéral) de toutes les ablations
testées dans ce chapitre. `rho_sae` (proxy de fidélité de reconstruction du
résidu) recule sensiblement (0,906 → 0,849), cohérent avec le fait qu'un budget de
capacité par token beaucoup plus faible (k=5 vs 32) réduit mécaniquement la
fraction du résidu reconstructible — attendu, et conforme à la logique du papier
(k plus faible = code plus parcimonieux, quitte à moins bien reconstruire, en
échange d'une meilleure interprétabilité par feature active).

**Lecture** : direction cohérente avec l'hypothèse du papier (k plus faible →
meilleure interprétabilité), mais non significatif isolément sur n=150. Voir
§23.4 pour la discussion de la coïncidence directionnelle avec l'ablation volume
(les deux ablations, indépendantes, pointent vers un gain d'interprétabilité du
même ordre de grandeur — +9,4 et +8,7 points — sans qu'aucune des deux
n'atteigne la significativité seule).

## 26. Évaluation quantitative du retrieval Latent Terms (jamais faite jusqu'ici)

**Supersédé, sans remplacement produit à ce jour** : cette section et §68(c)/§69(c)
mesurent un SAE **phrase-level** entraîné directement sur Mails.tsv, pas la méthode
token-level du papier (§3.1-3.2) — écart identifié et corrigé, `latent_terms.py`
réimplémenté fidèlement. L'OOM de `build_token_training_pool` qui bloquait la version
token-level est corrigé (le pool se construit, job 44666 : 33M tokens) ; le run le plus
récent a été tué par un budget `--time` insuffisant pendant l'entraînement du SAE, pas
par manque de mémoire — relancé (jobs 44995/44996). Chiffres ci-dessous conservés comme
trace historique (append-only), à ne pas citer comme résultat Latent Terms de référence
tant que 44995/44996 n'ont pas produit de JSON.

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
| Run principal | 1024/32 | 68/150 = 45,3% | 0,906 | 93,5% |
| Capacité doublée (§17.5, job 40953) | 2048/64 | 60/150 = 40,0% | — | — |
| **Ce run**, D_EXTRA seul | 2048/32 | **69/150 = 46,0%** | 0,925 | 91,2% |

Comparaison statistique au run principal : z=-0,12, aucun écart mesurable. `rho_sae`
s'améliore légèrement (0,906 → 0,925), cohérent avec plus d'atomes disponibles pour
reconstruire le résidu à budget de parcimonie identique.

**Lecture** : élargir le dictionnaire seul, sans toucher au budget de parcimonie,
ne change rien à l'interprétabilité (contrairement à l'hypothèse qu'un dictionnaire
plus sélectif -- ratio K/D plus faible -- produirait des features plus
spécialisées). Complète la conclusion déjà établie au §17.5 : ni la largeur du SAE
core, ni les époques, ni la capacité de l'extension (isolée OU combinée à un ratio
K/D constant OU variable) ne changent le taux d'interprétabilité une fois le
domaine du corpus corrigé -- ce protocole d'évaluation semble structurellement
plafonné par autre chose que les paramètres d'échelle testés jusqu'ici.

## 28. Ablation "échelle du modèle" : un effet dose-réponse net et significatif

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

| Run | Modèle | `d_model` | Taux interp. | z vs 12b | `clf_acc_email_axes` | `fve_pretrained` |
|---|---|---|---|---|---|---|
| `results_v13_ablation_model_scale_1b`, job 41494 (2h46min23s) | gemma-3-1b-it | 1152 | **18/150 = 12,0%** | 6,38 (sign.) | 88,2% | 0,565 |
| `results_v13_ablation_model_scale_4b`, job 41493 (3h01min14s) | gemma-3-4b-it | 2560 | **42/150 = 28,0%** | 3,12 (sign.) | 92,0% | 0,633 |
| Run principal | gemma-3-12b-it | 3840 | **68/150 = 45,3%** | — (référence) | 93,5% | — |

**Résultat : un effet dose-réponse net, monotone, et significatif à chaque palier**
-- 12,0% → 28,0% → 45,3%. Chaque comparaison deux-à-deux est significative :
1b vs 12b (z=6,38), 4b vs 12b (z=3,12), et **1b vs 4b directement (z=-3,46,
également significatif)** -- la progression n'est pas un artefact de comparaison
au même run de référence, c'est une vraie tendance croissante avec l'échelle. De
très loin le plus fort effet mesuré dans tout ce projet, sans commune mesure avec
les écarts de 1 à 9 points, tous non significatifs, observés sur TOUTES les
autres ablations de ce chapitre (largeur, époques, capacité, volume, seed,
K_EXTRA) -- c'est le premier et seul levier testé qui déplace réellement le taux
d'interprétabilité.

Fait notable en faveur d'une origine "qualité du juge" plutôt que "qualité des
features" : `clf_acc_email_axes` (séparabilité LINÉAIRE des axes de perturbation
dans l'espace des codes SAE, indépendante du juge LLM) suit une pente BEAUCOUP
plus douce (88,2% → 92,0% → 93,5%, seulement 5,3 points d'écart total) que le
taux d'interprétabilité qualitatif (33,3 points d'écart total) -- les features
elles-mêmes restent en grande partie utilisables pour la classification même à
1B, alors que l'évaluation qualitative (odd-one-out, jugée par ce même modèle)
s'effondre beaucoup plus fortement. `fve_pretrained` suit la même tendance
monotone (0,565 → 0,633 → ~0,83 typique 12B), cohérent avec une fraction de
variance expliquée par le core GemmaScope lui-même qui se dégrade avec l'échelle.
Le texte `diff_hypothesis` (généré par chaque modèle) est qualitativement confus
pour 1B (inversion logique cause/conséquence entre "énergie" et "sport"), plus
cohérent mais encore limité pour 4B, pleinement cohérent pour 12B -- une lecture
qualitative qui corrobore la tendance quantitative.

**Conclusion** : le choix du modèle extracteur/juge est, de loin, le levier le
plus déterminant testé dans ce projet pour le taux d'interprétabilité -- bien
au-delà de tout hyperparamètre du SAE (largeur, capacité, volume, seed). La
dissociation entre classification (robuste, dégradation modérée) et
interprétation qualitative (fragile, dégradation sévère) suggère que la capacité
de RAISONNEMENT du juge (formuler et vérifier un concept partagé entre 9
exemples) est le facteur limitant à petite échelle, plus que la qualité des
représentations latentes elles-mêmes. Piste directement actionnable pour la
suite : séparer les rôles extracteur/juge (garder un modèle de taille modeste
pour l'extraction, mais un juge plus capable) pour isoler laquelle des deux
capacités domine réellement cet effet.

## 29. Ablation largeur 262k (12b) : complète le balayage 16k/65k/262k

Troisième et dernière largeur de SAE core GemmaScope-2-12b-it disponible
localement (téléchargée initialement pour la mesure de couverture Neuronpedia,
§17.0 : 262k -> seulement 5,3% de couverture, très en dessous de 16k/65k). Run
`results_v13_ablation_width262k_only` (job 41487, terminé en 15h33min10s --
nettement plus long que 16k/65k du fait de la taille du core SAE lui-même,
262 144 dimensions).

| Run | Largeur | Taux interp. | z vs 16k | `clf_acc_email_axes` | `fve_pretrained` |
|---|---|---|---|---|---|
| Run principal | 16k | 68/150 = 45,3% | — | 93,5% | — |
| Largeur seule, job 40952 (§17.5) | 65k | 65/150 = 43,3% | — | — | — |
| **Ce run**, 262k | 262k | **70/150 = 46,7%** | -0,23 (non sign.) | 91,4% | 0,806 |

**Résultat** : aucun écart significatif (z=-0,23), cohérent avec 65k (déjà non
significatif). La couverture Neuronpedia très dégradée à 262k (5,3%) n'affecte
donc PAS le taux d'interprétabilité de l'EXTENSION (auto-labellisée par le juge,
indépendante du catalogue Neuronpedia) -- seul le catalogue "Core" pré-existant
en aurait souffert, sans impact sur la métrique suivie dans ce projet. Complète
proprement le balayage de largeur du SAE core (16k/65k/262k) : **aucune des
trois largeurs ne change significativement l'interprétabilité** -- cohérent avec
la conclusion déjà établie que ce protocole d'évaluation est plafonné par
autre chose que les paramètres d'échelle du SAE core, et renforce par contraste
à quel point l'effet de l'échelle du MODÈLE (§28) est hors norme parmi tous les
leviers testés dans ce projet.

## 30. Audit de méthodologie statistique : tests plus appropriés jamais utilisés

Ma question directe : le repo utilise-t-il tous les
tests statistiques appropriés ? Audit du code (pas seulement de la prose des
sections précédentes) et re-calcul rétroactif sur les données déjà en cache.

### 30.1. Comparaisons appariées traitées comme indépendantes (McNemar jamais utilisé)

`scripts/multilingual_judge_bias_test.py` et `scripts/judge_robustness_check.py`
ne calculent eux-mêmes AUCUN test de significativité (juste des taux bruts) --
tous les z de ce chapitre pour ces deux tests ont été calculés à la main, via un
test z à deux proportions **indépendantes**. Or les deux comparent le MÊME
ensemble de 150 features sous deux conditions (FR vs EN traduit ; décision
single-shot vs vote majoritaire) -- un plan **apparié**, pas deux échantillons
indépendants. Le test statistiquement approprié est **McNemar** (sur les paires
discordantes), pas un test à deux proportions indépendantes.

Recalcul à partir des paires discordantes déjà stockées en cache
(`multilingual_judge_bias_results.json`, `p1_judge_robustness.json`) :

| Test | Discordants (b/c) | z indépendant (déjà rapporté) | McNemar exact (binomial) |
|---|---|---|---|
| FR vs EN traduit (§22) | 27/29 | z=0,24 | **p=0,894** |
| Single-shot vs vote majoritaire (§13.1) | 26/21 | (non calculé formellement) | **p=0,560** |

**Aucun changement de conclusion** (les deux restent non significatifs), mais
McNemar est le test correct pour ce plan d'expérience et aurait dû être utilisé
depuis le début -- l'accord entre les deux approches ici est rassurant, pas une
validation de la méthode utilisée jusqu'ici.

### 30.2. Aucune correction multi-tests sur les ~15 comparaisons d'ablation

`src/analysis/cooccurrence.py` applique déjà `statsmodels.stats.multitest.
multipletests(..., method="fdr_bh")` -- mais UNIQUEMENT pour le test de diffing
Fisher exact sur les features individuelles d'un run (des centaines
d'hypothèses). Aucune correction n'a jamais été appliquée aux ~15 tests
"d'ablation" de ce chapitre (seed, multilingue, robustesse, K_EXTRA, D_EXTRA×2,
largeur×3, volume, échelle modèle×3, sanity check), chacun interprété
individuellement à α=0,05. Avec 15 tests indépendants sous une hypothèse nulle
globale, la probabilité qu'au moins un atteigne p<0,05 par pur hasard est
`1-0,95^15 ≈ 54%` -- une correction (Holm, ou Benjamini-Hochberg si on accepte un
contrôle du FDR plutôt que du FWER) serait la pratique rigoureuse. **Ne change
aucune conclusion en pratique ici** : le seul résultat significatif (échelle du
modèle, §28) est significatif à p<10⁻⁹, très loin de survivre n'importe quelle
correction raisonnable ; les résultats déjà qualifiés de "non significatifs"
(K_EXTRA=5 notamment, le plus proche du seuil à p≈0,106) le resteraient
d'autant plus après correction (le seuil corrigé est PLUS strict, pas plus
permissif). Reste une lacune méthodologique à corriger pour toute future
extension de ce chapitre, où plusieurs résultats pourraient être plus proches
du seuil.

### 30.3. Trend test manquant pour l'effet dose-réponse (§28)

Le résultat d'échelle du modèle (1b/4b/12b) a été analysé via 3 tests z par
paires (1b-12b, 4b-12b, 1b-4b) -- techniquement correct mais statistiquement
sous-optimal pour un plan dose-réponse à 3 niveaux ORDONNÉS : le test approprié
est un test de tendance (Cochran-Armitage), qui donne UNE seule statistique
condensant toute l'information de la tendance croissante, plus puissant que des
comparaisons par paires (et évite le problème de correction multi-tests du
§30.2 pour ce sous-groupe spécifique).

Recalculé sur les 3 points (1b : 18/150, 4b : 42/150, 12b : 68/150) :

| Pondération des groupes | z (Cochran-Armitage) | p |
|---|---|---|
| Scores linéaires (1, 2, 3) | 6,399 | **1,57×10⁻¹⁰** |
| log(nombre de paramètres) | 6,375 | **1,83×10⁻¹⁰** |

Confirme et renforce le résultat déjà rapporté au §28 (tendance monotone très
significative), avec une statistique unique et plus rigoureusement adaptée au
plan d'expérience, quasiment insensible au choix du barème de score (linéaire
vs log-échelle) -- la conclusion est robuste au choix arbitraire d'espacement
entre les trois points testés.

### 30.4. Autres lacunes identifiées (non corrigées dans cette passe)

- **Intervalles de confiance** jamais rapportés systématiquement à côté des
  taux ponctuels (seulement point estimate + z + significatif/non) -- un
  intervalle de Wilson ou Agresti-Coull serait plus informatif qu'un simple
  verdict binaire, en particulier pour les résultats proches du seuil.
- **Taille d'effet standardisée** jamais calculée (les écarts sont rapportés en
  points de pourcentage bruts, qui mélangent taux de base et magnitude
  d'effet) -- le h de Cohen pour proportions permettrait de comparer
  l'ampleur des effets entre ablations à des taux de base différents (ex.
  comparer l'effet K_EXTRA, parti de 45,3%, à l'effet largeur, parti de 45,3%
  aussi ici donc pas un problème pour CE chapitre spécifiquement, mais le
  deviendrait pour toute comparaison future à un taux de base différent).
- **Analyse de puissance** jamais formalisée : à n=150 et taux de base ~45%,
  quelle est la plus petite différence détectable à 80% de puissance ? Cadrerait
  les attentes sur ce qui est raisonnable de chercher avec ce protocole plutôt
  que de découvrir après coup qu'un écart de quelques points est indétectable.

## 31. Balayage MATRYOSHKA_DIM (F2LLM-v2-160M) : dégradation graduelle, pas abrupte

Ma question directe : `MATRYOSHKA_DIM` (troncature de
l'embedding F2LLM, `src/config.py`, défaut 320) n'avait jamais été varié. Fait
découvert en creusant : `F2LLM-v2-80M` a `hidden_size=320`, **exactement égal**
au défaut -- la "troncature" était un no-op pur pour ce backbone dans toutes
les comparaisons précédentes (§16.5-16.7), jamais remarqué. Seuls 160M
(640→320) et 330M (896→320) tronquaient réellement, toujours au même point
fixe. Ni la doc F2LLM (README HF, vérifié) ni ce projet n'ont jamais confirmé
un entraînement avec objectif Matryoshka (MRL) au sens strict.

4 runs sur F2LLM-v2-160M (hidden_size=640), `MATRYOSHKA_DIM` ∈ {64, 128, 320,
640(complet, aucune troncature)}, jobs 41598/41599/41600 + point 320 déjà
obtenu (`results_v10_p2_f2llm160m`) :

| `MATRYOSHKA_DIM` | Fraction du complet | NMSE | dead% | `clf_acc_email_axes` | `clf_acc_sae` (diffing) |
|---|---|---|---|---|---|
| 64 | 10,0% | 0,0706 | 0,0% | 72,8% | 64,8% |
| 128 | 20,0% | 0,0803 | 0,0% | 75,0% | 65,8% |
| 320 (défaut) | 50,0% | 0,0727 | 0,77% | 76,8% | 67,0% |
| 640 (complet) | 100% | 0,0697 | 3,31% | **77,4%** | **68,5%** |

**Résultat** : `clf_acc_email_axes` et `clf_acc_sae` augmentent tous deux de
façon MONOTONE avec la dimension, mais à rendements très décroissants
(+2,2 → +1,8 → +0,6 points pour `clf_acc_email_axes`) -- **64/640 dimensions
(10%) conservent déjà 94% de la performance du plein embedding** (72,8/77,4).
Dégradation graduelle, pas de chute brutale à aucun point testé -- cohérent
avec (sans le démontrer formellement, faute de confirmation MRL côté F2LLM) un
comportement Matryoshka-like : les dimensions de tête de l'embedding portent
disproportionnellement le signal utile à la classification.

Le NMSE, à l'inverse, ne suit aucune tendance monotone (0,0706 / 0,0803 /
0,0727 / 0,0697) -- vraisemblablement du bruit d'entraînement (un seul run par
point de dimension, pas de répétition sur seed), et de toute façon pas une
métrique directement comparable entre dimensions différentes (la variance du
signal à reconstruire change elle-même avec la dimension d'entrée). `dead_pct`
augmente avec la dimension (0,0% → 0,0% → 0,77% → 3,31%) -- à `D_SAE=8192`
dictionnaire fixe, plus de dimensions d'entrée semble produire plus d'atomes
redondants/inutilisés, une observation secondaire non creusée davantage ici.

**Conclusion pratique** : si le volume de calcul/stockage devient un enjeu pour
Pipeline 2, tronquer à 128 ou même 64 dimensions coûte peu en performance de
classification (perte de 2 à 5 points) pour un gain de calcul/mémoire de 5x à
10x -- un compromis a priori favorable si jamais nécessaire, bien qu'aucun des
runs de ce projet n'ait été contraint par cette ressource à ce jour.

## 32. Audit méthodologique — qualité 16k vs 65k (monosémanticité, pas seulement couverture)

Phase 1.2 de l'audit méthodologique du 2026-08-07 (mon hypothèse explicite :
le dictionnaire 65k a plus de labels Neuronpedia que le 16k
(§17/§29 : couverture 87,8% vs 82,6%), mais une partie pourrait ne pas être
monosémantique -- ex. inventé "ananas, Paris, chaise, colère" mélangeant des
concepts sans rapport). Jamais testé jusqu'ici : la comparaison précédente ne
portait que sur la couverture, jamais sur la qualité des labels eux-mêmes.
Script : `scripts/dictionary_width_quality_audit.py`, entièrement local (CPU,
`local_data/neuronpedia_labels/*.json` déjà en cache, aucun GPU).

**Vérification qualitative préalable** (avant d'écrire le script) : sur un
échantillon de labels contenant une virgule, le 65k contient bien des labels
visiblement incohérents que le 16k ne semble pas produire au même degré --
ex. "FCC ID, agricultural professionals, topological equivalence" ou "than
the, n be, y the, it in" (ce dernier n'est même pas un concept, plutôt un
fragment de titre mal segmenté).

**Mesure 1 — `frac_multi_part`, population ENTIÈRE** (tous les labels
labellisés, pas un échantillon ; label découpé sur virgule/point-virgule/" or
"/" and ") :

| Largeur | n labels | % multi-parties | IC95% |
|---|---|---|---|
| 16k | 13 535 | 49,6% | [48,8%, 50,5%] |
| 65k | 57 551 | 51,2% | [50,8%, 51,6%] |

Écart 65k−16k = +1,6 point, z=3,37, **p=7,4×10⁻⁴** -- mais sur une population
entière (n≈71k au total), la puissance est quasi infinie : même un écart
minime devient "significatif". Le h de Cohen (taille d'effet, indépendant de
n) donne la lecture pertinente : **h=0,032, négligeable** (repère usuel :
h<0,2 = petit effet). Le nombre brut de parties listées dans un label n'est
donc quasiment pas différent entre les deux largeurs.

**Mesure 2 — cohérence sémantique des labels multi-parties, échantillon
n=200/largeur** : embeddings bge-m3 (`src/sae/saev5.py::_embed_bge_m3`, déjà
utilisé ailleurs dans le projet pour ce rôle) de chaque partie d'un label,
similarité cosinus moyenne entre parties. Score bas = parties peu liées entre
elles (candidat polysémantique).

| Largeur | cohérence moyenne | médiane | incohérents (sim<0,3) | IC95% |
|---|---|---|---|---|
| 16k | 0,570 | 0,554 | 0/200 = 0,0% | [0,0%, 1,9%] |
| 65k | 0,560 | 0,550 | 1/200 = 0,5% | [0,1%, 2,8%] |

Mann-Whitney U (H1 : cohérence 65k < 16k) : U=18672, **p=0,125** -- non
significatif. Taux d'incohérence 65k vs 16k : diff=+0,5 point, z=1,00,
**p=0,317**, h de Cohen=0,142 (petit effet, non significatif).

**Conclusion** : l'hypothèse "65k contient significativement plus de labels
polysémantiques que 16k" **n'est PAS confirmée** par ces deux mesures
quantitatives -- les deux largeurs sont statistiquement indistinguables sur
le taux de labels multi-parties ET sur la cohérence sémantique de ces
parties entre elles. L'impression qualitative initiale (labels 65k
visiblement incohérents) semble provenir d'un biais d'échantillonnage
manuel sur quelques exemples frappants, pas d'un écart systématique.

**Limite connue de la mesure** : la mesure 2 capture "plusieurs concepts
RÉELS mais sans rapport" (ex. "ananas, Paris, chaise, colère"), pas
"texte dégénéré/fragment de phrase" (ex. "than the, n be, y the, it in") --
un fragment grammatical peut obtenir une similarité de parties élevée si ses
morceaux sont des mots-outils génériques qui s'embedent de façon similaire,
sans que le label soit pour autant un concept cohérent. Une mesure
complémentaire (ex. score de "labelness" -- le label ressemble-t-il à une
description de concept plutôt qu'à un fragment de phrase coupé) serait
nécessaire pour trancher cette hypothèse secondaire, non implémentée dans
cette passe.

**Décision pratique** : pas de justification, à ce stade, pour préférer 16k à
65k sur un critère de qualité -- le choix 65k déjà en place (`config.py`,
justifié par la couverture Neuronpedia supérieure, §17/§29) reste valide.

## 33. Audit méthodologique — HDBSCAN : UMAP-2D vs raw cosine vs UMAP-nD vs PCA-nD

Phase 1.1 de l'audit méthodologique du 2026-08-07. Point de départ : deux
endroits font tourner HDBSCAN sur une projection UMAP **2D**
(`src/sae/saev5.py::analyze_with_umap`, `src/analysis/cooccurrence.py
::cluster_in_feature_space`) — pratique déconseillée par McInnes et al.
(documentation du package `hdbscan` : réserver la 2D à la visualisation,
faire tourner HDBSCAN sur l'espace original ou une réduction modérée).
`results_v10_emails_main/results.json` montrait déjà un signal faible
(`n_clusters: 3` sur 2177 documents de test).

Script : `scripts/clustering_methodology_audit.py`, sur `p1_all_doc_acts.pt`
déjà en cache (aucune nouvelle extraction Gemma-3). 2177 documents de test,
8893/17408 features actives. Quatre configurations comparées, sweep de
`min_cluster_size` incluant la valeur littérale actuellement en production
(`N_DOCS // 15` = 145) :

| Config | Meilleur DBCV | n_clusters | AMI vs 14 classes connues | Stabilité inter-seed (ARI) |
|---|---|---|---|---|
| UMAP 10D → HDBSCAN | **0,851** | 2 | 0,026 | **[1.0, 1.0, 1.0]** |
| UMAP 20D → HDBSCAN | 0,850 | 2 | 0,026 | [1.0, 1.0, 1.0] |
| UMAP 50D → HDBSCAN | 0,842 | 2 | 0,026 | [0,60, 0,60, 1.0] |
| UMAP 2D → HDBSCAN (production actuelle) | 0,829 | 2 | 0,026 | [1.0, 0,64, 0,64] |
| PCA 10D → HDBSCAN (37,4% variance expliquée) | 0,275 | 3 | 0,022 | N/A (déterministe) |
| PCA 50D → HDBSCAN (58,4% variance expliquée) | 0,064 | 3 | 0,026 | N/A |
| Cosine brut (pas de réduction) → HDBSCAN | 0,061 | 4 | 0,021 | N/A |
| PCA 20D → HDBSCAN (46,0% variance expliquée) | 0,014 | 2 | 0,024 | N/A |

**Trois résultats, dans l'ordre d'importance pratique :**

1. **Aucune configuration ne récupère de structure sémantique significative.**
   L'AMI entre le clustering HDBSCAN (quelle que soit la config) et les 14
   classes connues (axes d'augmentation email) reste ~0,01-0,03 partout,
   y compris pour les meilleures configs par DBCV. Cohérent avec le
   `silhouette=0,006` déjà mesuré en production (séparabilité quasi nulle
   des mêmes 14 classes directement dans l'espace d'activation SAE, métrique
   indépendante). Changer où HDBSCAN s'exécute NE RÉSOUT PAS le problème de
   fond : les activations SAE max-poolées au niveau document ne séparent
   pas ces 14 classes connues, quelle que soit la méthode de clustering
   testée. Les "meilleures" configs par DBCV convergent d'ailleurs vers un
   partage trivial à 2 clusters (un petit groupe compact + tout le reste
   sans bruit) — pas 97 clusters informatifs, un signal d'alerte que DBCV
   seul ne suffit pas à juger la qualité d'un clustering.
2. **PCA est nettement DOMINÉE par UMAP sur ce jeu de données** (piste posée
   explicitement) : DBCV max 0,275 (PCA 10D, 37% de variance expliquée
   seulement) contre 0,83-0,85 pour toute config UMAP, et pire que même le
   cosine brut sans réduction pour PCA 20D/50D. La structure utile aux fins
   de clustering ici est visiblement non-linéaire — une réduction linéaire
   ne la capture pas, contrairement à l'hypothèse initiale que PCA serait
   un choix "plus sûr" pour un algorithme densité-connexe comme HDBSCAN.
   Hypothèse invalidée empiriquement sur ce corpus précis.
3. **UMAP à dimension modérée (10D/20D) domine UMAP-2D** sur DBCV (léger,
   +0,02) ET nettement sur la stabilité inter-seed (ARI=1,0 parfait contre
   0,64-1,0 pour la config 2D actuelle) — confirme la recommandation
   standard McInnes : réserver la 2D à la visualisation, faire tourner
   HDBSCAN sur un embedding UMAP à dimension modérée.

**Changement appliqué** (`src/sae/saev5.py::analyze_with_umap`,
`src/analysis/cooccurrence.py::cluster_in_feature_space`) : HDBSCAN tourne
désormais sur un embedding UMAP 10D dédié (mêmes hyperparamètres sinon),
l'embedding UMAP 2D existant reste calculé séparément pour la visualisation
Plotly uniquement. Gain attendu : reproductibilité des clusters affichés
d'un run à l'autre (ARI 1,0 vs 0,64), PAS une amélioration de la pertinence
sémantique des clusters eux-mêmes (aucune config testée ne l'apporte).

**Limite connue** : ce test porte sur UN SEUL jeu de labels connus (14 axes
d'augmentation email) et un seul corpus. Ne permet pas de conclure que le
clustering document-level échouerait sur toute structure sémantique
possible — seulement sur celle testée ici, avec ce SAE, ce corpus.

## 34. Audit méthodologique — reproductibilité par GROUPE de features vs feature individuelle

Phase 1.4 de l'audit méthodologique du 2026-08-07 (mon hypothèse explicite :
grouper les features par similarité gagnerait en
reproductibilité par rapport à la feature individuelle). Point de départ :
§21 (ablation seed-variance, SEED=42 "run principal" vs SEED=123
`results_v13_ablation_seed123`) montre que seulement 22/78 = 28,2% des
labels de features d'extension INTERPRÉTABLES sont des chaînes de
caractères IDENTIQUES entre les deux seeds.

**Pivot méthodologique documenté** (cf. docstring de
`scripts/feature_group_reproducibility_test.py`) : l'approche prévue
(appariement par corrélation d'activation entre les deux seeds + comparaison
des communautés Louvain de `cooccurrence_graph`) s'est révélée infaisable :
(1) le cache d'activations brutes de `results_v13_ablation_seed123` a été
purgé après le run (seuls `p1_judge_labels_extended.json` et `p1_npmi.pt`
restent) ; (2) même avec les activations, **0 des 150 features jugées par
seed n'apparaît comme nœud du graphe NPMI** déjà en cache dans les deux
seeds -- la sélection par magnitude (`feature_selection_by_magnitude`) et le
filtre de fréquence documentaire de `cooccurrence_graph` ([0.01, 0.5])
ciblent des ensembles quasi disjoints pour des features TopK-sparses
(K_EXTRA=32/1024).

Pivot vers un appariement par **similarité sémantique de label** (embeddings
bge-m3, `_embed_bge_m3`), qui ne nécessite que les labels déjà en cache pour
les deux seeds :

| Niveau | n | Similarité moyenne | Médiane |
|---|---|---|---|
| Feature-à-feature (appariement hongrois individuel) | 68 | 0,820 | 0,810 |
| Groupe-à-groupe (Louvain intra-seed sur similarité de label, puis appariement hongrois des groupes) | 3 groupes (seed 42) / 5 groupes (seed 123) | 0,948 | 0,963 |

Recouvrement EXACT de labels (réplique §21) : 21/68-71 -- cohérent avec les
22/78 du §21 (léger écart de dénominateur : §21 comptait sur l'union des
deux ensembles interprétables, ici sur l'intersection stricte des labels).

Mann-Whitney U (H1 : similarité groupe > similarité feature) : U=124,
**p=0,269 -- non significatif**.

**Lecture** : le sens de l'effet va dans la direction que j'attendais
(0,948 vs 0,820, +0,13 point de similarité cosinus), mais le
test est **fortement sous-dimensionné** -- seulement 3 et 5 groupes
obtenus à partir de 68-71 features labellisées (seuil de similarité
NPMI-like à 0,5 pour le graphe intra-seed), donc n=3 au niveau groupe pour
le test statistique. Impossible de conclure à ce stade que le regroupement
améliore significativement la reproductibilité -- l'hypothèse reste
**plausible mais non confirmée**, et mériterait d'être retestée avec
beaucoup plus de features labellisées (le run final à 1000 features,
Phase 4, donnera un échantillon nettement plus puissant pour ce test
précis).

**Limite connue** : ce test mesure la reproductibilité au niveau du LABEL
sémantique (texte), pas au niveau de l'activation elle-même (impossible à
mesurer directement pour l'extension entre deux seeds différents, cf.
pivot ci-dessus) -- un proxy raisonnable mais indirect.

*(§35 absent : aucun contenu n'a été retiré à cette position, la numérotation
saute directement de §34 à §36.)*

## 36. Audit méthodologique — GemmaScope-2 12b-it publie bien plus que le residual stream layer 24

Vérification factuelle que j'ai demandée ("il n'y a pas que ça
d'accessible sur GemmaScope") avant de lancer un balayage GPU (Phase 3).
`config.py:92` utilise UNIQUEMENT `resid_post`, layer 24, largeur 65k,
`l0_medium` -- jamais questionné jusqu'ici (choix justifié seulement par la
couverture de labels Neuronpedia, §17/§29, jamais par un critère
d'interprétabilité ni par une vérification de ce qui existe réellement).

Interrogation directe de l'API HuggingFace Hub (`list_repo_files`) sur
`google/gemma-scope-2-12b-it` (3067 fichiers) :

| Type de hook-point | Layers "curés" (l0 small/medium/big) | Layers "_all" (couverture complète) |
|---|---|---|
| `resid_post` (utilisé actuellement) | 12, 24, 31, 41 | 0-47 (48 layers) |
| `attn_out` | 12, 24, 31, 41 | 0-47 |
| `mlp_out` | 12, 24, 31, 41 | 0-47 |
| `transcoder` (MLP-in → MLP-out, technique différente d'un SAE classique) | 12, 24, 31, 41 | 0-47 |
| `crosscoder` (features partagées inter-layers/modèles, Anthropic) | -- | -- (40 fichiers, structure différente) |

**Confirmation factuelle de mon hypothèse** : GemmaScope-2 publie
des SAEs pré-entraînés sur l'attention (`attn_out`) et le MLP (`mlp_out`) en
plus du residual stream, à 4 layers "curés" (12, 24, 31, 41) representant
respectivement les 25%/50%/65%/85% de la profondeur du modèle (48 layers),
et sur l'intégralité des 48 layers pour qui veut une couverture complète.
Le projet n'a jamais chargé ni évoqué `attn_out`/`mlp_out`/`transcoder`
avant cette vérification (`grep` sur ces termes : 0 résultat avant ce
commit).

**Portée du prochain pas (Phase 3, budget GPU/SLURM)** : comparer, à
protocole d'évaluation identique (même corpus, même juge, même taille
d'échantillon que les ablations existantes, ex. n=150) :
1. `resid_post` aux 4 layers curés (12/24/31/41) -- le layer 24 actuel
   est-il vraiment optimal, ou seulement "suffisamment bon" ?
2. `resid_post` vs `attn_out` vs `mlp_out` à layer fixe (24, le choix
   actuel) -- le residual stream domine-t-il vraiment l'attention/le MLP
   pour l'interprétabilité des concepts métier (urgence, réclamation...) ?

`transcoder`/`crosscoder` volontairement EXCLUS de ce premier balayage :
techniques structurellement différentes d'un SAE standard (pas un simple
changement de point d'extraction), nécessiteraient une adaptation du
pipeline d'extraction/labellisation plutôt qu'un simple changement de
config -- à documenter séparément si jugé prioritaire après le balayage
resid_post/attn_out/mlp_out.

## 37. Audit méthodologique — fuite lexicale dans le corpus augmenté : `clf_acc_email_axes` mesure-t-il vraiment le SAE ?

Piste que je n'avais pas identifiée au départ, trouvée en creusant de mon
côté plutôt qu'en travaillant uniquement la liste de points déjà nommés
(cf. consigne du 2026-08-07 : élargir l'audit au-delà des exemples donnés).

**Hypothèse** : `clf_acc_email_axes` (sonde logistique 5 plis sur les
activations SAE, `src/analysis/metrics.py::downstream_classification`,
14 classes axis__level) est citée partout dans ce dépôt comme preuve que
les codes latents séparent bien émotion/urgence/registre/original
(93,5% rapporté). Mais le corpus augmenté est généré
par un LLM sous CONTRAINTE DE STYLE explicite par axe/niveau
(`src/data/augmentation.py::AXES`) — si le générateur retombe sur des
formulations quasi figées par instruction (biais connu des LLM sous
contrainte de style), un classifieur pourrait atteindre une haute accuracy
en repérant ces tics lexicaux de génération, pas un signal sémantique
capturé par le SAE.

**Vérification 1 — templating lexical, population du corpus train**
(n-grammes les plus fréquents par classe, `local_data/emails/augmented_mails.jsonl`
non rejetés) : **89,8% des documents, en moyenne sur les 14 classes,
contiennent au moins un des 5 trigrammes les plus fréquents de leur PROPRE
classe** — jusqu'à 100% pour `registre__soutenu` ("de bien vouloir"),
99,8% pour `registre__standard` ("madame monsieur je"), 97,0% pour
`urgence__panique` ("je vous prie"). Jamais mesuré ni discuté jusqu'ici.

**Vérification 2 — le test décisif** : même protocole EXACT que
`downstream_classification` (StratifiedKFold 5 plis, `random_state=42`,
`LogisticRegression`, mêmes 14 classes, même corpus train) mais sur des
features **TF-IDF du texte brut** (1-3 grammes, 20 000 features max) au
lieu des activations SAE — aucune information sémantique explicite, juste
la présence de mots/expressions. Script :
`scripts/augmentation_lexical_leakage_audit.py`.

| Sonde | Accuracy (14 classes) |
|---|---|
| SAE (activations, rapporté) | 93,5% |
| **TF-IDF texte brut (0 sémantique)** | **87,0%** |
| Écart SAE − lexical | +6,5 points |

**Lecture** : un classifieur qui ne voit QUE la présence de mots/expressions
atteint déjà 87,0% sur les 93,5% rapportés pour le SAE -- soit **93% du
signal rapporté comme preuve de séparation sémantique est déjà présent
dans le texte brut, sans aucune compréhension du contenu**. Le SAE ajoute
un gain réel mais modeste (+6,5 points), pas la démonstration forte
suggérée par le chiffre 93,5% cité isolément. La métrique n'est pas fausse,
mais la citer seule surestime ce qu'elle démontre : elle ne permet pas de
distinguer "le SAE
comprend l'urgence" de "le générateur d'augmentation a des tics de style
par instruction, et le SAE (comme n'importe quel modèle) les capte
partiellement".

**Limite de ce test** : ne teste PAS si le signal sémantique existe aussi
sur les mails ORIGINAUX (non augmentés, sans biais de génération LLM) —
seulement sur le corpus train dominé par l'augmentation (41 176 docs, dont
39 879 augmentés). Un test propre nécessiterait une classification
emotion/urgence/registre sur des mails naturellement variés (pas générés
sous contrainte de style), hors de portée sans nouvelle annotation manuelle.

**Recommandation** : ne plus citer `clf_acc_email_axes=93,5%` seul comme
preuve de compréhension sémantique du SAE sans mentionner le baseline
lexical (87,0%) à côté.

## 38. Audit méthodologique — le garde-fou qualité de l'augmentation rejette massivement 2 classes sur 13, jamais remarqué

Deuxième piste trouvée en creusant de mon côté (même élargissement de
l'audit que §37). Le taux de rejet global de l'augmentation (11,7%,
`src/data/augmentation.py::validate`) est cité dans le dashboard et le
rapport comme un chiffre unique ("garde-fou qualité de l'augmentation --
11,7% de rejet"). Je n'ai trouvé nulle part une décomposition par
axe/niveau -- vérifié en creusant.

**Décomposition par classe** (`local_data/emails/augmented_mails.jsonl`,
45 240 tentatives) : le taux de rejet n'est PAS uniforme du tout.

| Classe | Taux de rejet |
|---|---|
| `orthographe__degrade_fort` | **59,6%** |
| `emotion__impatience` | **47,2%** |
| Les 11 autres classes | 1,1% à 6,8% (moyenne ~4,1%) |

Test à deux proportions (`src.analysis.stats.two_proportion_test`) : ces 2
classes (3715/6960 rejets, 53,4%) vs les 11 autres (1576/38280, 4,1%) —
**h de Cohen = 1,23 (effet énorme), z=117,6, p≈0**. Pas un bruit d'échantillonnage :
un trait structurel de la génération jamais documenté.

**Cause identifiée** : `validate()` (`augmentation.py:106-120`) rejette si
`length_ratio` (longueur variante / longueur original) sort de [0,4 ; 2,5].
**94-96% des rejets de ces deux classes ont un ratio < 0,4** (texte généré
environ 3× plus court que l'original, médiane ≈0,33) -- Gemma-3, quand on
lui demande une orthographe très dégradée ou de l'impatience, écrit
systématiquement des messages nettement plus courts que l'original, et le
garde-fou de longueur (calibré pour détecter des troncatures/générations
défaillantes, pas pensé par axe) les rejette presque tous.

**Conséquence directe, jamais reliée à cette cause** : classes
finales déséquilibrées dans le corpus train -- `orthographe__degrade_fort`
n'a que 1406 exemples utilisables contre ~3300-3400 pour la plupart des
autres classes (confirmé dans l'échantillon de `scripts
/augmentation_lexical_leakage_audit.py`, §37), `emotion__impatience` 1839.

**Lecture à double tranchant** : (a) risque de biais de sélection --
"impatience" et "orthographe très dégradée" écrites BRIÈVEMENT sont
peut-être stylistiquement plus authentiques (un client impatient écrit
court dans la vraie vie) que les rares variantes qui passent le filtre de
longueur, ce qui pourrait systématiquement écarter les variantes les plus
réalistes de ces deux axes au profit de versions artificiellement
rallongées ; (b) mais ce n'est PAS nécessairement un problème pour les
métriques déjà rapportées (classification, interprétabilité) : moins
d'exemples pour 2 classes sur 14 n'invalide pas les conclusions existantes,
et `StratifiedKFold` gère le déséquilibre. Aucune conclusion déjà publiée
dans ce dépôt ne dépend d'un déséquilibre supposé inexistant.

**Recommandation, non appliquée dans cette passe** (change le corpus
augmenté, effet en cascade sur tous les runs -- décision à prendre
consciemment, pas en aparté d'un audit) : soit desserrer le seuil bas de
`length_ratio` spécifiquement pour les axes où un texte plus court est
attendu par construction (`impatience`, `degrade_fort`), soit accepter le
déséquilibre en le documentant explicitement au lieu de le citer comme "un"
taux de rejet global qui masque une hétérogénéité de x50 entre classes.

## 39. Bug corrigé — `facts_lost` (garde-fou d'augmentation) rejetait du pur reformatage, pas de la perte réelle

Troisième piste trouvée en creusant, dans le prolongement direct du §38
(même thème : le garde-fou qualité de l'augmentation rejette pour de
mauvaises raisons). `facts_lost` est la 2e cause de rejet la plus fréquente
(1083/5291 = 20,5% de tous les rejets, `augmentation.py::validate`) : la
variante est rejetée si un "fact" numérique (téléphone, montant, date)
présent chez le parent n'apparaît plus, MOT POUR MOT, chez la variante.

**Preuve directe** (pas une supposition à partir des données agrégées --
test direct de `_facts()` sur des paires construites, avant fix) :

| Cas | `facts_lost` déclenché avant fix ? |
|---|---|
| Téléphone reformaté (`0476356490` → `0476 35 64 90`) | **Oui (faux positif)** |
| Téléphone re-ponctué (`0476 35 64 90` → `04.76.35.64.90`) | **Oui (faux positif)** |
| Date sans zéro de padding (`18/7/13` → `18/07/2013`) | **Oui (faux positif)** |
| Montant en virgule décimale FR au lieu du point (`20.73€` → `20,73 €`) | **Oui (faux positif)** |
| Code postal identique | Non (correct) |

Cause racine double : (1) `_FACT_RE` ne capturait un numéro de téléphone
espacé/ponctué que par FRAGMENTS de 4+ chiffres contigus (`\b\d{4,}\b`) --
"0476 35 64 90" ne produisait que le fragment "0476", jamais le numéro
complet ; (2) `_facts()` comparait les sous-chaînes CAPTURÉES littéralement,
sans normaliser séparateurs, padding ou style décimal. Un échantillon des
raisons `facts_lost` stockées (`local_data/emails/augmented_mails.jsonl`)
montre exactement ce pattern : fragments de téléphone isolés
(`['0567', '417756']`), dates sans padding (`'18/7/13'`, `'24/2/2023'`),
montants en point (`'20.73€'`) -- cohérent avec des reformatages légitimes
plutôt que des pertes réelles, sur la majorité des cas inspectés.

**Fix appliqué** (`src/data/augmentation.py`) : `_FACT_RE` capture désormais
un numéro de téléphone FR complet quel que soit son espacement/ponctuation
(`\b0\d(?:[ .\-]?\d{2}){4}\b`, en tête d'alternative pour primer sur le
fallback générique) ; `_normalize_fact()` (nouveau) normalise avant
comparaison : séparateurs retirés pour les nombres purs, virgule décimale
unifiée pour les montants, année 2 chiffres étendue à 4 chiffres pour les
dates. 6 tests (`tests/test_augmentation_facts_normalization.py`) : les 4
faux positifs ci-dessus corrigés, ET vérification que la perte réelle
(numéro de contrat disparu, date effectivement changée) reste détectée.

**Limite non corrigée, documentée plutôt que masquée** : une date réécrite
en toutes lettres ("18 juillet 2013") n'est pas normalisable sans parsing
NLP des noms de mois -- reste flaguée `facts_lost` (choix conservateur :
mieux vaut sur-rejeter que risquer un faux négatif non vérifiable).

**Portée du fix** : corrige `validate()` pour toute future génération
d'augmentation -- **ne change PAS rétroactivement** le corpus déjà généré
(`augmented_mails.jsonl`, 45 240 tentatives déjà figées). Combien des 1083
rejets `facts_lost` existants étaient des faux positifs reste inconnu sans
regénération (nécessite GPU, hors de portée de cette passe) -- piste
ajoutée au plan (cf. section Prochaines étapes) plutôt que traitée ici.

## 40. Validation de `find_interesting_pairs` par injection synthétique (résultat positif)

Piste "reste à faire" du rapport de stage (perspectives, point 8) :
valider `find_interesting_pairs`/`cooccurrence_graph` (`src/analysis/
cooccurrence.py`) contre un signal connu, à la manière de la validation par
injection synthétique du papier de référence interp_embed (Appendix E.2). Le
biais réel "Objet:" n'est plus reproductible tel quel (déjà corrigé dans le
pipeline de génération) -- réplique le PRINCIPE : `scripts/
interesting_pairs_synthetic_validation.py` construit un corpus synthétique
(2000 docs, 200 features, bruit de fond sparse aléatoire) avec UNE paire de
features injectée en co-occurrence parfaite (actives ensemble dans ~40% des
docs, jamais l'une sans l'autre) et des labels délibérément dissimilaires
(embeddings orthogonaux), plus des features "quasi-doublons" de contrôle
négatif (labels proches, ne doivent PAS être remontées).

**Résultat : validation positive.** La paire injectée est retrouvée
(NPMI=1,000, seule arête du graphe formée sur ce signal de fond aléatoire) et
c'est l'unique paire "intéressante" détectée (rang 1/1) -- aucun faux positif
parmi les features de contrôle à labels proches. `tests/
test_interesting_pairs_synthetic.py` : régression ajoutée.

**Limite** : valide que la fonction fonctionne sur un signal fort et propre,
pas sa sensibilité sur un signal faible/bruité comme celui rencontré en
pratique (§16.3 : seulement 3 paires "intéressantes" trouvées sur le corpus
réel, dont 2/3 avec une feature non labellisée) -- un test de sensibilité au
bruit (dégrader progressivement le taux de co-occurrence injecté) serait le
prolongement naturel, non fait ici.

## 41. Erreur juge vs erreur SAE : lecture indépendante sur 30 features non-interprétées

Le résidu non-interprété (~55-59%) est attribué depuis §12 soit au protocole
de jugement (bruit d'ordre, §13.1 : 31,3% de décisions instables), soit aux
features elles-mêmes -- jamais tranché. Sur les 30 features (échantillon
aléatoire des 82 rejetées, §15/39) je juge moi-même, à partir des seuls MOTS
CIBLÉS (`<<...>>`) des exemples positifs -- pas du label LLM proposé, pour
éviter l'ancrage -- si un déclencheur cohérent existe.

**Verdict, sans détour** : sur 30, **8 erreurs juge nettes** (déclencheur
cohérent et évident : "vous demande de/d'annuler/d'examiner..." pour
`F16497` ; "J'attends/J'attendais" pour `F16549` ; nom de famille en fin de
message pour le triplet `F16720`/`F16949`/`F16852`...), 3 bordeline (motif
réel mais faible), et **19 erreurs SAE** -- les mots ciblés ne partagent
AUCUN déclencheur cohérent (`F16662` : "le", "un", "ce", "du" -- de la
ponctuation/déterminants comme "concept" ; `F17221` labellisée
"mécontentement" mais un exemple cible "bénéficié", sentiment positif
contradictoire ; `F16546` : cinq "l'..." génériques sur six). **Majorité
nette (63%) côté SAE, pas côté juge.** Le résidu non-interprété n'est donc
PAS principalement un artefact du protocole de jugement -- la plupart de ces
features n'ont simplement pas de concept token-level cohérent à trouver,
juge parfait ou non.

**Limite** : échantillon de 30/82 (une passe complète sur les 82 donnerait un
intervalle plus serré), et je voyais le label contrastif proposé en même
temps que les exemples (risque d'ancrage) -- atténué par le fait que
plusieurs verdicts CONTREDISENT frontalement le label proposé (`F17221`,
`F16411`, `F17312`) plutôt que de le confirmer par défaut.

## 42. Juge par échantillonnage (temp=0,7) : stable, contrairement au réordonnancement

`scripts/judge_sampling_ensemble_test.py` : 5 générations par feature, MÊME
ordre d'exemples à chaque fois (isole la variance de génération de la
variance d'ordre déjà mesurée en §13.1). Taux single-shot 45,3%, vote
majoritaire 52,7% (+7,4 points), **accord moyen 0,992** entre les 5 tirages.

**Contraste net avec §13.1** (réordonnancement seul : 30,7% de décisions
unanimes) : la variance de température, à ordre fixe, est quasi nulle --
le juge est stable face à l'échantillonnage, instable face à l'ordre de
présentation. Le bruit du protocole est donc spécifiquement un biais de
POSITION, pas un artefact de décodage stochastique. Confirme et précise
§13.1 plutôt que de le contredire.

## 43. Séparation juge/extraction : gemma-3-4b-it juge 2x moins de features interprétables

`scripts/judge_model_separation_test.py` : mêmes 150 features, mêmes
exemples, juge alternatif gemma-3-4b-it (déjà en cache) au lieu de
gemma-3-12b-it (le modèle habituellement rechargé deux fois -- extraction
ET jugement, jamais un vrai second modèle jusqu'ici).

| Juge | Taux interp. |
|---|---|
| gemma-3-12b-it (= modèle d'extraction) | 45,3% |
| gemma-3-4b-it | **24,7%** |

Accord 56,7% (85/150), 48 features basculent interprétable→non avec le
petit juge contre seulement 17 dans l'autre sens -- **asymétrie nette, pas
un simple bruit symétrique**. Ne tranche PAS, avec cette seule comparaison,
entre deux explications concurrentes : (a) gemma-3-4b-it est simplement un
moins bon juge (capacité de raisonnement inférieure sur une tâche
odd-one-out à 10 items), ou (b) gemma-3-12b-it bénéficie d'un biais de
auto-préférence en jugeant ses propres représentations internes. Un juge
de capacité comparable mais famille différente (Llama, Mistral, Qwen)
serait nécessaire pour isoler (b) -- non fait ici (aucun modèle de cette
taille/famille en cache local). **Ce que cette comparaison établit
fermement** : le chiffre 45,3% n'est PAS robuste au choix du juge -- il ne
doit plus être cité comme une propriété intrinsèque du SAE sans préciser
le juge utilisé.

## 44. Revue externe multi-perspective (avocat du diable) sur le rapport de stage

Panel de 5 reviewers indépendants et aveugles l'un à l'autre (plugin
`academic-research-skills`), chacun avec accès direct au dépôt pour vérifier
les affirmations plutôt que faire confiance aux auto-citations du rapport.
Décision : Major Revision, tranchée par deux constats CRITICAL de l'avocat
du diable (C1/C2 ci-dessous), tous deux résolus depuis (§46, §48/§50/§52).

Deux constats CRITICAL validés par vérification directe : (1) l'affirmation
centrale du rapport (20,0%→45,3%, corpus generic n=10 vs corpus emails
n=150, §12) n'avait jamais été testée statistiquement -- recalcul indépendant
z≈1,56, p≈0,12, non significatif au seuil que le rapport applique partout
ailleurs ; (2) ~92% du corpus d'entraînement "corrigé" est généré par le même
modèle (gemma-3-12b-it) qui sert aussi d'extracteur et de juge -- boucle
auto-référentielle nommée en une phrase dans le rapport mais jamais
quantifiée ni bornée. Constats corroborés par plusieurs reviewers
indépendants : §37 (fuite lexicale, ~93% du signal `clf_acc_email_axes`
reproductible sans SAE) et §43 (dépendance au juge) jamais mentionnés dans
le rapport bien que déjà présents dans ce fichier ; erreur de transcription
z=2,91 vs z=2,86 (§19) ; validation exclusivement sur corpus synthétique pour
un objectif de déploiement en contexte réglementé, jamais revisitée en
limites. Corrections vérifiées et à faible risque appliquées directement au
rapport (z, description ρ_interp, caveats §37/§43/C1, citations manquantes
SAEBench et Chanin et al. 2024, renvoi §3.5.3→§3.12). Restent à la décision
de l'utilisateur : test de confirmation à n apparié pour le constat (1), run
juge croisé à plus grande échelle pour borner (2).

## 45. Réplication multi-seed `K_EXTRA=5` (jobs 42685/42686) : direction confirmée, significativité toujours pas atteinte

Deux seeds supplémentaires (7, 99) du run `K_EXTRA=5` (§25, seed 42 original à
54,7%), même protocole, mêmes 150 features :

| Seed | Taux interp. |
|---|---|
| 42 (original, §25) | 54,7% (82/150) |
| 7 | 50,0% (75/150) |
| 99 | 55,3% (83/150) |
| K_EXTRA=32 (baseline, seed 42 unique) | 45,3% (68/150) |

**Direction cohérente sur les 3 seeds** (toutes au-dessus du baseline
K_EXTRA=32), contrairement à un résultat de bruit pur qui basculerait des deux
côtés. Test z groupé (240/450 vs 68/150) : **z=1,70, p≈0,089 — toujours en
dessous du seuil conventionnel |z|>1,96**, et ce chiffre est probablement
optimiste : il traite les 3 seeds comme un échantillon groupé homogène en
ignorant la corrélation intra-seed, alors que le baseline K_EXTRA=32 n'a
qu'un seul seed pour comparer (pas de réplication symétrique). Un modèle à
effets mixtes (seed en effet aléatoire) serait la bonne façon de trancher,
non fait ici faute de ressources supplémentaires justifiées pour ce seul
point.

**Verdict honnête** : la réplication tend à renforcer la piste K_EXTRA=5
plutôt qu'à l'infirmer (3/3 seeds dans la même direction), mais ne suffit pas
à la faire passer d'"hypothèse à répliquer" à "résultat établi" -- le calcul
z groupé reste sous le seuil, et sans réplication symétrique du baseline
K_EXTRA=32 la comparaison reste asymétrique. Réplication du volume-25M
(jobs 42687/42688) échouée sans traceback (voir §47), à resoumettre.

## 46. Test confirmatoire C1 (job 42748) : effet domaine confirmé à n apparié

Suite de §44/§45. Réplique le baseline pré-correctif (corpus generic
energy/sports/support, `results_v9_full`) à n=150 au lieu de n=10, checkpoint
`p1_frozen_core_d1024_k32.pt` réutilisé (pas de réentraînement), seule
l'extraction Gemma-3-12B + le jugement odd-one-out sont recalculés
(`CONFIRMATORY_DOMAIN_BASELINE=1`, `src/sae/saev5.py` __main__, branche
ajoutée guardée par env var).

| Corpus | n | Taux interp. |
|---|---|---|
| Generic (confirmatoire, n apparié) | 150 | 30,0% (45/150) |
| Emails (run principal) | 150 | 45,3% (68/150) |

**z=2,74, p≈0,006 — significatif**, contrairement à la comparaison originale
n=10 vs n=150 (z≈1,56, p≈0,12, non significative). **Le constat CRITICAL C1
de la revue externe est résolu favorablement à l'affirmation centrale du
rapport** : l'effet domaine est réel et maintenant démontré à échantillon
apparié, pas seulement suggéré par une comparaison sous-puissante. Le taux
absolu sur corpus generic est plus bas ici (30,0%) que dans le run original
à n=10 (20,0%), cohérent avec le fait que n=10 était trop bruité pour être
une estimation fiable dans un sens comme dans l'autre -- l'écart avec le
corpus emails reste net et maintenant statistiquement fondé (45,3% vs 30,0%,
+15,3 points).

## 47. Échecs d'infrastructure sur 5 jobs

Incident d'infrastructure, détail : `docs/ops_journal.md` §47.

## 48. Test C2 (gratuit, CPU) : l'origine des exemples positifs ne prédit pas l'interprétabilité

Suite au constat CRITICAL C2 (boucle auto-référentielle juge/générateur) :
test à coût nul, reproduisant
exactement la sélection déterministe des exemples positifs de
`build_feature_examples_with_control` (`src/sae/judge.py`) à partir des
activations déjà en cache (`p1_all_doc_acts_ext_d1024.pt`), pour retrouver
l'origine (mail original vs variante augmentée générée par Gemma-3-12b-it)
de chacun des 9 exemples positifs présentés au juge, pour les 150 features
déjà jugées de `results_v10_emails_main`.

**Hypothèse testée** : si le juge reconnaît son propre style génératif
plutôt que de comprendre un concept réel, les features jugées
interprétables devraient tirer leurs exemples positifs de façon
disproportionnée du texte généré par Gemma (augmenté) plutôt que des mails
originaux.

| Groupe | n | frac. moyenne d'exemples "augmentés" |
|---|---|---|
| Features interprétables | 68 | 95,10% |
| Features non interprétables | 82 | 92,68% |
| Taux de base du corpus (augmenté/train) | — | 91,99% |

Mann-Whitney U=2974,0, **p=0,418 — non significatif**. Les deux groupes
dépassent légèrement le taux de base du corpus (cohérent avec le fait que
les features actives sur des exemples courts/stylés sont plus faciles à
retrouver dans un corpus déjà dominé à 92% par du texte augmenté), et la
direction de l'écart (interprétables > non-interprétables) va bien dans le
sens que C2 prédirait, mais l'écart de 2,4 points n'est pas distinguable du
bruit sur cet échantillon.

**Verdict honnête** : ce test précis ne trouve pas la signature spécifique
de C2 qu'il cherchait -- **affaiblit** le constat sans le **résoudre**. Il ne
teste qu'un mécanisme étroit (dépendance par feature à l'origine
document-par-document des exemples) ; il ne teste ni un éventuel biais de
style diffus non capturé par l'origine du document, ni le côté génération du
corpus. Les deux tests qui trancheraient réellement C2 (juge cross-famille,
régénération partielle du corpus avec un modèle différent) restent à faire
-- décision utilisateur en attente sur le budget de calcul.

## 49. Bug corrigé — hook attn_out/mlp_out : mauvais chemin d'attribut sur le wrapper multimodal Gemma3

Complète §47 : `resid_post` fonctionnait (jobs 42812/42813, layers 12/31
terminés avec succès), mais `attn_out`/`mlp_out` (42815/42816) échouaient
avec `AttributeError: 'Gemma3Model' object has no attribute 'layers'` --
pas un problème de poids manquants (déjà corrigé en §47), un vrai bug de
code. `AutoModelForCausalLM.from_pretrained` sur `gemma-3-12b-it` charge la
classe multimodale `Gemma3ForConditionalGeneration` : `llm.model` est un
`Gemma3Model` (wrapper vision+texte), dont les couches du transformer sont
sous `llm.model.language_model.layers`, pas `llm.model.layers` (qui
n'existe que sur `Gemma3TextModel`/`Gemma3ForCausalLM`, la classe texte
seul). Le chemin `resid_post` ne touchait jamais `.model.layers`
(`output_hidden_states=True` seul), d'où le bug resté invisible jusqu'à
l'usage réel des hooks `attn_out`/`mlp_out` cette session. Corrigé dans
`src/sae/saev5.py` (deux occurrences, `register_forward_pre_hook`/
`register_forward_hook`). Jobs resoumis (42871, 42872).

**Bug additionnel trouvé en resoumettant** (`attn_out`, job 42871) : une fois
le hook lui-même corrigé, l'assertion de forme plantait --
`shape captée torch.Size([4, 512, 4096]) != D_MODEL=3840`. `D_MODEL` (residual
stream, `text_config.hidden_size`) est correct pour `resid_post`/`mlp_out`
(confirmé : `mlp_out` capte bien du 3840-d) mais **pas** pour `attn_out`, qui
capte l'entrée de `self_attn.o_proj` -- en amont de la projection multi-head
vers `hidden_size`, donc une dimension différente (4096, confirmé
empiriquement sur `w_enc.shape` du SAE téléchargé). Corrigé en dérivant la
dimension attendue du SAE préentraîné déjà chargé (`pretrained_sae.cfg.d_in`)
plutôt que du constant global `D_MODEL`, aux 3 endroits concernés (buffer de
réservoir, reconstruction depuis fragments, assertion). `FrozenCoreResidualSAE`
utilisait déjà `core_sae.cfg.d_in` correctement -- seul le code d'extraction
brute dans `saev5.py` avait le bug. 41/41 tests toujours verts après le
correctif. `attn_out` resoumis.

## 50. C2 (suite) : re-jugement faisable à 100% + le contrôle core affaiblit encore le constat

Deux résultats complémentaires à §48, calculés sur les activations déjà en
cache (`p1_all_doc_acts_ext_d1024.pt`), zéro calcul GPU :

**Faisabilité d'un re-jugement "original uniquement"** : pour les 150
features déjà jugées, **150/150 ont au moins 9 candidats positifs provenant
exclusivement de mails originaux** (pas de texte augmenté/généré par Gemma) --
un re-jugement complet, sans aucune dépendance au texte généré, est donc
possible sur l'échantillon entier, pas seulement un sous-ensemble.

**Contrôle par les features core (Neuronpedia, vérité indépendante)** :
86 features core (labellisées indépendamment par DeepMind, jamais influencées
par notre juge ni notre corpus d'entraînement) tirent en moyenne 94,25% de
leurs exemples positifs du texte augmenté sur NOTRE corpus -- **autant, voire
légèrement plus**, que les features d'extension (93,78%). Mann-Whitney
U=7215,5, p=0,069 (limite mais non significatif, et dans le sens opposé à ce
que prédirait C2 si l'effet existait sur les features d'extension
spécifiquement). Confirme l'interprétation de §48 : le taux élevé
d'exemples "augmentés" est une propriété de la composition du corpus (92%
augmenté), pas une signature spécifique de la boucle juge/générateur de ce
projet -- même des features totalement étrangères à cette boucle présentent
le même pattern.

**Prochaine étape, directement actionnable et bon marché** (SAE déjà
entraîné réutilisé, même juge, seule la sélection d'exemples change) :
re-jugement réel des 150 features avec Gemma-3-12b-it restreint aux
9 meilleurs exemples originaux uniquement. Si le taux d'interprétabilité
reste proche de 45,3%, c'est la preuve directe la plus forte possible contre
C2 sans changer ni modèle ni corpus. Nécessite un job SLURM (juge LLM,
GPU) mais aucun réentraînement ni téléchargement.

## 51. Balayage layer resid_post (12/31/41) : layer 31 significativement meilleur que le layer 24 par défaut

Complète §36 (disponibilité confirmée) et §49 (bug de dimension corrigé
avant que ces jobs puissent tourner). Même protocole que le run principal
(150 features jugées, corpus emails+augmentés, Gemma-3-12b-it), seul
`LAYER` varie (12, 31, 41 vs 24 par défaut) :

| Layer | Taux interp. | z vs layer 24 (baseline, 68/150) |
|---|---|---|
| 12 | 45,3% (68/150) | z=0,00, p=1,000 |
| 24 (baseline, run principal) | 45,3% (68/150) | — |
| 31 | **58,0% (87/150)** | **z=2,20, p=0,028** |
| 41 | 52,7% (79/150) | z=1,27, p=0,204 |

Layer 31 est le seul écart nominalement significatif de ce balayage
(+12,7 points). Comme pour tout le chapitre d'ablations (aucune correction
multi-tests appliquée, cf. §30/audit externe), ce résultat est à lire comme
une piste à répliquer sur un second seed avant d'être traité comme établi,
pas comme une conclusion déjà solide -- mais c'est la première fois que le
choix par défaut du layer 24 (retenu uniquement pour sa couverture
Neuronpedia, jamais pour un critère d'interprétabilité, cf. Chapitre 1 du
rapport) apparaît potentiellement sous-optimal sur le critère qui compte
réellement pour ce projet.

## 52. C2 RÉSOLU : re-jugement original-only (job 42878) -- 44,7% vs 45,3%, écart non significatif

Suite de §48/§50. Réutilise le SAE d'extension déjà entraîné et le même juge
(Gemma-3-12b-it), exemples positifs restreints aux mails originaux
uniquement (`scripts/c2_original_only_rejudge.py`, faisabilité 150/150
confirmée en §50) :

| Condition | Taux interp. |
|---|---|
| Référence (exemples mixtes, run principal) | 45,3% (68/150) |
| Original uniquement (0% texte généré par Gemma) | **44,7% (67/150)** |

**z=-0,12, p=0,908 -- écart non significatif, quasiment nul.** Retirer
COMPLÈTEMENT le texte généré par Gemma des exemples présentés au juge ne
change pas le taux agrégé. Au niveau feature individuelle : 55,3% d'accord
(83/150), 44,7% de bascule -- taux de bruit comparable ou supérieur au
réordonnancement (§13.1, 31,3%) et à la langue (§13, 38,6%), cohérent avec le
constat déjà établi que le statut d'une feature isolée est bruité sous
quasi toute perturbation de surface, alors que le taux agrégé reste stable.

**C2 est résolu.** Les trois tests (§48 corrélation origine/interprétabilité,
§50 contrôle par features core, §52 re-jugement causal direct) convergent :
la boucle auto-référentielle juge/générateur n'explique pas le taux
d'interprétabilité mesuré. Contrairement à C1 (qui a confirmé l'effet
domaine du rapport), C2 est résolu dans le sens **négatif** -- l'hypothèse de
l'avocat du diable n'est pas soutenue par les données. Rapport mis à jour
(`04_limites_et_perspectives.md`, "Facteurs non contrôlés dans le corpus
augmenté").

## 53. Hook-point resid_post vs attn_out vs mlp_out (layer 24) : mlp_out significativement meilleur qu'attn_out

Complète §36/§49/§51 -- dernier volet du balayage hook-point de la Phase 3.
Même protocole (150 features jugées, corpus emails+augmentés), `LAYER=24`
fixe, seul `HOOK_TYPE` varie :

| Hook-point | Taux interp. | z vs resid_post (baseline, 68/150) |
|---|---|---|
| `resid_post` (baseline, run principal) | 45,3% (68/150) | — |
| `mlp_out` | 52,7% (79/150) | z=1,27, p=0,204 |
| `attn_out` | 35,3% (53/150) | z=-1,77, p=0,078 |

Aucun des deux écarts individuels vs `resid_post` n'atteint la
significativité conventionnelle, mais **`mlp_out` vs `attn_out` directement
est net et significatif : z=3,02, p=0,0025**. Cohérent avec un a priori
raisonnable : `attn_out` capte l'entrée de `self_attn.o_proj`, un espace
pré-projection multi-head moins "digéré" sémantiquement que le residual
stream, tandis que `mlp_out` (sortie du MLP, déjà reprojetée vers
`hidden_size`) en reste proche. Anecdote illustrative : la feature choisie
pour la démo de steering sur `attn_out` s'est labellisée "variable
assignment `p =`" -- un concept de nature différente de ce qu'on voit
typiquement sur `resid_post`/`mlp_out`, cohérent avec un espace moins
sémantiquement structuré à ce point du réseau.

**Conclusion du balayage complet (layer + hook-point, §51+§53)** : aucune
configuration alternative testée à ce jour ne bat `resid_post`/layer 24 de
façon indiscutable une fois toutes les comparaisons prises ensemble (layer
31/resid_post reste la seule significative du lot, §51, et devrait être
répliquée avant adoption) ; `attn_out` est le point d'extraction le moins
prometteur des cinq configurations testées.

## 54. Infrastructure de diagnostic + fix racine du blocage volume-200M

Initiative demandée pour donner au dépôt les outils de monitoring nécessaires
(courbes d'entraînement, balayages consolidés) avant toute reproduction fidèle
des papiers de référence — code uniquement, aucun rerun coûteux à ce stade.

**Réservoir de résidus : RAM plate → memmap disque** (`src/sae/saev5.py`,
`open_mmap_reservoir`). Root cause du blocage "volume-200M systématiquement
rejeté" diagnostiqué : `torch.empty(N_TOKENS_EXTRA_TRAIN, d_in, ...)` allouait
tout le réservoir en RAM anonyme (200M tokens × 3840 × 2 octets ≈ 1,4 To),
forçant `--mem=1800G` sur des nœuds à ~2 To — une demande proche de la
capacité totale du nœud, jamais satisfaite simultanément sur un cluster
partagé (`sacct` : jobs 41658/42145/42694, tous restés `CANCELLED`/`FAILED`
en attente). Remplacé par `torch.from_file(shared=True, ...)` (mmap standard,
pages récupérables par l'OS, jamais résidentes en totalité) : le fichier
memmap EST désormais le cache lui-même (plus de `torch.save` intermédiaire
qui doublait la question), un sidecar JSON stocke le nombre réel de lignes
remplies. Débloque 200M (et au-delà) avec une demande `--mem` de quelques
dizaines de Go au lieu de near-total-node.

**Logging par step + validation loss, Pipeline 1** (`src/sae/sae_shared.py::
load_or_train_extended_sae`). Jusqu'ici : historique par ÉPOQUE calculé mais
jamais persisté séparément (seulement embarqué dans le state_dict du
checkpoint final, invisible sans le recharger), et aucune évaluation sur un
split tenu à l'écart du gradient pendant l'entraînement (seule une métrique
finale post-hoc, `compute_sae_metrics`, existait). Corrigé : historique PAR
STEP (aligné sur la convention déjà en place côté Pipeline 2,
`phrase_sae.py`), split de validation (5%, jusqu'à 8192 tokens, seed fixe)
évalué à chaque époque, écriture systématique d'un `*_history.json`
standalone. **Rétrocompatibilité vérifiée** : tous les checkpoints
`p1_extended_sae.pt` déjà produits contiennent encore l'ancien historique par
époque dans leur state_dict — récupérable et traçable sans aucun rerun (cf.
§0.5 ci-dessous).

**Suppression du filtre FineWeb2 par mots-clés** (`UTILITY_COMPLAINT_KEYWORDS`,
`src/data/keywords.py` — supprimé, pas commenté, git conserve l'historique).
Sa précision qualitative mesurée restait faible (~15-20%, §23.1) pour un rôle
qui n'a jamais eu besoin de pertinence thématique : le filler de volume sert
uniquement à isoler un effet de VOLUME BRUT de tokens sur le SAE résiduel,
jamais ajouté à `train_texts`. Remplacé par `sample_fineweb2_chunks`
(`src/data/preparation.py`) : sous-échantillonnage sans filtre, mêmes
garde-fous de déduplication (URL + hash de chunk). Ne change aucune
conclusion déjà tirée (le filler n'a jamais influencé les métriques
d'interprétabilité, seulement le réservoir résiduel) — simplifie le code et
réduit la dépendance à un filtre dont la pertinence était déjà mise en doute.

**Outillage de diagnostic, nouveau** : `src/analysis/plotting.py` (5
fonctions Plotly réutilisables : courbes d'entraînement, métriques vs
hyperparamètre, distribution d'activation/rho_interp par groupe, heatmap de
corrélation, histogramme d'accord du juge) ; `scripts/generate_diagnostic_plots.py`
(agrégation RÉTROACTIVE, zéro rerun — scanne les 35 runs `results_*/`
existants : 39 courbes d'entraînement produites, dont TOUTES les runs
Pipeline 1 antérieures au fix ci-dessus grâce à l'historique par époque déjà
embarqué dans leurs checkpoints ; 25 distributions rho_interp ; 6 figures de
balayage consolidées — K_EXTRA, D_EXTRA, volume, layer, hook-point, échelle
du modèle — remplaçant les tables texte dispersées de ce journal) ; nouvel
onglet dashboard "Diagnostics d'entraînement" (`src/visualization/dashboard.py`)
lisant ces figures pré-générées, cohérent avec la philosophie du dashboard
(lecture d'artefacts disque uniquement) ; `docs/sae_diagnostics_playbook.md`
(checklist ordonnée : convergence → fidélité → capacité → interprétabilité →
significativité → indépendance du juge), référencé depuis `CLAUDE.md`.

**Non couvert rétroactivement** (nécessiterait un rerun) : heatmap de
corrélation NPMI (§24, aucune matrice jamais persistée sur disque) et
histogramme de sensibilité à l'ordre du juge (§13.1, seul le résumé agrégé a
été conservé) — documenté explicitement dans le docstring du script plutôt
que silencieusement omis.

`pytest tests/ -q` reste vert (41/41) après chacun de ces changements.

## 55. Phase 1 — le SAE core seul fait-il mieux que core+extension ? (test de "pollution")

Question posée directement : l'extension `FrozenCoreResidualSAE` entraînée
sur le résidu du core GemmaScope pourrait-elle polluer plutôt qu'enrichir le
signal exploité en aval (classification, structure de cluster) ? Protocole
CPU-only, zéro GPU/LLM, zéro rerun de pipeline : réutilise les activations
denses déjà en cache du run principal (`results_v10_emails_main/cache/
p1_all_doc_acts_ext_d1024.pt`, [44253, 17408] = d_core 16384 + D_EXTRA 1024),
reconstruit uniquement les labels de split (déterministe, `CORPUS_SPLIT_SEED=42`,
vérifié train=41176/test=2177/diff=900 = 44253, exact) pour ré-aligner les
lignes du tenseur, puis compare `acts[:, :16384]` (core seul) à `acts` (complet)
sur les mêmes métriques que le run principal — mêmes folds de validation
croisée pour les deux conditions, autorisant un test de McNemar apparié
(`scripts/core_vs_extension_ablation.py`, CPU-only, soumis en SLURM plutôt
que sur le nœud de login, ce dernier étant partagé et déjà proche de la
saturation mémoire, ~23/31 Go utilisés, swap quasi plein — un run tué
silencieusement dans ces conditions n'écrit ni traceback ni artefact).

Job compute-bound (régression logistique multinomiale sur matrice dense
~41176×17408, solveur lbfgs) plutôt que memory-bound (`MaxRSS≈11 Go` mesuré
pour `--mem=48G` demandé) : `--cpus-per-task=32` avec threads BLAS
explicitement fixés (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`)
plutôt qu'une simple augmentation de la mémoire demandée.

**Résultat** :

| Métrique | Core seul (16384 dims) | Core+extension (17408 dims) |
|---|---|---|
| `silhouette` (test, axes email) | 0,008643046952784 | 0,008643048815429 |
| `clf_acc_email_axes` (14 classes) | 93,67% | 93,59% |
| `clf_acc_sae` (energy/sports) | 60,0% | 60,0% |
| `dead_pct` | 56,0% | 52,8% |
| `L0` moyen | 1574,3 | 2446,7 |

McNemar apparié (mêmes folds, mêmes documents) : axes email b=346/c=316,
z-équivalent p=0,26 (non significatif) ; energy/sports b=0/c=0, p=1,0 —
**zéro désaccord de prédiction sur les 600 documents**, les deux conditions
classent chaque document identiquement.

**L'extension n'apporte ni gain ni dégradation mesurable sur ces sondes.**
La silhouette est identique à la 6ᵉ décimale près, la classification varie de
moins d'un point (dans les deux sens selon la sonde), et aucun des deux tests
de McNemar n'atteint la significativité. Pas de pollution détectable (aucune
métrique ne se dégrade en ajoutant l'extension), mais pas non plus de gain
linéairement décodable pour ces deux tâches précises. Cohérent avec le reste
du chapitre : la valeur ajoutée mesurée de l'extension dans ce projet est
spécifiquement dans l'interprétabilité individuelle des features
(odd-one-out, 45,3% sur les features d'extension) et dans la couverture de
concepts absents du core, pas dans un signal linéairement séparable
supplémentaire pour des sondes de classification déjà proches de saturation
sur le core seul (`clf_acc_email_axes` core-seul, 93,67%, dépasse même de
peu le run principal cœur+extension, 93,5%).

## 56. Réplication seed multiple de l'ablation volume 25M (§23.4)

**Question** : le §23.4 mesurait un écart numérique de +8,7 points sur un
seul seed (54,0% vs 45,3% référence), non significatif seul mais notée comme
piste à répliquer, coïncidant en direction avec l'ablation `K_EXTRA=5`
(§25/§45). Deux seeds supplémentaires (7, 99) tranchent-ils entre effet réel
de petite taille et coïncidence entre les deux ablations ?

**Écart à la configuration de référence** : identique à `results_v13_ablation_volume25m`
(§23.4) — `N_TOKENS_EXTRA_TRAIN=25 000 000`, filler FineWeb2-fr sur le
réservoir résiduel uniquement, `PIPELINES=p1` — seul `SEED` change (7 et 99
au lieu de 42).

**Méthode statistique** : test z sur deux proportions (`src/analysis/stats.py`),
chaque seed comparé individuellement au run principal (68/150), puis les 3
seeds (42 original + 7 + 99) combinés.

**n** : 150 features jugées par seed (450 au total, 3 seeds).

**Résultat** (jobs 42949 seed=7, 42950 seed=99, `results_v13_ablation_volume25m_seed7`/`_seed99`) :

| Seed | Interprétabilité | Écart au run principal (45,3%) | z | p |
|---|---|---|---|---|
| 42 (original, §23.4) | 54,0% (81/150) | +8,7 pts | −1,50 | 0,133 |
| 7 | 46,0% (69/150) | +0,7 pt | 0,12 | 0,908 |
| 99 | 44,7% (67/150) | −0,6 pt | −0,12 | 0,908 |
| **3 seeds combinés** | **48,2% (217/450)** | +2,9 pts | 0,61 | 0,540 |

`clf_acc_email_axes` reste stable sur les deux nouveaux seeds (91,2% et
91,0%, cohérent avec 91,3% pour le seed original).

**Conclusion** : les deux seeds supplémentaires ne reproduisent pas l'écart
du seed original — ils se situent quasiment sur la référence (46,0% et
44,7% contre 45,3%), loin du 54,0% initial. Le pool des 3 seeds combinés
(48,2%, z=0,61) est nettement plus proche de la référence que ne le
suggérait le seed unique. **La piste "volume 25M améliore l'interprétabilité"
ne réplique pas** : le résultat du seed 42 était vraisemblablement du bruit
d'échantillonnage plutôt qu'un effet de volume, contrairement à ce que la
coïncidence directionnelle avec `K_EXTRA=5` (§25/§45) laissait supposer.

**Limite connue** : la coïncidence de direction avec `K_EXTRA=5` était notée
au §23.4 comme un indice faible, pas une preuve combinée — cette réplication
ne teste que le levier volume, pas `K_EXTRA=5` lui-même (déjà répliqué
séparément sur 3 seeds au §45, résultat qui lui reste cohérent sur les 3
seeds, contrairement à celui-ci).

## 57. Audit méthodologique 2026-08 : bug `INTENT_KEYWORDS_FR` sans impact mesuré, fuite de groupe `clf_acc_email_axes` d'ampleur faible, dictionnaire résiduel quasi-orthogonal

**Question** : trois hypothèses indépendantes issues d'un audit méthodologique externe
(`docs/AUDIT_2026-08.md`) — (a) le pattern regex `\b(radical)\b` de
`INTENT_KEYWORDS_FR` (`src/data/dataset.py`) rate les formes fléchies, le taux
`intent_urgency_probe.py` (§13.2) en est-il affecté ? (b) `clf_acc_email_axes`
(`downstream_classification`, `StratifiedKFold` non group-aware) surestime-t-il la
performance faute de group-awareness par mail source ? (c) le dictionnaire résiduel
entraîné (`W_dec_extra`, D_EXTRA=1024) a-t-il développé une structure de superposition,
ou est-il resté proche de son initialisation PCA orthonormale ?

**Écart à la configuration de référence** : aucun — recalcul rétroactif sur les
activations et checkpoints déjà en cache (`results_v10_emails_main/`), zéro rerun de
pipeline, zéro GPU. `scripts/audit_2026_08_palier1_batch.py`,
`slurm/validation/run_audit_palier1_batch.slurm` (job 43222, CPU-only, 32 cœurs —
`--cpus-per-task=16` insuffisant en premier essai, job 43216 tué après 34 min sans
progression mesurable, cf. `CLAUDE.md`).

**Méthode statistique** : (a) comparaison directe des `n_pos`/accuracy avant/après
correctif regex, mêmes activations, même sonde (`downstream_classification`) ; (b)
`GroupKFold` (groupes = mail source, reconstruit par réplication déterministe de
`build_email_train_test_corpus`) vs `StratifiedKFold` actuel, mêmes activations ; (c)
distribution des cosinus hors diagonale de `W_dec_extra` normalisé (523 776 paires),
comparée au cosinus moyen attendu pour des directions aléatoires en dimension 3840
(√(2/(π·d)) ≈ 0,013).

**n** : (a) 3300 mails originaux (train, hors variantes augmentées) ; (b) 41176 lignes
train (14 classes, 3300 groupes/mails source) ; (c) 1024 directions, 523 776 paires.

**Résultat (a)** — impact du correctif `\b(radical)\b`→`\b(radical\w*)\b` :

| Intention | n_pos original | n_pos corrigé | acc_SAE original | acc_SAE corrigé | Δ vs baseline (orig / corrigé) |
|---|---|---|---|---|---|
| `intent_reclamation` | 1819 | 1819 | 97,7% | 97,73% | +42,6 / +42,6 pts |
| `intent_remboursement` | 479 | 479 | 84,5% | 84,64% | −1,0 / −0,8 pt |
| `intent_information` | 599 | 599 | 87,8% | 87,79% | +6,0 / +5,9 pts |
| `intent_urgence` | 968 | 971 | 97,7% | 97,88% | +27,0 / +27,3 pts |
| `intent_resiliation` | 1 | 1 | — | — | ignoré (classe dégénérée), les deux versions |

**Résultat (b)** — `StratifiedKFold` (actuel) vs `GroupKFold` (parent-aware) :

| Protocole | acc moyenne (5 plis) | folds |
|---|---|---|
| StratifiedKFold (actuel) | 93,65% | 0,939 / 0,939 / 0,937 / 0,933 / 0,936 |
| GroupKFold (corrigé) | 93,28% | 0,931 / 0,933 / 0,932 / 0,937 / 0,930 |

Écart : +0,38 pt. `ConvergenceWarning` (lbfgs, max_iter=1000) sur au moins un des dix
ajustements — accuracies stables entre plis malgré l'avertissement (écart-type inter-plis
≈0,003).

**Résultat (c)** — cosinus hors diagonale de `W_dec_extra` (D_EXTRA=1024) :
cosinus max 0,984 (≈1 paire sur 523 776), cosinus moyen 0,0248, fraction >0,5 : 1,7×10⁻⁵
(≈9 paires), fraction >0,9 : 1,9×10⁻⁶ (≈1 paire).

**Conclusion** :
(a) Le bug regex est réel au niveau mot mais **n'a aucun effet mesurable sur le
résultat déjà publié** (+27,0/+42,6 points inchangés à 0,3 point près) — les mails
originaux du corpus sont assez longs et lexicalement redondants pour que le radical
non fléchi apparaisse ailleurs dans le même document dans la quasi-totalité des cas.
Correctif à appliquer (une ligne, aucune régression identifiée) mais sans conséquence
sur les chiffres déjà rédigés.
(b) La fuite de groupe dans `clf_acc_email_axes` est réelle et de sens attendu
(le protocole actuel surestime légèrement) mais **d'ampleur faible** (+0,4 point,
93,5%→93,3% recommandé) — pas la surestimation sévère qu'on associerait typiquement à
une fuite de groupe non corrigée. Distinct et sans lien avec la question, séparée et
toujours ouverte, du signal lexical/templating (`augmentation_lexical_leakage_audit.py`,
qui utilise lui-même `StratifiedKFold` non corrigé pour son baseline TF-IDF).
(c) Le dictionnaire résiduel entraîné est **quasi-orthogonal** (cosinus moyen à peine
2× la valeur attendue pour des directions aléatoires, une seule paire quasi-colinéaire
sur plus d'un demi-million) — cohérent avec l'hypothèse que le SAE d'extension
(D_EXTRA=1024 pour d_model=3840) est resté proche de son initialisation PCA plutôt que
de développer une structure de superposition riche, sans qu'on puisse en conclure que
la taille du dictionnaire est la cause (nécessiterait le balayage D_EXTRA complet,
non fait ici).

**Limite connue** : (a)/(b) recalculés sur `results_v10_emails_main` uniquement (le run
principal cité au chapitre 3), pas sur les runs d'ablation dérivés qui utilisent aussi
`clf_acc_email_axes`/les labels `intent_*`. (c) mesure la géométrie finale du
dictionnaire, pas sa dérive par rapport à l'init PCA (nécessiterait de conserver le
checkpoint d'initialisation, non fait dans l'entraînement actuel) ni un lien causal
avec le taux d'interprétabilité (répond à "le dictionnaire a-t-il développé une
structure riche", pas à "une structure riche rendrait-elle les features plus
interprétables"). Détail complet, preuve et code : `docs/AUDIT_2026-08.md`
(constats B.26, B.6, B.1.1).

## 58. Audit méthodologique 2026-08 (suite) : ΔCE intégrée (métrique standard SAEBench/SAE Boost), confond `input_scale` du sanity check Frozen Decoder corrigé

**Question** : (a) `ce_loss_increase` (`src/sae/compare/crosslingual.py`) implémente
déjà la métrique ΔCE standard du domaine (CE(patched) - CE(clean)) mais n'était jamais
appelée — quel est le ΔCE du core seul vs core+extension, dans un référentiel comparable
à GemmaScope/SAE Boost ? (b) `FrozenDecoderExtendedSAE` (sanity check Korznikov et al.,
§19, 29,3%) hérite `input_scale=1,0` non calibré, contrairement à `ExtendedSAE`
(45,3%, échelle calibrée ≈3993) — la comparaison publiée 45,3% vs 29,3% est-elle
faussée par cette différence d'échelle, et dans quel sens ?

**Écart à la configuration de référence** : (a) aucun — recalcul sur un échantillon de
contrôle (60 mails originaux), même checkpoint d'extension déjà entraîné
(`p1_frozen_core_d1024_k32.pt`). (b) retrain de l'encodeur seul (décodeur aléatoire
figé inchangé, même principe que `FrozenDecoderExtendedSAE`), sur le réservoir de
résidus déjà en cache (`p1_raw_residuals.pt`, 500k tokens, aucun forward LM
supplémentaire pour l'entraînement), avec `input_scale` calibré sur la médiane des
normes du résidu (même formule que `ExtendedSAE._init_from_residual_pca`, direction du
décodeur non touchée).

**Méthode** : (a) forward hook sur `model.model.language_model.layers[LAYER-1]` (aplati
`[batch,seq,d]→[n,d]` avant `encode`/`decode`, trois bugs distincts corrigés en série —
détail complet dans `docs/AUDIT_2026-08.md`, constat B.20/E.6), CE calculée par
`AutoModelForCausalLM` avec labels décalés. (b) `ScaleCalibratedFrozenDecoderSAE`
(sous-classe locale de `FrozenDecoderExtendedSAE`, script d'audit autonome, ne modifie
pas `frozen_core.py`), 10 epochs, `lr=3e-4`, rejugée avec le protocole odd-one-out
standard sur les mêmes 150 features que le run principal.

**n** : (a) 60 documents. (b) 150 features (mêmes que le run principal et que
`FrozenDecoderExtendedSAE` original).

**Résultat (a)** :

| Condition | CE clean | CE patchée | ΔCE |
|---|---|---|---|
| Core seul | 2,177 | 3,475 | 1,298 |
| Core + extension | 2,177 | 2,582 | 0,404 |

**Résultat (b)** :

| Configuration | `input_scale` | Décodeur | Taux interp. |
|---|---|---|---|
| `ExtendedSAE` (référence, §12) | calibré (≈3993) | appris | 45,3% (68/150) |
| `FrozenDecoderExtendedSAE` original (§19) | 1,0 (non calibré) | aléatoire figé | 29,3% |
| `FrozenDecoderExtendedSAE`, échelle calibrée (ce test) | calibré (≈3993) | aléatoire figé | **16,0% (24/150)** |

**Conclusion** :
(a) L'extension réduit le ΔCE de 1,298 à 0,404 (−69%) — première mesure de fidélité
fonctionnelle (pas seulement de reconstruction) de ce projet, dans un référentiel
directement comparable à la littérature. Cohérent avec et renforce la lecture déjà
tirée de la FVE (le résidu non capturé par le core seul reste fonctionnellement
significatif pour le LLM).
(b) Corriger le confond `input_scale` fait **chuter** le score du décodeur aléatoire
(29,3%→16,0%), pas monter comme l'hypothèse de départ le craignait — l'écart avec le
SAE entraîné se creuse (29,3 points) plutôt que de se resserrer une fois l'échelle
égalisée. Le sanity check Korznikov et al. en ressort **renforcé**, pas affaibli : le
chiffre 45,3% n'était pas favorisé par un confond de comparaison.

**Limite connue** : (a) n=60, une seule couche, pas de comparaison chiffrée directe aux
ΔCE publiés de GemmaScope/SAE Boost (échelles différentes). (b) mécanisme non élucidé
de la chute 29,3%→16,0% — en principe la calibration d'un scalaire d'échelle devrait
être absorbée par l'entraînement de l'encodeur (biais, seuil BatchTopK appris) sans
changer le score final ; qu'elle le fasse chuter suggère soit un effet d'entraînement à
budget d'epochs fixe (10 epochs, peut-être insuffisant dans le nouveau régime
d'échelle), soit un effet réel non trivial — non tranché dans cette passe, à
recreuser avant de citer l'écart exact (16,0% vs 29,3%) comme autre chose qu'indicatif.
Détail complet, preuve et code : `docs/AUDIT_2026-08.md` (constats B.20, A.3.3, E.6).

## 59. Audit méthodologique 2026-08 (suite) : biais de longueur du max-pooling très fort (ρ=0,906), McNemar odd-one-out/contrastif non interprétable

**Question** : (a) le pooling documentaire max sur les codes SAE (`doc_vec[f] =
max_t enc(x_t)[f]`) dépend-il mécaniquement de la longueur du document, comme le
prédit sa nature de statistique d'ordre ? (b) le gate odd-one-out et la labellisation
contrastive directe (`p1_contrastive_labels.json`, §15.4) sont-ils en désaccord
structuré sur les mêmes features, une fois testé formellement par McNemar ?

**Écart à la configuration de référence** : aucun — recalcul CPU-only sur les
activations et labels déjà en cache, zéro GPU.

**Méthode statistique** : (a) corrélation de Spearman entre longueur du document
(caractères) et deux quantités dérivées de `doc_vec` (nombre de features actives,
norme du vecteur). (b) McNemar apparié (`src/analysis/stats.py::paired_mcnemar_test`)
sur les features en commun entre les deux caches de labels.

**n** : (a) 41 176 documents train. (b) 150 features en commun.

**Résultat (a)** :

| Paire | ρ (Spearman) | p |
|---|---|---|
| Longueur ↔ nombre de features actives | **0,906** | ≈0 |
| Longueur ↔ norme du vecteur document | **0,755** | ≈0 |

**Résultat (b)** : `confident=True` pour 150/150 features du protocole contrastif
(`rate_contrastive_confident=1,0`, aucune exception) ; taux odd-one-out 45,3% (68/150)
sur les mêmes features. McNemar : statistique=80,0, p=3,7×10⁻¹⁹.

**Conclusion** :
(a) Corrélation très forte, confirmant sans ambiguïté l'hypothèse du biais de longueur.
Le nombre de features actives par document (moyenne 2454/17408) suit majoritairement
la longueur du texte, pas uniquement son contenu. Tout ce qui est calculé sur `doc_vec`
(`clf_acc_email_axes`, silhouette, diffing) hérite potentiellement de ce confond dans
une mesure non encore quantifiée précisément métrique par métrique.
(b) Le test McNemar est calculable mais **non interprétable en l'état** : une des deux
conditions comparées (le gate `confident` du protocole contrastif) n'a aucune variance
sur cet échantillon — le rejeter systématiquement à `True` rend toute comparaison
structurée avec l'odd-one-out vide de sens statistique, même si le p formel est
extrême. N'ajoute rien à ce que §15.4 avait déjà établi qualitativement (biais de
complaisance du juge contrastif).

**Limite connue** : (a) mesure la corrélation, pas l'effet causal sur une métrique
particulière une fois contrôlé — un pooling alternatif (moyenne, max normalisé) reste
à tester pour quantifier l'ampleur réelle de la contamination sur `clf_acc_email_axes`.
(b) tant que le gate `confident` du protocole contrastif reste dégénéré, aucune
comparaison formelle à l'odd-one-out ne sera informative — corriger le gate
(seuil de cohérence inter-labels, cf. §15.4) est un préalable, pas cette comparaison
elle-même. Détail complet, preuve et code : `docs/AUDIT_2026-08.md` (constats B.9,
A.2/F.2).

## 60. CORRECTIF MAJEUR — le premier correctif de B.26 (§57) était lui-même cassé ; une fois réellement corrigé, l'impact est massif, pas négligeable

**§57 est caduc sur ce point précis et ne doit plus être cité comme concluant sur
l'ampleur de B.26.** Le correctif appliqué dans `audit_2026_08_palier1_batch.py`
(`v[:-3] + r"\w*)\b"`) n'ajoutait `\w*` qu'à la **toute dernière** alternative de
chaque groupe `(a|b|c)`, jamais aux précédentes — un artefact de découpage de chaîne,
pas une erreur de conception. Les radicaux réellement cassés identifiés en §57
(`contest`, `r[ée]clamation`, `r[ée]sili`, `clôtur`, `rembours`, `renseign`,
`imm[ée]diat`) restaient donc non corrigés dans le "correctif", ce qui explique les
flips quasi nuls (0-3 documents) observés — ils ne mesuraient l'effet que sur la
dernière alternative de chaque groupe, presque toujours déjà un mot complet ou une
expression peu affectée.

**Question** : quel est l'impact réel une fois `\w*` appliqué à CHAQUE alternative
concernée (pas seulement la dernière) ?

**Méthode** : patterns réécrits à la main, alternative par alternative (pas de
transformation générique — leçon tirée de l'échec du premier correctif) :
`\b(radical\w*|...)\b` pour chaque radical à un seul mot, alternatives-phrases (avec
espaces) laissées inchangées. Vérification préalable par échantillonnage manuel : 40
occurrences de « résili » tirées au hasard avec contexte, lues directement — **40/40
sont des mentions authentiques d'intention de résiliation**, aucun faux positif de
bruit détecté (répond à l'hypothèse concurrente posée par le round d'audit qui a
signalé ce problème : le sous-comptage d'origine était bien aussi sévère que redouté,
pas un radical qui capterait du bruit).

**n** : 3480 mails originaux (impact corpus), 3300 mails train (rejugement).

**Résultat — impact corpus (documents concernés)** :

| Intention | n_pos original (buggé) | n_pos corrigé (motif réel) | Facteur |
|---|---|---|---|
| Réclamation | 1905 | 1906 | ×1,00 |
| **Résiliation** | **1** | **864** | **×864** |
| Remboursement | 499 | 796 | ×1,60 |
| Information | 627 | 2111 | ×3,37 |
| Urgence | 1022 | 1177 | ×1,15 |

**Résultat — sonde de classification (n=3300, mêmes activations en cache)** :

| Intention | n_pos (corrigé) | Précision sonde | Baseline | Écart | vs référence buguée (z, p) |
|---|---|---|---|---|---|
| Réclamation | 1820/3300 | 97,6% | 55,2% | +42,4 pts | z=0,02, p=0,98 (ns — inchangé) |
| **Résiliation** | **813/3300** | **97,8%** | **75,4%** | **+22,4 pts** | *(n/a — précédemment exclue, n_pos=1)* |
| Remboursement | 761/3300 | 80,4% | 76,9% | **+3,5 pts** | z=8,89, p=6,3×10⁻¹⁹ (n_pos change massivement) |
| Information | 2005/3300 | 90,2% | 60,8% | **+29,4 pts** | z=35,4, p≈0 (changement massif) |
| Urgence | 1117/3300 | 95,2% | 66,2% | +29,1 pts | z=3,95, p=8×10⁻⁵ (significatif mais modéré) |

**Conclusion, sans atténuation** :
1. **Réclamation reste quasi inchangée** (+42,4 vs +42,6 pts) — cohérent avec le fait
   que ses radicaux dominants (« inadmissible », « scandaleux ») étaient déjà des mots
   complets échappant au bug d'origine.
2. **Résiliation passe d'exclue (n_pos=1, "classe dégénérée") à un résultat fort et
   testable (+22,4 pts)** — un résultat entièrement absent du rapport jusqu'ici,
   maintenant disponible.
3. **Remboursement change de signe** : de −1,0 pt (le SAE ne bat pas la baseline,
   lu comme un échec ou une limite du label) à +3,5 pts (signal réel, modeste mais
   positif). La lecture "remboursement n'est pas capté par le SAE" du rapport actuel
   n'est plus soutenable telle quelle.
4. **Information change radicalement** : de +6,0 pts (résultat faible) à **+29,4 pts**
   (un des résultats les plus forts du tableau) — l'intention la plus sous-évaluée par
   le bug d'origine.
5. **Urgence** reste dans le même ordre de grandeur (+29,1 vs +27,0 pts) mais avec un
   changement de composition significatif (z=3,95) — la prévalence corrigée (33,8% vs
   29,3%) reste dans la plage où la conclusion qualitative ne change pas.

**Le tableau §13.2 (et sa reprise en `03_experiences_et_resultats.md` §5.1, présenté
dans le rapport comme la preuve la plus fiable des objectifs urgence/intention) doit
être intégralement remplacé par les chiffres ci-dessus, avec une nouvelle ligne
Résiliation.** Mis à jour dans le rapport (`03_experiences_et_resultats.md` §5.1) suite
à ce constat.

**Ce que ce correctif ne règle pas encore** : `src/data/dataset.py::INTENT_KEYWORDS_FR`
en production contient toujours le pattern buggé d'origine — seul le script d'audit
autonome (`audit_2026_08_b26_round2_fix.py`) utilise les patterns corrigés, par
monkey-patch, sans toucher le fichier de production. Appliquer le correctif en
production reste **en attente de confirmation explicite** avant modification (fichier
partagé, `load_mails_tsv` consommé par de nombreux scripts et par le cache
`intent_urgency_probe_results.json` déjà publié).

**Limite connue** : vérification manuelle faite uniquement pour « résili » (le plus
gros facteur de changement, ×864) — les trois autres intentions dont `n_pos` change
significativement (remboursement ×1,60, information ×3,37) n'ont pas reçu le même
contrôle qualité manuel, à faire avant de publier ces chiffres sans réserve. Détail
complet, preuve et code : `docs/AUDIT_2026-08.md` (B.26, section réouverte).

## 61. Audit méthodologique 2026-08, round 3 : baseline décodeur entraîné/init aléatoire, resserrement statistique de B.20/B.21

**Question** : (a) A.1 -- une fois le décodeur ENTRAÎNÉ (pas figé) mais parti d'une
init aléatoire (pas PCA), l'écart 29,3%→45,3% se referme-t-il ? Isole l'effet de
l'entraînement du décodeur de celui de l'initialisation. (b) B.20/B.21 -- les résultats
ΔCE et bf16/fp32 résistent-ils à un contrôle de significativité formel et à un
échantillonnage ciblant spécifiquement leur population critique ?

**Écart à la configuration de référence** : (a) `ExtendedSAE` standard mais
`domain_residuals=None` (décodeur reste à l'init aléatoire de
`FrozenCoreResidualSAE.__init__`, `input_scale` reste 1,0), entraînement complet
(décodeur + encodeur), même réservoir de résidus, même protocole de jugement que le
run principal. (b) mêmes conditions que §58, avec `return_per_doc=True` sur
`ce_loss_increase` et un échantillon élargi (150 vs 40 documents) incluant
explicitement le 1er token de contenu pour le diagnostic bf16/fp32.

**Méthode** : (a) entraînement + rejugement complet sur les mêmes 150 features que le
run principal. (b) test de Wilcoxon signed-rank apparié (ΔCE par document) + sanity
check indépendant (SAE identité, doit donner ΔCE=0) ; intervalle de confiance bootstrap
(2000 tirages) sur les moyennes par strate de norme.

**n** : (a) 150 features. (b) 60 documents appariés (ΔCE), 54 982 tokens (bf16/fp32).

**Résultat (a)** :

| Configuration | Décodeur | Init | `input_scale` | Taux interp. |
|---|---|---|---|---|
| `ExtendedSAE` (référence) | entraîné | PCA | calibré | 45,3% (68/150) |
| `FrozenDecoderExtendedSAE` (§19) | figé | aléatoire | 1,0 | 29,3% |
| `FrozenDecoderExtendedSAE`, scale calibrée (§58) | figé | aléatoire | calibré | 16,0% |
| **Décodeur entraîné, init aléatoire (ce test)** | **entraîné** | **aléatoire** | **1,0** | **30,7% (46/150)** |

**Résultat (b1) — ΔCE, test apparié** : Wilcoxon signed-rank (H1 : ΔCE core seul > ΔCE
core+extension), statistique=1830, **p=8,15×10⁻¹²** ; extension meilleure sur 60/60
documents. Sanity check SAE identité : ΔCE=0,000000 exactement.

**Résultat (b2) — bf16/fp32, échantillon élargi + 1er token inclus** :

| Strate | n | \|Δrésidu\|/\|résidu\| | IC95% bootstrap |
|---|---|---|---|
| < 2σ | 54 710 | 6,86% | [6,80%, 6,91%] |
| 2-4σ | 271 | 7,00% | [6,45%, 7,55%] |
| > 4σ | 1 | 18,88% | *(n=1, non concluant)* |

**Conclusion** :
(a) Entraîner le décodeur à partir d'un point aléatoire ne récupère que **1,4 point**
sur la baseline figée (29,3%→30,7%), très loin des 45,3% obtenus avec l'init PCA —
**l'essentiel de l'écart vient de l'initialisation, pas de l'entraînement du décodeur
en tant que tel**. Reconstruction nettement meilleure (`val_loss` 0,273 vs 1,51) sans
gain d'interprétabilité correspondant : la qualité de reconstruction ne prédit pas le
taux d'interprétabilité, cohérent avec Korznikov et al.
(b1) Le résultat ΔCE (§58) est confirmé avec une preuve statistique forte et un effet
unanime (60/60 documents), pas seulement un effet moyen. Le mécanisme de mesure
lui-même est validé indépendamment (sanity check exact).
(b2) La strate >4σ n'est plus vide (contrairement à §58, où `skip_first_content_token`
l'excluait par construction) mais reste extrêmement rare (1/54982 tokens, 0,0018%)
même avec un échantillonnage délibérément ciblé sur le site le plus à risque.
L'unique observation (18,9% d'erreur relative, contre ~7% pour les strates normales)
est cohérente avec l'hypothèse d'amplification à mesure que la norme croît, mais n=1
ne permet aucune conclusion statistique.

**Limite connue** : (a) `input_scale=1,0` non calibrée dans ce test (pour rester
comparable au sanity check original) -- la case "décodeur entraîné + init aléatoire +
échelle calibrée" reste non testée. (b2) capturer davantage d'observations >4σ
demanderait un échantillonnage encore plus ciblé (documents pré-identifiés comme
contenant des activations massives connues), pas seulement un tirage aléatoire élargi.
Détail complet, preuve et code : `docs/AUDIT_2026-08.md` (A.1, B.20, B.21).

## 62. Audit méthodologique 2026-08, round 2 (suite) : variance inter-graine de C2 mesurée directement (B.17)

**Question** : `c2_original_only_rejudge.py` (§52, "C2 RÉSOLU", 44,7% vs 45,3%) ne
posait jamais de graine avant cet audit — l'écart avec la référence reflète-t-il le
protocole lui-même, ou en partie le bruit d'un négatif/ordre reconstruit
différemment à chaque exécution ? Round 2 demandait une mesure directe plutôt qu'une
estimation par analogie avec l'instabilité de réordonnancement déjà connue (§13.1).

**Écart à la configuration de référence** : seeding ajouté au script (`random.seed`,
`SEED` env var, sortie dépendante du seed pour ne pas écraser le fichier original),
aucun autre changement de protocole.

**Méthode** : rejeu complet à seed=7 et seed=99, mêmes 150 features, même juge, même
SAE d'extension déjà entraîné.

**n** : 150 features par run (3 runs seedés/non-seedés au total avec la référence).

**Résultat** :

| Run | n_interp/150 | Taux |
|---|---|---|
| Original (non seedé, job 42878) | 67 | 44,7% |
| Seed=7 | 73 | 48,7% |
| Seed=99 | 73 | 48,7% |
| Référence (run principal, exemples mixtes) | 68 | 45,3% |

**Conclusion** : les deux graines donnent le même total agrégé (73/150) mais des
fichiers de sortie distincts (vérifié par `diff`/`md5sum`, 1394 lignes de différence)
— coïncidence sur le total, pas sur les décisions individuelles. L'amplitude observée
sur les 4 runs (44,7-48,7%, 4 points) est cohérente avec le bruit d'échantillonnage
binomial attendu à n=150 (écart-type théorique ≈4,1 points) — pas besoin d'invoquer un
effet supplémentaire pour l'expliquer. La conclusion "C2 résolu" (§52) tient sur les
4 runs : aucun ne s'approche du niveau qui suggérerait un effondrement de l'effet.

**Limite connue** : `judge_model_separation_test.py` a reçu le même correctif de
seeding mais n'a pas encore été rejoué à plusieurs graines — la même vérification
reste à faire pour le résultat §43 (dépendance au juge, 45,3% vs 24,7%). Détail
complet : `docs/AUDIT_2026-08.md` (B.17).

## 63. Audit méthodologique 2026-08, round 2 (suite) : variance inter-graine de la dépendance au juge mesurée directement (B.17)

**Question** : `judge_model_separation_test.py` (§43, "dépendance au juge confirmée",
45,3% juge 12b vs 24,7% juge 4b, accord 56,7%) ne posait pas de graine avant cet
audit — écart identique au précédent §62 pour `c2_original_only_rejudge.py`, cette
fois pour le script qui compare deux modèles de juge distincts plutôt que deux runs
du même juge.

**Écart à la configuration de référence** : seeding ajouté au script (`random.seed`,
`SEED` env var, sortie dépendante du seed pour ne pas écraser le fichier original),
aucun autre changement de protocole.

**Méthode** : rejeu complet à seed=7, mêmes 150 features, mêmes deux juges
(gemma-3-12b-it, gemma-3-4b-it), même SAE d'extension déjà entraîné.

**n** : 150 features.

**Résultat** (job SLURM 43472, `p1_judge_model_separation_seed7.json`) :

| Run | `interp_rate_original` (12b) | `interp_rate_alternative` (4b) | Accord | Bascules 0→1 / 1→0 |
|---|---|---|---|---|
| Référence (non seedé, §43) | 45,3% | 24,7% | 56,7% (85/150) | — |
| Seed=7 | 45,3% | 24,7% | 55,3% (83/150) | 18 / 49 |

**Conclusion** : les deux taux marginaux par juge sont quasi identiques à la
référence (écart <0,1 point sur les deux), contrairement à §62 où le total agrégé
variait de plusieurs points — ce script ne re-tire pas le négatif à chaque rejeu, seul
l'ordre de présentation dépend du seed, et l'effet sur le total par juge est ici
négligeable. L'accord inter-juges varie un peu plus (56,7%→55,3%, -1,4 point) mais
reste du même ordre. **L'écart de ~21 points entre juges 12b et 4b (§43) est donc
robuste au choix de graine** — pas un artefact du tirage aléatoire non contrôlé à
l'origine.

**Limite connue** : une seule graine alternative testée (seed=7, pas de seed=99 comme
pour §62) — suffisant pour confirmer l'ordre de grandeur de l'écart entre juges, pas
pour caractériser précisément la distribution du taux d'accord lui-même. Détail
complet : `docs/AUDIT_2026-08.md` (B.17).

## 64. Audit méthodologique 2026-08 (suite) : Soft-Frozen Decoder (Korznikov et al.) implémenté, ne reproduit pas la quasi-parité rapportée avec le SAE entraîné (A.3.2)

**Question** : le Soft-Frozen Decoder (décodeur entraînable mais contraint à
cosinus≥0,8 de son init aléatoire) est la baseline la plus informative de Korznikov
et al. (0,88, quasi égale au SAE entraîné 0,90) mais n'avait jamais été implémentée
dans ce dépôt, qui ne compare qu'au Frozen Decoder pur (strictement figé). Le
reproduire ici confirme-t-il la quasi-parité rapportée par les auteurs ?

**Écart à la configuration de référence** : `SoftFrozenDecoderSAE` (sous-classe locale
de `FrozenCoreResidualSAE`, `scripts/audit_2026_08_soft_frozen_decoder.py`) — décodeur
entraîné puis reprojeté après chaque step sur le bord exact de la calotte sphérique à
cosinus=0,8 de son init aléatoire s'il en sort (projection exacte, pas une
interpolation linéaire) ; encodeur libre ; `input_scale` calibrée dès le départ (même
protocole que A.3.3, comparaison à échelle égale avec `ExtendedSAE`).

**Méthode** : entraînement complet (10 epochs, même config que le run principal),
puis jugement odd-one-out sur les mêmes 150 features de référence.

**n** : 150 features.

**Résultat** (job SLURM 43473) :

| Configuration | Décodeur | `input_scale` | Taux interp. |
|---|---|---|---|
| `ExtendedSAE` (référence) | libre, init PCA | calibré | 45,3% (68/150) |
| Décodeur entraîné, init aléatoire (§61) | libre | 1,0 | 30,7% (46/150) |
| Frozen Decoder pur (§19) | figé | 1,0 | 29,3% |
| Frozen Decoder pur, scale calibrée (§58) | figé | calibré | 16,0% |
| **Soft-Frozen (cos≥0,8), scale calibrée** | **contraint au cône** | **calibré** | **26,0% (39/150)** |

Contrainte vérifiée activement à l'exécution : cosinus final min=0,8000 (exactement à
la borne), moyenne=0,8037. `val_loss` final=0,3652, entre le Frozen Decoder pur
(1,51) et le décodeur totalement libre (0,273), comme attendu pour une contrainte
intermédiaire.

**Conclusion** : **désaccord net avec Korznikov et al.** Leur écart Soft-Frozen/SAE
entraîné est de 2 points (0,88 vs 0,90) ; ici il est de 19,3 points (26,0% vs 45,3%).
Le Soft-Frozen se comporte plus comme une variante dégradée du Frozen Decoder pur
que comme une approximation du SAE librement entraîné sur ce domaine. Comme pour A.1
(§61), la reconstruction intermédiaire (`val_loss`) ne se traduit pas par une
interprétabilité proportionnellement intermédiaire.

**Limite connue** : un seul seuil de cosinus testé (0,8, valeur de Korznikov et al.),
pas de balayage ; comparaison uniquement à échelle calibrée, pas de variante scale=1
pour vérifier si la calibration dégrade ce résultat comme elle l'a fait pour le Frozen
Decoder pur (A.3.3, 29,3%→16,0%). Détail complet : `docs/AUDIT_2026-08.md` (A.3.2).

## 65. Audit méthodologique 2026-08, round 2 (suite) : B.17 clos — 2e graine pour la dépendance au juge (§63)

**Question** : compléter §63 (une seule graine, seed=7) avec une deuxième graine pour
`judge_model_separation_test.py`, au même niveau de rigueur que C2 (§62, 2 graines).

**Écart à la configuration de référence** : identique à §63, `SEED=99` au lieu de 7.

**Méthode** : identique à §63.

**n** : 150 features.

**Résultat** (job SLURM 43735, `p1_judge_model_separation_seed99.json`) :

| Run | `interp_rate_original` (12b) | `interp_rate_alternative` (4b) | Accord |
|---|---|---|---|
| Référence (non seedé, §43) | 45,3% | 24,7% | 56,7% (85/150) |
| Seed=7 (§63) | 45,3% | 24,7% | 55,3% (83/150) |
| Seed=99 | 45,3% | 28,0% | 61,3% (92/150) |

**Nuance méthodologique** : `interp_rate_original` est chargé depuis un cache fixe
(`p1_judge_labels_extended.json`, `judge_model_separation_test.py:39,46,88`), pas
recalculé — identique à 45,3% sur les 3 runs par construction, pas parce que le juge
12b serait insensible au seed. Seul le juge 4b est réellement rejugé à chaque graine ;
la variance mesurée (24,7%→28,0%, amplitude 3,3 points sur le taux marginal ;
55,3%→61,3%, amplitude 6 points sur l'accord) porte donc sur ce rejeu seul.

**Conclusion** : amplitude cohérente avec le bruit d'échantillonnage binomial attendu
à n=150 (écart-type théorique ≈4,1 points), du même ordre que celle observée pour C2
(§62, amplitude 4 points sur 4 runs). L'écart entre juges 12b et 4b reste large sur les
3 runs (17 à 21 points), jamais proche de 0 — **la conclusion "dépendance au juge
confirmée" (§43) est robuste au choix de graine**. B.17 est maintenant vérifié au même
niveau pour les deux scripts qui en manquaient (C2 et dépendance-au-juge).

**Limite connue** : seules 2 graines testées pour ce script (comme pour C2) — une
caractérisation complète de la distribution demanderait davantage de graines, non fait
ici. Détail complet : `docs/AUDIT_2026-08.md` (B.17).

## 66. Audit méthodologique 2026-08 (suite) : témoin aléatoire pour le test de regroupement de features (B.27), qui complète §34

**Question** : §34 comparait la reproductibilité inter-seed au niveau feature-à-feature
(0,820) vs groupe-à-groupe Louvain (0,948), Mann-Whitney p=0,269 non significatif,
conclusion « hypothèse plausible mais non confirmée, test sous-dimensionné (n=3-5
groupes) ». L'audit round 2 (B.27) fait remarquer que moyenner des vecteurs avant de
comparer leur similarité cosinus augmente mécaniquement la similarité par réduction de
variance — un effet géométrique pur, indépendant de toute structure sémantique — et
qu'aucun témoin ne permet de l'écarter dans le test original. Ce témoin manquant
apporte-t-il un éclairage différent malgré la non-significativité déjà honnêtement
rapportée en §34 ?

**Écart à la configuration de référence** : mêmes labels interprétables (deux seeds,
`p1_top_extended_features.json`), mêmes embeddings bge-m3 (aucun recalcul) — ajoute un
troisième niveau, groupe-à-groupe ALÉATOIRE : 200 tirages de partitions aléatoires de
même distribution de tailles que les communautés Louvain réelles (`np.random.default_rng(42).permutation`).

**Méthode** : Mann-Whitney unilatéral (scipy) sur 3 comparaisons : groupe réel >
feature (réplique §34), groupe réel > groupe aléatoire, groupe aléatoire > feature.

**n** : 68 paires feature-à-feature, 3 groupes réels (seed 42, tailles [19,13,36]) × 5
groupes réels (seed 123, tailles [26,3,18,11,13]), 600 paires groupe-aléatoire (200
tirages × 3 groupes assignés par tirage, appariement hongrois).

**Résultat** (job SLURM 43736) :

| Niveau | Similarité moyenne | n |
|---|---|---|
| Feature-à-feature | 0,820 | 68 |
| Groupe-à-groupe réel (Louvain) | 0,948 | 3 |
| Groupe-à-groupe aléatoire (200 tirages) | 0,964 | 600 |

| Comparaison | U | p |
|---|---|---|
| Groupe réel > Feature | 124,0 | 0,269 (identique à §34, déterministe) |
| Groupe réel > Groupe aléatoire | 756,0 | 0,675 — non significatif |
| Groupe aléatoire > Feature | 24938,0 | **0,0013 — significatif** |

**Conclusion** : un regroupement **purement aléatoire**, sans structure sémantique,
bat déjà significativement le niveau feature-à-feature (p=0,0013) — confirme que
l'essentiel de l'écart observé en §34 est l'artefact de moyennage, pas un signal
sémantique. Le regroupement Louvain réel n'est **pas distinguable** du regroupement
aléatoire (p=0,675), malgré n=600 contre n=3 côté aléatoire (puissance nettement
supérieure). **§34 doit être requalifié** : pas seulement « non confirmé faute de
puissance » mais « l'effet mesurable, même avec bien plus de puissance via le témoin,
n'est pas distinguable de l'artefact géométrique pur ».

**Limite connue** : n=3/5 groupes réels reste la limite fondamentale — aucun nouveau
run avec plus de features labellisées n'a été fait ici, seul le témoin manquant a été
ajouté sur les données déjà en cache. Détail complet : `docs/AUDIT_2026-08.md` (B.27).

## 67. Audit méthodologique 2026-08 (suite) : Soft-Frozen Decoder à scale=1, qui complète §64 (A.3.2)

**Question** : §64 testait le Soft-Frozen Decoder (Korznikov et al.) uniquement à
échelle calibrée (26,0%, écart 19,3 points avec la référence 45,3%). La calibration de
l'échelle dégrade systématiquement les décodeurs contraints/figés dans ce dépôt (Frozen
Decoder pur : 29,3%→16,0%, A.3.3) — la variante scale=1 du Soft-Frozen referme-t-elle
davantage l'écart avec Korznikov et al. (0,88 vs 0,90, 2 points) ?

**Écart à la configuration de référence** : identique à §64
(`SoftFrozenDecoderSAE`, cos≥0,8, projection exacte sur calotte sphérique), sauf
`input_scale` qui reste à 1,0 (défaut de `FrozenCoreResidualSAE.__init__`, pas de
`calibrate_scale()`) — `scripts/audit_2026_08_soft_frozen_decoder_scale1.py`.

**Méthode** : entraînement complet (10 epochs) + jugement odd-one-out sur les mêmes 150
features de référence.

**n** : 150 features.

**Résultat** (job SLURM 43734) :

| Configuration | `input_scale` | Taux interp. |
|---|---|---|
| `ExtendedSAE` (référence) | calibré | 45,3% (68/150) |
| Décodeur entraîné, init aléatoire (§61) | 1,0 | 30,7% (46/150) |
| **Soft-Frozen (cos≥0,8), scale=1 (ce test)** | **1,0** | **30,7% (46/150)** |
| Frozen Decoder pur (§19) | 1,0 | 29,3% |
| Soft-Frozen (cos≥0,8), scale calibrée (§64) | calibré | 26,0% (39/150) |
| Frozen Decoder pur, scale calibrée (§58) | calibré | 16,0% |

Contrainte de cône vérifiée active : cosinus final min=0,8000, moyenne=0,8022.

**Vérification supplémentaire (pas seulement le taux agrégé)** : le taux à scale=1 est
strictement identique à celui du décodeur libre/init aléatoire (§61, 30,7%=30,7%,
46/150 dans les deux cas). Comparaison des ensembles de features jugées interprétables
entre les deux runs (`jq`+`comm`, aucun calcul) : **16/46 features en commun**, proche
du chevauchement attendu par pur hasard sous indépendance (46×46/150≈14,1 attendu).
L'identité du taux est donc une coïncidence numérique — les deux décodeurs convergent
vers des solutions largement différentes, pas la même solution.

**Conclusion** : à scale=1 comme à échelle calibrée, le Soft-Frozen Decoder ne referme
pas l'écart avec le SAE librement entraîné (écart 14,6 points à scale=1, 19,3 points à
échelle calibrée — vs 2 points chez Korznikov et al.). La calibration de l'échelle
dégrade le résultat pour le Soft-Frozen comme pour le Frozen Decoder pur (même sens
dans les deux cas), donc n'explique pas le désaccord avec Korznikov et al. Le
Soft-Frozen se comporte, à scale=1, plus comme une variante indépendante du décodeur
libre/init aléatoire que comme une approximation du SAE entraîné à init PCA — la
contrainte de cône (cos≥0,8) ne semble pas capturer, sur ce domaine, le bénéfice
rapporté par les auteurs.

**Limite connue** : un seul seuil de cosinus testé (0,8) ; pas de balayage de
`COS_THRESHOLD` pour situer où (ou si) une transition frozen→libre plus informative se
produirait. Détail complet : `docs/AUDIT_2026-08.md` (A.3.2).

## 68. Audit méthodologique 2026-08 (suite) : B.26 point 4 — propagation du correctif d'intention aux 3 derniers consommateurs

**Question** : les patterns `INTENT_KEYWORDS_FR` V2 (correctif réel, §60) ont déjà été
appliqués à `intent_urgency_probe.py`. Trois autres scripts cités par B.26 comme
consommateurs en aval (`explanation_fidelity_test.py`, `steering_fidelity_test.py`,
`latent_retrieval_precision_eval.py`, tous déjà cités dans le rapport,
`03_experiences_et_resultats.md` §5.2/5.3/5.5) restent à revalider.

**Écart à la configuration de référence** : monkey-patch identique à §60/§67
(`src.data.dataset.INTENT_KEYWORDS_FR` remplacé en mémoire par les patterns V2, fichier
de production non touché) — `scripts/audit_2026_08_b26_propagate_fidelity.py`, qui
appelle les trois `main()` originaux sans les modifier, sauvegarde l'original avant
écrasement, restaure après.

**Méthode** : identique aux trois scripts originaux (protocoles inchangés), seule la
source des colonnes `intent_*` change.

**n** : identique aux runs originaux (200 documents/intention pour les deux tests de
fidélité, 3480 mails pour le retrieval).

**Résultat** (job SLURM 43957) :

**(a) `explanation_fidelity_test.py` — conclusion robuste** :

| Intention | Chute top-10 (orig → V2) | Ratio top/random (orig → V2) |
|---|---|---|
| Réclamation | 0,576 → 0,535 | 576 225× → 3 421× |
| **Résiliation** | *(absente, n_pos insuffisant)* → 0,983 | *(n/a)* → **25 450×** |
| Remboursement | 0,9997 → 0,995 | 1 058× → 451× |
| Information | 0,9998 → 0,996 | 251× → 28 169× |
| Urgence | 0,612 → 0,883 | 42 837× → 52 641× |

Tous les ratios restent écrasants dans les deux régimes de labels (au minimum 451×) —
la conclusion "l'explication porte réellement la décision" ne dépend pas du bug B.26.

**(b) `steering_fidelity_test.py` — conclusion NON robuste, 2 intentions basculent** :

| Intention | Ratio roundtrip/inplace (orig → V2) |
|---|---|
| Réclamation | 1,74× → 1,87× |
| **Résiliation** | *(absente)* → 0,82× |
| Remboursement | 0,02× → 0,37× |
| **Information** | **0,00× (neutralisé) → 0,99× (préservé)** |
| **Urgence** | **0,90× (préservé) → 0,11× (neutralisé)** |

Information et urgence échangent leur catégorie qualitative ("neutralisé" ↔
"préservé") entre les deux régimes de labels. Seule la conclusion générale (effet
hétérogène, dépendant de l'intention, pas de mécanisme causal uniforme) tient dans les
deux cas — les affirmations spécifiques par intention du tableau original ne sont pas
fiables telles quelles.

**(c) `latent_retrieval_precision_eval.py` — conclusion robuste, taux de base
d'information change fortement** :

| Intention | Taux de base (orig → V2) | P@10 Latent Terms (orig → V2) | P@10 TF-IDF (orig → V2) |
|---|---|---|---|
| Réclamation | 54,7% → 54,8% | 1,00 → 1,00 | 1,00 → 1,00 |
| Remboursement | 14,3% → 22,9% | 1,00 → 1,00 | 0,00 → 0,30 |
| **Information** | **18,0% → 60,7%** | 1,00 → 1,00 | 0,20 → 0,70 |
| Urgence | 29,4% → 33,8% | 0,00 → 0,00 | 0,80 → 0,80 |

Le taux de base d'"information" quasi triple (18,0%→60,7%), cohérent avec le ×3,37 déjà
mesuré sur `n_pos` en §60. Le classement qualitatif (Latent Terms parfait sur 3/4
intentions, échec total sur urgence, TF-IDF compétitif seulement sur urgence) est
inchangé, mais P@10=1,00 pour Latent Terms sur "information" est moins remarquable à un
taux de base de 60,7% qu'à 18,0% — la lecture "généralisation sémantique" pour cette
intention spécifique est affaiblie, pas invalidée.

**Conclusion générale** : sur 3 scripts revalidés, 1 conclusion se renforce
(explanation_fidelity), 1 reste qualitativement inchangée mais avec un chiffre annexe à
nuancer (latent_retrieval), et 1 s'avère NON robuste avec inversion de catégorie pour 2
intentions sur 4 (steering_fidelity). B.26 illustre une seconde fois (après le §60
initial) qu'un résultat "vérifié" sur des labels faibles n'est pas automatiquement
fiable une fois le label corrigé — ici dans les deux sens (renforcement ET
affaiblissement selon le script). Rapport mis à jour :
le rapport de stage (résultats et perspectives).

**Limite connue** : `src/data/dataset.py::INTENT_KEYWORDS_FR` en production reste non
corrigé (décision utilisateur en attente) — ces trois résultats V2 restent, comme §60,
des rejeux par monkey-patch, pas le comportement par défaut du dépôt. Détail complet :
`docs/AUDIT_2026-08.md` (B.26).

## 69. B.26 — correctif appliqué en production, sanity checks de non-régression, découverte et correction d'une source de non-déterminisme distincte (LogisticRegression liblinear non seedée)

**Question** : sur autorisation explicite de l'utilisateur, `INTENT_KEYWORDS_FR` V2
(§60) remplace les patterns buggés directement dans `src/data/dataset.py`. Les scripts
consommateurs, non modifiés, reproduisent-ils exactement les chiffres déjà obtenus par
monkey-patch (§60, §68) une fois rejoués sans patch ?

**Écart à la configuration de référence** : aucun — c'est le point du test. Rejeu
direct de `intent_urgency_probe.py`, `explanation_fidelity_test.py`,
`steering_fidelity_test.py`, `latent_retrieval_precision_eval.py`, tous inchangés
(hors le fix `random_state` ci-dessous), contre le `dataset.py` désormais corrigé.
Originaux à labels buggés sauvegardés (`*.orig_bug_backup.json`) avant écrasement.

**Résultat (a) — `intent_urgency_probe.py` (job 44106) : match EXACT, à la décimale**,
sur les 5 intentions (n_pos, accuracy SAE, baseline, delta) contre §60. Le correctif de
production se comporte identiquement au monkey-patch qui l'a validé.

**Résultat (b) — `explanation_fidelity_test.py`/`steering_fidelity_test.py`
(jobs 44107/44108) : match QUALITATIF mais pas bit-exact contre §68.** Exemple
(`explanation_fidelity`, réclamation) : ratio top/random 50 558× (ce rerun) vs
28 169× (§68, monkey-patch) — même ordre de grandeur écrasant, chiffre exact différent.
**Cause identifiée** : les deux scripts appellent `LogisticRegression(solver="liblinear")`
sans fixer `random_state` — contrairement à `lbfgs` (déterministe, utilisé par
`downstream_classification`/`src/analysis/metrics.py` pour toute tâche multi-classe et
la plupart des tâches binaires de ce dépôt), `liblinear` a un chemin d'optimisation
non-déterministe sans seed. Le mécanisme concret ici : `explanation_fidelity_test.py`
sélectionne son échantillon de documents via un seuil dur (`probs > 0.7`) sur la sortie
`predict_proba` du classifieur — une différence numérique infime près de ce seuil peut
faire basculer un document dans ou hors de l'échantillon, cascadant vers un ensemble de
documents testés différent et donc un chiffre final différent, même si les coefficients
du classifieur convergent presque au même optimum (problème convexe, solution unique en
théorie, mais chemin d'optimisation différent en pratique sans seed).
**Corrigé** (`random_state=SEED` ajouté aux deux appels, une ligne chacun, même
principe que le seeding déjà appliqué pour B.17). **La conclusion qualitative de §68
est renforcée, pas affaiblie** : `steering_fidelity_test.py` rejoué (seed classifieur
différent de facto avant ce fix, labels identiques) reproduit la même catégorisation
par intention que le monkey-patch (réclamation amplifiée ~1,87×, résiliation
partiellement préservée ~0,64-0,82×, remboursement partiellement préservé ~0,28-0,37×,
information quasi totalement préservée ~0,97-0,99×, urgence fortement neutralisée
~0,11-0,13×) — deux runs indépendants avec labels corrigés mais seeds classifieur
différents convergent vers la même lecture qualitative, ce qui renforce la confiance
dans le changement de catégorie observé par rapport aux labels buggés (§68), pas
seulement un artefact d'un seul rerun.

**Résultat (c) — `latent_retrieval_precision_eval.py` (job 44109) : match quasi
parfait.** Taux de base identiques aux 4 décimales près (confirme le même jeu de
labels sous-jacent) ; Precision@10/@20 Latent Terms strictement identiques sur les 4
intentions. Seul écart : P@10/P@20 TF-IDF sur "information" (0,6/0,75 ici vs 0,7/0,7
en §68) — un seul document qui bascule dans/hors du top-10, sans lien apparent avec
`LogisticRegression` (ce script n'en utilise pas ; le baseline TF-IDF est un calcul
cosinus déterministe sur texte brut). Cause non identifiée dans cette passe (candidat
le plus probable : ordre non figé d'un tri à égalité stricte de score cosinus sur des
textes très templatés) — écart d'un seul document, ne change aucune conclusion.

**Conclusion** : le correctif de production se comporte comme attendu. Une source de
non-déterminisme non liée à B.26 (LogisticRegression liblinear non seedée) a été
découverte à l'occasion de cette vérification et corrigée dans les deux scripts
concernés — une leçon distincte de B.17 (qui portait sur `random`/`numpy` global, pas
sur le solveur scikit-learn) mais de même nature méthodologique.

**Limite connue** : la correction `random_state` n'a pas été vérifiée sur d'autres
appels `LogisticRegression` du dépôt au-delà des deux scripts concernés — les autres
sites (`src/analysis/metrics.py`, `core_vs_extension_ablation.py`) utilisent
`solver="lbfgs"` pour les tâches multi-classe (déterministe, pas concerné) et
`solver="liblinear"` uniquement pour les tâches binaires simples sans ré-échantillonnage
en aval dépendant d'un seuil dur — le mécanisme de cascade identifié ici ne s'y
applique probablement pas, mais ceci n'a pas été vérifié empiriquement par un rerun
répété. Détail complet : `docs/AUDIT_2026-08.md` (B.26).

## 70. Audit méthodologique 2026-08 (suite) : `embedding_model_comparison_test.py` — résultat déjà obtenu (job 40730, session antérieure) jamais écrit, comble un renvoi de commentaire sans preuve en production

**Question** : `select_latents_by_similarity`/le bloc "Corrélations intéressantes"
(`saev5.py:1240`) utilise bge-m3 plutôt que F2LLM, avec un commentaire renvoyant à
« la note ci-dessus » et affirmant que bge-m3 est « plus fiable que F2LLM sur des
labels courts » — sans qu'aucune section de ce fichier ne documente le test ayant
justifié ce choix. Recherche exhaustive (balayage systématique des scripts référencés
par `slurm/*/*.slurm` contre leurs mentions dans ce fichier) : `logs/analysis/embed_cmp_40730.log`
existe et contient un run complet et réussi de `embedding_model_comparison_test.py`,
jamais transcrit ici.

**Écart à la configuration de référence** : aucun — retranscription d'un résultat déjà
produit, aucun rerun nécessaire.

**Méthode** : `embedding_model_comparison_test.py` (diagnostic ponctuel, motivé par un
échec observé de F2LLM sur la requête "facturation résiliation panne" avec `select_latents_by_similarity`
en production) — embeddings bge-m3 des 13 685 labels de features disponibles
(Neuronpedia 16k + extension jugée), similarité cosinus contre 2 requêtes tests
("urgence réclamation client", "facturation résiliation panne"), top-15 affiché.

**n** : 2 requêtes, 13 685 labels comparés.

**Résultat (job 40730)** :

| Requête | Top match bge-m3 (sim) |
|---|---|
| "urgence réclamation client" | "urgent requests and invoices" (0,745), "[EXT] Réclamation Urgente" (0,743 ×2) |
| "facturation résiliation panne" | "[EXT] Facture contestée" (0,605 ×2), "[EXT] Résiliation contrat" (0,604) |

Les 15 premiers résultats des deux requêtes sont sémantiquement cohérents avec la
requête dans les deux cas (aucun résultat hors-sujet dans le top-15) — contrairement au
comportement dégradé de F2LLM sur la seconde requête qui avait motivé ce diagnostic
(docstring du script).

**Conclusion révisée — pas une comparaison à taille égale, confondue par la taille du
modèle.** Le script ne précise pas quelle variante F2LLM-v2 avait produit le mauvais
résultat qui motive ce diagnostic, mais `src/config.py::EMB_MODEL` par défaut est
`codefuse-ai/F2LLM-v2-80M` — probablement celle en cause. Taille sur disque comparée
directement (`du -sh`, pas de calcul, juste une lecture du système de fichiers) :

| Modèle | Taille sur disque | Paramètres (config) |
|---|---|---|
| bge-m3 | 4,3 Go | hidden=1024, 24 couches (XLM-R-large, ≈568M) |
| F2LLM-v2-80M | 166 Mo | ≈80M |
| F2LLM-v2-160M | 318 Mo | ≈160M |
| F2LLM-v2-330M | 653 Mo | ≈330M |

**bge-m3 est ≈26× plus gros que F2LLM-v2-80M sur disque**, et encore ≈6,6× plus gros
que F2LLM-v2-330M — la variante que le projet a par ailleurs jugée « assez grande »
pour servir de backbone Pipeline 2 (§16.4). Le résultat de job 40730 ne permet donc
pas de distinguer « bge-m3 est architecturalement mieux adapté au matching de labels
courts multilingues » de « un modèle ≈6-26× plus gros gagne, sans rapport avec le
choix d'architecture ». **Le commentaire de production (`saev5.py:1240`, "plus fiable
que F2LLM sur des labels courts") affirme une explication architecturale que ce test
ne peut pas trancher** — la conclusion initialement écrite ici (« empiriquement
justifié ») était trop forte, corrigée.

**Limite connue** : comparaison qualitative (inspection visuelle du top-15), pas de
métrique chiffrée (precision@k, ou score de cohérence formalisé comme dans B.29) ; F2LLM
lui-même n'a pas été rejoué en parallèle dans ce job pour une comparaison directe
chiffrée, seule la sortie bge-m3 est capturée ; **confond de taille de modèle non
contrôlé** (ci-dessus) — un test à refaire avec F2LLM-v2-330M (déjà disponible en
local, même échelle que la décision §16.4) donnerait une comparaison beaucoup plus
informative que F2LLM-v2-80M. Détail complet : `docs/AUDIT_2026-08.md`.

## 71. E.9 (suite) — comparaison à taille réduite (F2LLM-v2-330M au lieu de 80M) : le confond de taille explique une partie de l'écart, pas tout

**Question** : §70 identifie que job 40730 comparait bge-m3 (≈568M) à F2LLM-v2-80M
(≈80M, écart ≈26×) — un confond de taille non contrôlé. À taille bien plus proche
(F2LLM-v2-330M, ≈330M, écart réduit à ≈6,6×), l'avantage de bge-m3 observé en job
40730 se maintient-il ?

**Écart à la configuration de référence** : identique à job 40730 (mêmes 2 requêtes,
même jeu de 13 685 labels, même corpus de comparaison), F2LLM-v2-330M ajouté en
embedding parallèle (mean-pooling masqué, convention `phrase_sae.py::_mean_pool`, pas
le CLS pooling de bge-m3) — `scripts/audit_2026_08_e9_size_matched_embedding_compare.py`.

**Méthode** : identique à job 40730 (similarité cosinus, top-15 par requête et par
modèle), les deux modèles tournés dans le même job pour garantir un jeu de labels et
un encodage strictement identiques entre les deux conditions.

**n** : 2 requêtes, 13 685 labels comparés, 2 modèles.

**Résultat (job 44111)** :

- **Requête 1 ("urgence réclamation client") : F2LLM-v2-330M compétitif, voire
  légèrement meilleur.** Top-10 des deux modèles entièrement cohérent avec la requête
  (aucun résultat hors-sujet). Scores de similarité F2LLM plus élevés (0,765-0,736)
  que bge-m3 (0,745-0,696) sur ce top-10 — pas de dégradation visible en passant de
  568M à 330M.
- **Requête 2 ("facturation résiliation panne") : F2LLM-v2-330M dégrade nettement
  plus que bge-m3, malgré la taille rapprochée.** bge-m3 : 8/10 résultats cohérents
  (Facture contestée, Résiliation contrat, Réclamations Factures, Contestation
  Facture, Litige Facture), 2/10 vaguement adjacents ("recharge or cost calculation",
  "recoup back rent or clearance"). F2LLM-v2-330M : seulement 3/10 clairement
  cohérents (Facture contestée ×2, Facture énergie), le reste hors-sujet ou dégénéré
  ("danger catastrophe death rape fines", "disclaimermodelexplicitcontentrapediscriminationstorylanguagesignincapable"
  — un label manifestement corrompu/dégénéré du dictionnaire lui-même, indépendant du
  modèle d'embedding testé).

**Conclusion, sans atténuation** : **le confond de taille explique une partie de
l'écart observé en job 40730, pas la totalité.** Pour la requête simple (concepts
déjà proches lexicalement des labels), F2LLM-v2-330M rattrape bge-m3 — la taille
seule suffisait à expliquer l'écart initial sur cette requête. **Pour la requête
compositionnelle (3 concepts combinés, "facturation résiliation panne"), l'écart
bge-m3 > F2LLM persiste même à taille comparable** — le commentaire de production
(`saev5.py:1240`, "plus fiable que F2LLM sur des labels courts") tient donc
partiellement : pas comme affirmation générale non nuancée, mais spécifiquement pour
les requêtes multi-concepts, où la dégradation de F2LLM ne semble pas qu'un effet de
taille. Un label dégénéré est apparu dans le top-10 F2LLM
(`disclaimermodelexplicitcontentrapediscriminationstorylanguagesignincapable`) —
signale un problème de qualité du dictionnaire de labels lui-même (probablement un
résidu de génération de labels automatique mal filtré), indépendant du choix de
modèle d'embedding, qui mériterait sa propre vérification.

**Limite connue** : toujours pas de métrique chiffrée (precision@k contre une vérité
terrain), seulement une lecture qualitative du top-10/15 — comme pour job 40730. Le
label dégénéré repéré n'a pas été tracé à sa source. Détail complet :
`docs/AUDIT_2026-08.md` (E.9).

## 72. E.10 (nouveau) — le dictionnaire de labels Neuronpedia core (16k) contient 99 entrées dégénérées, dont 39 sont des transcriptions brutes de raisonnement LLM (jusqu'à 9340 caractères)

**Question** : le label dégénéré repéré en §71
(`disclaimermodelexplicitcontentrapediscriminationstorylanguagesignincapable`, feature
9676) est-il un cas isolé, ou un symptôme d'un problème plus large dans le fichier de
labels Neuronpedia core (`local_data/neuronpedia_labels/neuronpedia_labels_24-gemmascope-2-res-16k.json`,
consommé par `select_latents_by_similarity`, `find_interesting_pairs`, le dashboard,
et tout module citant des labels de features core) ?

**Écart à la configuration de référence** : aucun rerun — inspection pure lecture
(`jq`, aucun calcul) du fichier de labels déjà en cache.

**Méthode** : distribution de la longueur (en caractères) des 13 535 labels non-vides
du fichier, tri par tranche.

**n** : 13 535 labels (cohérent avec la couverture 16k déjà citée en B.16, 82,6%).

**Résultat** :

| Longueur | n | Exemple |
|---|---|---|
| médiane | — | 24 caractères |
| moyenne | — | 37,4 caractères (tirée vers le haut par les outliers ci-dessous) |
| max | — | 9 340 caractères |
| 80-200 caractères | 45 | phrases longues, borderline mais pas nécessairement dégénérées |
| 200-1000 caractères | 15 | clairement pas des labels concis |
| **>1000 caractères** | **39** | **transcriptions brutes du raisonnement de la LLM d'auto-interprétation Neuronpedia** (ex. feature dont le "label" commence par « Here's the explanation for the neuron's behavior:... » et enchaîne plusieurs paragraphes d'hypothèses successives, jamais réduits à une phrase finale) |

Le label court et dégénéré repéré en §71 (`disclaimermodelexplicitcontentrapediscriminationstorylanguagesignincapable`,
75 caractères) est d'une NATURE DIFFÉRENTE des 39 cas >1000 caractères : pas une
transcription de raisonnement, mais une chaîne de mots-clés visiblement issus d'un
refus de contenu de la LLM d'auto-interprétation, concaténés sans espaces — un second
mode de dégénérescence distinct dans le même pipeline de génération de labels.

**Conclusion** : le fichier de labels core (16k) contient au moins 2 modes de
dégénérescence distincts affectant ≥99/13535 entrées (0,73%) : (a) 39 cas où le
raisonnement complet de la LLM d'auto-interprétation Neuronpedia n'a jamais été réduit
à un label final court (le post-traitement censé extraire la phrase finale a
visiblement échoué), (b) au moins 1 cas où un refus de contenu a été concaténé en une
chaîne de mots-clés sans espaces. **Vérifié : aucune des 150 features de l'extension
jugées dans le run principal (`p1_judge_labels_extended.json`) n'est concernée** — ces
labels viennent d'un pipeline différent (juge local odd-one-out, pas Neuronpedia), donc
le chiffre phare 45,3% n'est pas affecté. L'impact réel touche les usages du
dictionnaire CORE : `find_interesting_pairs`/corrélations intéressantes (embeddings de
labels core), le module de comparaison B.24, le dashboard d'exploration des features
core, et potentiellement toute citation d'un label core spécifique dans le rapport si
elle tombe sur l'une de ces 99 entrées — non tracé exhaustivement dans cette passe.

**Limite connue** : la source du bug (extraction Neuronpedia elle-même, ou
`fetch_neuronpedia_labels`/le post-traitement de ce dépôt) n'a pas été identifiée —
n'a pas vérifié si le fichier source distant (avant téléchargement local) a le même
problème, ce qui trancherait entre un bug amont (Neuronpedia) et un bug local
(post-traitement de ce dépôt). Les 45 cas "longish" (80-200 caractères) n'ont pas été
inspectés individuellement — possible qu'une partie soit des labels légitimement
verbeux plutôt que dégénérés. Détail complet : `docs/AUDIT_2026-08.md` (E.10).

## 73. B.24/E.7 — premier essai réel du module de comparaison inter-modèles : verdict "comparable", mais le seuil par permutation censé être le plus rigoureux est dégénéré et n'était de toute façon jamais appliqué

**Question** : `src/sae/compare/` (alignement cross-modèle de features SAE,
détection de features non alignées/"polluées") est entièrement implémenté mais
n'avait jamais été exécuté sur un cas réel avant cette passe — le module
produit-il un résultat exploitable, et le seuil de significativité par
permutation qu'il calcule (`q_null_npmi95`) mais n'applique pas est-il
lui-même fiable ?

**Écart à la configuration de référence** : `--model-a` = F2LLM-v2-80M
(backbone Pipeline 2 en production), `--model-b` = bge-m3 — pas une
comparaison Pipeline 1 vs Pipeline 2. `pipeline.py --mode compare` entraîne
DEUX nouveaux `PhraseLevelSAE` from-scratch pour l'occasion, pas les
checkpoints existants ; pooling mean pour les deux modèles (`embed_corpus`),
alors que bge-m3 est utilisé en pooling CLS ailleurs dans ce dépôt.

**Méthode** : `src/sae/compare/pipeline.py --mode compare` (job 43958,
chemin de sortie corrigé au préalable, `results_v9` en dur → variable
d'environnement `COMPARE_PIPELINE_OUT`) ; inspection a posteriori des
parquets produits pour vérifier le seuil par permutation (job 44110, aucun
recalcul).

**n** : 2 SAE entraînés from-scratch sur 3480 mails, 207 features (modèle A,
30 communautés NPMI) et 317 features (modèle B, 57 communautés) survivant au
filtre de fréquence du graphe.

**Résultat** : entraînement sain des deux SAE (NMSE 0,021→0,013, dead
fraction quasi nulle en fin d'entraînement). `verdict="comparable"` (aucun
modèle >1,5× l'autre en masse d'activation portée par les features
flaggées), `mean_match_corr=0,177` (corrélation moyenne faible entre
features appariées par Hongrois), 0 feature flaggée comme polluée dans les
deux modèles (`n_flagged=0`, `model_score=0,000`). Le seuil réellement
utilisé pour flagger (mean+2×std du `pollution_score` z-scoré) donne 4,18
(A) et 4,27 (B) contre un score maximum observé de 3,96 (A)/3,76 (B) —
aucune feature ne franchit ce seuil, de peu. Le seuil par permutation calculé
mais jamais exploité (`q_null_npmi95`) vaut **1,0000 dans les deux
modèles** — la valeur maximale possible de NPMI, dégénéré, qui ne pourrait
jamais flagger quoi que ce soit s'il était réellement appliqué à la place du
seuil actuel.

**Conclusion** : le module fonctionne de bout en bout et produit un verdict
exploitable une fois le bug de chemin de sortie corrigé. Le seuil par
permutation prévu pour être le plus rigoureux ne l'est pas — il est
dégénéré sur ce type de graphe (probablement le même phénomène de
sous-puissance que §66/B.27 : peu de features actives survivent aux
permutations sans perdre leur structure de co-occurrence, produisant un null
artificiellement extrême). Conséquence pratique sur ce run précis : ça ne
change probablement pas le verdict "comparable", le seuil correctement
calculé aurait été encore plus permissif que celui effectivement utilisé,
pas plus strict.

**Limite connue** : le verdict "comparable" avec corrélation moyenne faible
(0,177) est compatible avec deux lectures opposées — les deux backbones
capturent des structures sémantiques réellement différentes sans qu'aucun
ne soit "pollué" relativement à l'autre, ou le détecteur manque de
puissance sur ce corpus (3480 mails, D_SAE=2048) pour distinguer les deux
cas. Un balayage avec un `--model-b` attendu comme nettement moins adapté au
domaine (anglais générique plutôt que français) permettrait de vérifier que
le détecteur peut au moins détecter une différence connue avant de faire
confiance à un verdict "comparable" sur une paire dont le comportement
attendu est inconnu. Le seuil `mean+2×std` réellement utilisé pour flagger
reste, lui, non corrigé (`src/sae/compare/model_compare.py`) : soit le
dériver de `q_null_npmi95` une fois ce dernier correctement calibré, soit
retirer la colonne vestigiale.

## 74. Le biais de longueur hypothétique de l'augmentation (troncature du prompt à 2048 tokens) n'est pas mesurable sur ce corpus

**Question** : `validate()` (`augmentation.py`) compare la variante générée
aux faits du mail parent COMPLET, mais le prompt d'augmentation est tronqué
à 2048 tokens — un mail parent plus long que ce seuil verrait sa génération
fondée sur un contexte partiel, potentiellement en délicatesse avec
`facts_lost` (comparaison contre le texte complet). Les mails longs
seraient-ils de fait sous-représentés parmi les variantes acceptées ?

**Écart à la configuration de référence** : aucun — mesure rétrospective sur
`local_data/emails/archive/augmented_mails_shard{0..7}of8_manifest.parquet`
+ `augmented_mails_test_manifest.parquet` (45 942 tentatives de génération
liées à 3474 mails parents après filtrage `load_and_clean_emails`), aucun
calcul GPU.

**Méthode statistique** : corrélation de Spearman entre longueur du mail
parent (caractères) et (a) rejet global, (b) `facts_lost` spécifiquement,
restreint aux axes non-`orthographe` (seuls concernés par ce garde-fou, cf.
§38-39) ; comptage des mails parents dépassant le seuil de troncature
approximatif (2048 tokens × ~4 caractères/token ≈ 8192 caractères).
`scripts/augmentation_rejection_length_bias_test.py` →
`local_data/augmentation_rejection_length_bias_results.json`.

**n** : 45 942 variantes (35 340 hors axe `orthographe`), 3474 mails parents.

**Résultat** : **1 seul mail parent sur 3474 dépasse le seuil de troncature
approximatif.** La corrélation longueur↔rejet est statistiquement
significative mais négligeable en amplitude (ρ=0,0117, p=0,012) — taux de
rejet quasi plat par quartile de longueur (11,3% à 12,6%, Q4 le plus long
pas nettement au-dessus des autres). La corrélation longueur↔`facts_lost`
restreinte aux axes concernés n'est même pas significative (ρ=0,0073,
p=0,17, n=35 340). Le sous-groupe des 10 variantes issues de l'unique mail
dépassant le seuil a un taux `facts_lost` de 20% contre 3,1% pour le reste —
à ne PAS lire comme confirmatoire (n=10, un seul mail parent, aucune
puissance statistique).

**Conclusion** : l'hypothèse B.9 (biais de longueur via troncature du
prompt) ne se vérifie pas sur ce corpus — mécaniquement, elle ne PEUT quasi
pas jouer : le corpus de mails réels ne contient qu'un seul mail assez long
pour être concerné par la troncature à 2048 tokens. Le biais de rejet
massivement documenté par axe (§38, jusqu'à 59,6%) est réel et important,
mais sa cause est le seuil `length_ratio` interagissant avec des axes qui
raccourcissent le texte par construction (orthographe très dégradée,
impatience) — indépendant de la longueur du parent. À corriger dans
`docs/archive/audits/AUDIT_SAE_2026-08.md` : l'item B.9 tel que formulé (troncature → perte de
faits → sous-représentation des mails longs) est réfuté par cette mesure ;
B.8 (taux d'acceptation par axe) était déjà couvert par §38, pas absent
comme l'audit l'affirmait.

**Limite connue** : l'approximation 4 caractères/token est grossière (pas de
tokenisation réelle du parent) — un mail juste sous le seuil réel pourrait
être classé à tort côté "ne dépasse pas". Avec 1 seul mail dépassant le
seuil approximatif sur 3474, une erreur d'approximation de quelques mails
autour du seuil ne changerait pas la conclusion (le mécanisme resterait
anecdotique, pas structurel). N'exclut pas un biais de longueur ailleurs
dans le pipeline (cf. §59, sans rapport avec l'augmentation) — seule
l'hypothèse spécifique de troncature du prompt d'augmentation est testée ici.

## 75. Un seuil dur sur le garde-fou B.5 (négatif odd-one-out) annulait `interp_score` pour la quasi-totalité des features de l'extension

**Question** : le correctif B.5 (négatif écarté si son activation réelle
dépasse `threshold_pos`) se comporte-t-il comme attendu sur un run réel, ou
seulement sur les tests unitaires synthétiques qui l'ont validé ?

**Écart à la configuration de référence** : run de validation 100k tokens,
layer 24, `N_FEATURES_TO_LABEL=20`, sinon identique à la configuration de
référence.

**Méthode statistique** : comparaison avant/après sur deux runs GPU
identiques par ailleurs (job 44778 avant les correctifs de cette session,
job 44831 après) ; ré-exécution directe de
`build_feature_examples_with_control` sur les activations mises en cache
des deux runs pour isoler la cause.

**n** : 20 features labellisées par run.

**Résultat** : `interp_score` moyen 45% (44778) → 0% (44831, 20/20 features
à `neg_example=None`). Cause isolée : le seuil dur de B.5
(`candidate_magnitude > threshold_pos: continue`) rejette la totalité des
20 candidats examinés pour chaque feature testée, sur les deux runs quand on
y rejoue le code corrigé — les features de l'extension sont sélectionnées
par magnitude (B.2), donc denses par construction, et leur 5e percentile
d'activation documentaire (~0,2-0,27 mesuré) est loin d'être nul : aucun
document du pool n'est un négatif "vrai" au sens strict. Le run 44778
montrait 45% parce qu'il utilisait encore l'ancien code, sans ce garde-fou.

**Conclusion** : seuil dur remplacé par un choix du candidat de plus faible
activation réelle parmi ceux examinés (`src/sae/judge.py`,
`build_feature_examples_with_control` et
`build_phrase_examples_with_control`) — `neg_example` n'est plus jamais
`None` si au moins un candidat existe, `neg_magnitude` reste traçable pour
un filtrage en aval si besoin. Revérifié sur les mêmes activations en cache
(44778, 44831) : `neg_example` non nul pour les features testées après ce
correctif.

**Limite connue** : ré-exécuté sur les activations en cache, pas sur un
nouveau run GPU de bout en bout (juge non ré-invoqué) — le taux
d'interprétabilité réel après ce correctif reste à mesurer sur un run
complet avant de comparer B.3/B.4/B.6/B.7/B.11 à la référence 45,3%.

## 76. Entraîner l'extension sur les mails originaux seuls, sans les variantes générées par Gemma-3, augmente fortement le taux d'interprétabilité mesuré

**Question** : le corpus d'entraînement de l'extension est dominé (~93% au
run de référence) par du texte généré par Gemma-3-12B-it, le même modèle qui
juge ensuite les features. B.1 (docs/archive/audits/AUDIT_SAE_2026-08.md) demandait un premier
test : réentraîner sur les mails originaux seuls, comparer le taux
d'interprétabilité.

**Écart à la configuration de référence** : run de validation 100k tokens,
layer 24, `N_FEATURES_TO_LABEL=20`, `LOCAL_AUGMENTED_MAILS_PATH` vide
(train/test = mails réels seuls, ~3480 mails au lieu de ~41k train/2,2k test
habituels). `N_VOLUME_FILLER_TARGET_CHUNKS=3000` inchangé — **confondu de
façon significative** : le filler FineWeb2 (jamais dans `train_texts`, mais
dans le réservoir résiduel qui entraîne les poids de l'extension) passe de
6,3% du corpus (3000/47253, run de référence) à **40,7%** (3000/7374, ce
run) puisque son volume absolu est fixe alors que `train_texts` a été divisé
par 12. Le SAE de ce run n'est donc pas "entraîné sur les originaux seuls" :
il est entraîné sur un réservoir ~6-7x plus dominé par du texte web
générique non filtré. Confondu aussi avec l'ensemble des correctifs
perf/data-science de cette session (B.3/B.4/B.5/B.6/B.7/B.11, cf. §75) : ce
run n'isole aucune variable seule — comparé ici au run de référence
`results_v21_validation_100k_layer24_v7_sharding` (job 44778), qui utilise
encore le code d'AVANT cette session (donc sans le bug puis le correctif de
§75) et une part de filler à 6,3%.

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`).

**n** : 20 features labellisées par run.

**Résultat** : interp_score 45% (9/20, IC95% [26%, 66%], corpus mixte, code
de référence) vs **85% (17/20, IC95% [64%, 95%], mails originaux seuls,
code de cette session)** — IC non chevauchants, z=-2,65, **p=0,008**, h de
Cohen=-0,88 (effet très large). `neg_example` non nul pour les 20/20
features (le correctif §75 fonctionne correctement sur ce run). Fidélité de
reconstruction dégradée sans les augmentés (attendu, moins de données
d'entraînement) : ρ_SAE 0,7904 (vs ~0,83), dead_pct 68,1% (vs ~57,8%) — la
sonde `acc_axes_email` n'a pas pu tourner (pas assez d'exemples par axe sans
augmentation).

**Conclusion** : effet large et significatif à cet effectif, mais **pas
attribuable à l'hypothèse B.1 telle que formulée** — la part de filler passant
de 6,3% à 40,7% (ci-dessus) est un mécanisme concurrent au moins aussi
plausible que "retirer le texte généré par le juge améliore
l'interprétabilité" : un SAE entraîné sur un réservoir dominé par du texte
web générique peut apprendre des features plus faciles à interpréter pour
des raisons sans rapport avec l'indépendance du juge. Ce run ne permet pas
de trancher entre les deux mécanismes. Deuxième résultat individuellement
significatif du dépôt après layer 31 vs layer 24 (§51), avec les mêmes
réserves : **non répliqué**, et ici en plus confondu par construction, pas
seulement par les correctifs de session (§75).

**Limite connue** : pour isoler réellement B.1, refaire ce run avec
`N_VOLUME_FILLER_TARGET_CHUNKS` réduit pour garder la même proportion de
filler (~6,3%) que la référence, ou à 0 pour un originaux-seuls strict —
condition nécessaire avant de citer ce résultat comme test de B.1. n=20 très
en dessous du n≥150 requis pour un taux agrégé
informatif (checklist diagnostics, `CLAUDE.md`) ; comparaison confondue par
tous les correctifs de cette session, pas une ablation à variable unique ; le
run originaux-seuls a moins de données d'entraînement pour l'extension
(rho_sae/dead_pct dégradés), un facteur de confusion supplémentaire non
isolé de l'effet corpus lui-même ; réplication à n≥150 sur seed différente
nécessaire avant de citer ce résultat comme établi.

## 77. Validation post-correctif B.5 (négatifs valides) sur corpus mixte, n=20

**Question** : le correctif B.5 (§75) répare la sélection de négatifs pour
le juge odd-one-out (auparavant `neg_example=None` sur tout le run après le
bug introduit cette session) — ce run vérifie qu'il produit bien des
jugements exploitables sur le corpus mixte de référence (pas seulement sur
le run de diagnostic ciblé de §75).

**Écart à la configuration de référence** : `N_FEATURES_TO_LABEL=20` (signal
rapide, pas n≥150), sinon config de référence (corpus mixte complet,
`N_TOKENS_EXTRA_TRAIN=100000` au lieu de 500000 — budget réduit pour un
tour rapide, cf. §12 pour la non-sensibilité déjà établie à ce paramètre).

**n** : 20.

**Résultat** : 8/20 = 40,0% interprétable (IC95% [22%, 61%]), **0/20
`dead_feature`/insuffisant** — le correctif fonctionne sur un run complet, pas
seulement le cas ciblé de §75.

**Conclusion** : correctif B.5 confirmé opérationnel en conditions réelles.
Sert aussi de bras de comparaison à n identique pour §78 (même
`N_FEATURES_TO_LABEL=20`, même `N_TOKENS_EXTRA_TRAIN=100000`, seule variable
: présence du corpus augmenté).

**Limite connue** : n=20, un seul seed.

## 78. B.1 corrigé : mails originaux + filler recalibré (~6,3%) vs corpus mixte, même échelle — écart énorme et significatif

**Question** : §76 a montré que le run "originaux seuls" précédent
confondait deux mécanismes (retrait du texte augmenté ET explosion de la
part de filler FineWeb2 de 6,3% à 40,7%) — ce run isole la variable unique
en fixant `N_VOLUME_FILLER_TARGET_CHUNKS=294` pour retrouver ~6,3% de filler
avec le corpus réduit (294/(3300+174+900+294)=6,3%, cf. commentaire
`run_validation_100k_layer24_v11_originals_filler_matched.slurm`).

**Écart à la configuration de référence** : identique à §77
(`N_FEATURES_TO_LABEL=20`, `N_TOKENS_EXTRA_TRAIN=100000`), sauf
`LOCAL_AUGMENTED_MAILS_PATH` vide et `N_VOLUME_FILLER_TARGET_CHUNKS=294` (au
lieu de la valeur par défaut, dimensionnée pour le corpus complet).

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`).

**n** : 20 features labellisées par run, comparaison à §77 (même n, même
code, même jour, seule variable isolée par construction).

**Résultat** : 8/20 = 40,0% (corpus mixte, §77) vs **19/20 = 95,0% (IC95%
[76%, 99%], mails originaux + filler recalibré)** — z=-3,71, **p=0,0002**, h
de Cohen=-1,32 (effet énorme). `dead_feature` : 0/20 dans les deux bras.

**Conclusion** : à la différence de §76, ce résultat isole correctement la
variable unique visée par B.1 (le retrait du texte généré par Gemma-3, à
part de filler constante) — l'effet est encore plus large qu'en §76 (95%
contre 85%), et cette fois sans le mécanisme confondant identifié. C'est,
à ce jour, l'écart individuel le plus large et le plus significatif de
tout le dépôt (devant layer 31 vs layer 24, §51, z=2,20). Réplication à
n=150 lancée (`run_validation_500k_layer24_v12_originals_filler_matched_n150.slurm`)
pour confirmer avant de citer ce résultat comme établi — cf. limite
ci-dessous.

**Limite connue** : n=20 est un signal, pas une réplication statistiquement
puissante (même réserve qu'en §76) ; un seul seed ; les indices de features
comparés entre les deux bras ne référencent PAS les mêmes features
sémantiques (deux SAE d'extension entraînés sur des corpus différents,
comparaison de taux agrégés uniquement, pas appariée) ; fidélité de
reconstruction non comparée ici (à vérifier que le SAE originaux-seuls
n'est pas simplement sous-entraîné, cf. §76 où rho_sae était dégradé).

## 79. B.2 tranché : sélection stratifiée par fréquence très significativement supérieure à la sélection par magnitude

**Question** : B.2 (docs/archive/audits/AUDIT_SAE_2026-08.md §5) — `feature_selection_by_magnitude`
sélectionne systématiquement les features les plus denses (proches de
directions génériques/stop-word), un échantillon non comparable à un chiffre
publié (Bills et al. échantillonnent au hasard, EleutherAI/Paulo stratifient).
`feature_selection_stratified_by_frequency` (App. J interp-embed, bins de
fréquence log-espacés) existait en opt-in depuis cette session sans run de
comparaison qui en valide l'effet.

**Écart à la configuration de référence** : aucun — même SAE d'extension déjà
entraîné (`results_v10_emails_main/`), même juge, mêmes caches d'activations.
Seule la fonction de sélection des 150 features à juger change
(`scripts/b2_stratified_selection_rejudge.py`).

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`).

**n** : 150 features par bras (deux échantillons disjoints à 16/150 près,
cf. limite).

**Résultat** : 68/150 = 45,3% (magnitude, référence) vs **134/150 = 89,3%
(stratifié par fréquence)** — z=-8,12, **p=4,5×10⁻¹⁶**, h de Cohen=-1,00
(effet énorme). Chevauchement des deux échantillons de 150 features :
16/150 seulement.

**Conclusion** : B.2 tranché sans ambiguïté — la sélection par magnitude
sous-estimait massivement le taux d'interprétabilité réel du dictionnaire.
`FEATURE_SELECTION_METHOD=stratified` passe par défaut (`src/sae/saev5.py`).
Toute comparaison future entre configurations (K_EXTRA, D_EXTRA, volume,
seed, layer) faite sous l'ancien défaut magnitude reste valide en tant que
comparaison **relative** (même biais des deux côtés), mais aucune ne peut
plus être lue comme une mesure absolue du taux d'interprétabilité du
dictionnaire — le chiffre absolu de référence (45,3%) est à traiter comme un
plancher, pas comme le taux réel.

**Limite connue** : les deux échantillons de 150 ne sont pas appariés
(seulement 16 features en commun) — l'écart mesure la différence entre deux
échantillons de features différentes, pas un effet causal sur les mêmes
features. Une fraction de l'écart pourrait aussi venir de features à
fréquence très faible (peu d'exemples positifs disponibles) étant
paradoxalement plus faciles à juger que des features dominantes/génériques
au comportement diffus — non décomposé ici par bin de fréquence.

## 80. Latent Terms (token-level, hors-domaine) : premiers résultats chiffrés — bat TF-IDF sur remboursement/urgence, perd sur réclamation

**Question** : §26 avait établi que Latent Terms (Clavié et al. 2026) devait
être réimplémenté token-level, hors-domaine (§3.1 du papier) — l'OOM de
`build_token_training_pool` bloquait tout résultat chiffré depuis (déjà
corrigé en réalité avant cette session, cf. correction de statut périmé,
commit qui précède celui-ci). Premier run complet à produire un JSON de
résultats.

**Écart à la configuration de référence** : n/a, protocole dédié
(`scripts/latent_retrieval_precision_eval.py`, docstring pour le détail) —
SAE token-level entraîné par reconstruction pure sur 33M tokens FineWeb2-fr
génériques (pool déjà en cache, job 44666), indexé sur les 3480 mails
originaux entiers.

**n** : 4 requêtes en paraphrase (une par intention), Precision@10/@20 contre
TF-IDF, corpus complet.

**Résultat** :

| Intention (taux de base) | P@10 Latent Terms | P@10 TF-IDF | P@20 Latent Terms | P@20 TF-IDF |
|---|---|---|---|---|
| réclamation (54,8%) | 0,50 | **1,00** | 0,55 | **1,00** |
| remboursement (11,1%) | **0,90** | 0,30 | **0,95** | 0,50 |
| information (60,7%) | 0,70 | 0,70 | 0,55 | 0,70 |
| urgence (33,8%) | 0,80 | 0,80 | 0,80 | 0,60 |

**Conclusion** : pas de victoire uniforme d'une méthode sur l'autre — Latent
Terms domine largement sur `remboursement` (taux de base faible, 11,1% :
TF-IDF proche du hasard, Latent Terms très au-dessus), à égalité sur
`information`/`urgence`, et nettement dominé sur `réclamation` où TF-IDF
atteint P@10=1,00 (taux de base 54,8%, le plus élevé des quatre — TF-IDF
profite mécaniquement d'un intitulé fréquent avec un vocabulaire régulier
type "réclamation"/"insatisfait"). Cohérent avec la réserve méthodologique
déjà documentée (le rapport de stage) : la vérité
terrain (`INTENT_KEYWORDS_FR`) est elle-même lexicale, ce qui favorise
structurellement TF-IDF sur les intentions à vocabulaire homogène.

**Limite connue** : n=4 requêtes, une seule par intention (pas de réplication
par paraphrase multiple) — évaluation indicative (docs/archive/audits/AUDIT_SAE_2026-08.md),
pas un benchmark IR (pas de MAP/nDCG/BEIR, cf. §1 métriques App. G
disponibles mais pas encore branchées sur ce protocole).

## 81. B.1 à pleine puissance (n=150) : le signal de §78 ne réplique pas sous sélection stratifiée

**Question** : réplication à n=150 de §78 (job 45081,
`run_validation_500k_layer24_v12_originals_filler_matched_n150.slurm`), qui
avait mesuré 95% (19/20) vs 40% (8/20) à n=20 sous l'ancien défaut
`FEATURE_SELECTION_METHOD=magnitude`.

**Écart à la configuration de référence** : entre-temps, B.2 (§79) a basculé
le défaut de sélection à `stratified` — ce run tourne donc sous stratifié,
pas sous le même réglage que §78. La comparaison directement appariée sur
la même méthode de sélection est contre §79 (mixte, stratifié, 134/150), pas
contre le chiffre magnitude de §78/référence historique (68/150).

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`).

**n** : 150 features par bras.

**Résultat** : 123/150 = 82,0% (originaux + filler recalibré, stratifié) vs
**134/150 = 89,3% (mixte, stratifié, §79)** — z=-1,81, **p=0,070, non
significatif**, et le sens de l'écart s'inverse par rapport au signal de
§78 (mixte nominalement *au-dessus* des originaux, pas en dessous). Contre
l'ancien plancher publié (68/150, mixte, magnitude) l'écart reste énorme
(z=-6,60, p=4×10⁻¹¹) mais ce dernier chiffre confond désormais deux
variables (corpus ET méthode de sélection), pas une comparaison valide de
B.1 isolément.

**Conclusion** : **le signal de §78 ne réplique pas à pleine puissance.**
Sous la méthode de sélection maintenant retenue comme la plus fiable
(stratifiée, §79), retirer le texte augmenté du corpus d'entraînement n'a
pas d'effet démontrable sur le taux d'interprétabilité — au mieux un effet
nul, au pire un léger effet inverse, dans tous les cas non significatif à
n=150. L'énorme écart de §78 (h de Cohen -1,32) était vraisemblablement du
bruit d'échantillonnage à n=20 amplifié par une interaction avec le biais
de sélection par magnitude alors en vigueur dans les deux bras — pas un
effet réel de l'indépendance du juge. **B.1 est donc résolu dans le sens
négatif** : comme C2 (§48/§50/§52), l'hypothèse de contamination du corpus
d'entraînement par le style du modèle juge n'est pas soutenue par les
données à l'échelle testée.

**Limite connue** : un seul seed pour le bras originaux+filler-matché ; pas
de run originaux-seuls sous l'ancien défaut magnitude à n=150 pour vérifier
si l'interaction sélection×corpus suspectée ci-dessus est réelle (piste de
suite si le sujet redevient pertinent, coût faible — réutiliser
`run_validation_500k_layer24_v12_originals_filler_matched_n150.slurm` avec
`FEATURE_SELECTION_METHOD=magnitude`) ; fidélité de reconstruction non
comparée entre bras dans ce run.

## 82. Sweep taille du modèle extracteur (4B/12B/27B, setup classique K_EXTRA=5) : effet significatif 4B→12B, plateau 12B→27B

**Question** : réplique et étend le sweep taille de modèle archivé
(`docs/archive/archived_runs_manifest.md`, pré-correctif juge) sous le code et le
protocole actuels — setup classique du papier SAE Boost (K_EXTRA=5,
D_EXTRA=1024), volume 25M tokens, sélection stratifiée (défaut depuis §79),
n=150, corpus mixte de référence. Chaque palier utilise son layer "~2/3
profondeur" propre (`src/config.py::_PRESETS`) : 4B→layer 17, 12B→layer 31,
27B→layer 40 (62 couches, curation GemmaScope-2 confirmée aux mêmes
proportions ~25/50/65/85% que 12B).

**Écart à la configuration de référence** : uniquement `MODEL_SIZE`/`LAYER`/
`SAE_ID`/`K_EXTRA` (5 au lieu de 32) — reste identique par ailleurs à la
référence (`results_v10_emails_main/`, mixte, 25M tokens).

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`).

**n** : 150 features par palier.

**Résultat** :

| Modèle | Layer | Interp. | IC95% |
|---|---|---|---|
| 4B | 17 | 72,0% (108/150) | — |
| 12B | 31 | 82,0% (123/150) | — |
| 27B | 40 | 83,3% (125/150) | — |

4B vs 12B : z=-2,06, **p=0,040** (significatif). 12B vs 27B : z=-0,31,
p=0,76 (non significatif). 4B vs 27B : z=-2,36, **p=0,018** (significatif).

**Conclusion** : confirme et affine l'effet déjà identifié comme le plus
robuste du dépôt (checklist diagnostics, `CLAUDE.md` : "seul le choix de
taille du modèle extracteur/juge produit un effet massif et répliqué à
chaque palier") — mais avec des **rendements décroissants nets** : le gain
4B→12B (+10 points, significatif) est bien plus large que 12B→27B
(+1,3 point, dans le bruit). Doubler la taille du modèle au-delà de 12B
n'apporte plus de gain mesurable à cette échelle de mesure (n=150). Sweep
1B lancé en parallèle (job 45365, résultat à ajouter en suite de section une
fois disponible).

**Limite connue** : un seul seed par palier ; les layers diffèrent entre
paliers (confondu avec la taille du modèle, pas isolé) — mais chaque layer
est déjà la meilleure estimation "~2/3 profondeur" disponible pour son
palier, pas un choix arbitraire qui introduirait un biais différent par
palier.

## 83. Séparation juge/extraction, famille différente : Qwen3.8-27B juge 74% de plus de features interprétables que gemma-3-12b-it (extracteur=juge)

**Question** : jusqu'à l'introduction de `JUDGE_MODEL_ID` (§7,
`docs/archive/audits/AUDIT_SAE_2026-08.md`), le juge d'auto-interprétation était systématiquement
le même checkpoint que celui dont on extrait les activations. §43/§63/§65
avaient déjà mesuré une forte dépendance au choix du juge (gemma-3-12b-it
45,3% vs gemma-3-4b-it 24,7%, écart robuste à 2 graines) mais ne pouvaient
pas isoler biais d'auto-préférence et simple effet de capacité, faute d'un
juge de famille différente et de capacité comparable en cache local
(`scripts/judge_model_separation_test.py`, docstring : "un juge de famille
différente serait nécessaire pour isoler (b) -- non fait ici"). Qwen3.8-27B
(bf16, famille différente, capacité comparable au palier 27b de Gemma-3)
comble ce manque.

**Écart à la configuration de référence** : uniquement le juge —
`scripts/judge_model_separation_test.py`, `ALT_JUDGE_MODEL_ID=Qwen3.8-27B`,
rejuge les 150 features déjà en cache (`p1_judge_labels_extended.json`,
jugées à l'origine par gemma-3-12b-it = extracteur), mêmes exemples, même
protocole odd-one-out. La variante FP8 de Qwen3.8-27B a été essayée en
premier (empreinte VRAM réduite) mais ne peut générer avec aucun de ses deux
chemins d'inférence sous torch==2.6.0 (pin du dépôt) — abandonnée, bf16
utilisé (52 Go). Un biais de génération distinct a aussi été trouvé et
corrigé avant ce résultat : Qwen3 active par défaut un préambule
`<think>...</think>` qui épuisait `max_new_tokens=8` (étape odd-one-out) sans
jamais atteindre la réponse (`interp_rate` mesuré à 0,0/150 avant correctif —
écarté, pas un résultat) ; `enable_thinking=False` ajouté à
`_batched_generate` (`src/sae/judge.py`) a résolu le problème.

**Méthode statistique** : `paired_mcnemar_test` (`src/analysis/stats.py`) sur
les 150 paires (même feature, deux juges) — le test apparié adapté ici,
contrairement à `two_proportion_test` (§82) qui suppose deux échantillons
indépendants alors que ce sont littéralement les mêmes 150 features rejugées.

**n** : 150 features.

**Résultat** (job 45619, h100,
`p1_judge_model_separation_Qwen3.8-27B_seed42.json`) :

| Juge | Taux interp. | IC95% (Wilson) |
|---|---|---|
| gemma-3-12b-it (= extracteur) | 45,3% (68/150) | [37,6% ; 53,3%] |
| Qwen3.8-27B (famille différente) | 78,7% (118/150) | [71,4% ; 84,5%] |

Accord 49,3% (74/150) ; 63 features basculent non-interprétable→interprétable
avec Qwen contre seulement 13 dans l'autre sens — asymétrie nette, McNemar
apparié sur les 76 paires discordantes : χ²=31,59, **p=1,90e-8**.

**Conclusion** : écart massif et hautement significatif, mais dans le sens
**opposé** à ce qu'un biais d'auto-préférence simple prédirait — si
gemma-3-12b-it gonflait son propre taux en se jugeant lui-même, on
s'attendrait à ce qu'un juge externe soit PLUS sévère, pas 74% plus généreux
(45,3%→78,7%). Ce résultat ne confirme donc pas l'hypothèse d'auto-préférence
testée ; il indique plutôt que gemma-3-12b-it est, sur ce protocole odd-one-out
précis, un juge nettement plus sévère (ou moins capable sur cette tâche
spécifique) que Qwen3.8-27B — cohérent avec §43 (où gemma-3-4b-it, plus
petit, jugeait aussi MOINS de features interprétables que gemma-3-12b-it :
la capacité du juge semble corréler positivement avec le taux mesuré, dans
les deux comparaisons). Les labels produits par Qwen ont été vérifiés
manuellement (lisibles, cohérents avec le domaine emails EDF) — l'écart n'est
pas un artefact de format.

**Limite connue** : une seule graine ; ne teste qu'UN juge de famille
différente (pas de deuxième famille pour vérifier que l'effet n'est pas
spécifique à Qwen) ; ne distingue pas "Qwen est un meilleur juge sur cette
tâche" de "Qwen est plus complaisant/moins strict sur le critère odd-one-out"
— seul un audit qualitatif plus poussé des faux positifs/négatifs de chaque
juge trancherait.

## 84. App K.1 (Diffing) — première mesure verification_rate/coverage du dépôt : 40% des hypothèses SAE valides, 92,5% de couverture

**Question** : sans `HypothesisVerifier` (App K.1), aucun chiffre de diffing du
dépôt n'était comparable au papier (§1, `docs/archive/audits/AUDIT_SAE_2026-08.md`) — la
"vérification" des hypothèses SAE se limitait à la génération d'une phrase
libre (`generate_llm_diff_hypothesis`), jamais quantifiée. `verification_rate`
(fraction des hypothèses dont la différence de fréquence vérifiée dépasse 1%,
Figure 11 du papier) et `coverage` (fraction des documents cible couverts par
au moins une hypothèse valide, Figure 12) implémentés dans
`src/analysis/hypothesis_verifier.py` et exercés ici pour la première fois.

**Écart à la configuration de référence** : prompt K.1 repris verbatim du PDF
(pas du code des auteurs, qui en diverge légèrement — cf. docstring du
module) ; hypothèses = labels des 10 features SAE les plus significatives
(triées par q, sens énergie>sports) de `p1_diff_energy_sports.csv` (déjà en
cache, `corpus_diff_stats`) plutôt qu'une génération LLM structurée (App D.2,
toujours "à faire", §1) ; corpus de vérification reconstruit frais
(`prepare_domain_dataset`, 40 documents energy + 40 sports, FineWeb2-fr) --
pas nécessairement les mêmes documents que ceux ayant produit les labels,
seulement le même domaine.

**Méthode statistique** : `proportion_with_ci` (`src/analysis/stats.py`,
Wilson) sur `verification_rate` (n=10 hypothèses) et `coverage` (n=40
documents énergie).

**n** : 10 hypothèses × 80 documents (40 énergie + 40 sports).

**Résultat** (job 45620, h100, `diffing_hypothesis_verification.json`) :

| Métrique | Valeur | IC95% (Wilson) |
|---|---|---|
| `verification_rate` (seuil 1%) | 40,0% (4/10) | [16,8% ; 68,7%] |
| `coverage` (documents énergie) | 92,5% (37/40) | [80,1% ; 97,4%] |

Détail des 10 hypothèses (`rate_in_group`=énergie, `rate_out_group`=sports) :
4 valides dans le sens énergie>sports ("conjunctions and introductory
phrases" 85,0% vs 72,5%, "rivers and geographies" 2,5% vs 0,0%, "spatial
prepositions" 65,0% vs 62,5%, "punctuation followed by conjunctions" 15,0%
vs 12,5%), 6 invalides (rate=0,0% dans les deux groupes -- features mortes ou
non pertinentes à ce corpus, features syntaxiques génériques comme "pulsing
with"/"shadows lengthen, fled, or deepen" incluses car App K.1 ne filtre pas
les labels syntaxiques, contrairement à l'App E.1 des corrélations, §85).

**Conclusion** : verification_rate=40% signifie que 6 des 10 hypothèses
SAE les mieux classées par NPMI/q-value du run archivé ne se répliquent PAS
sur un corpus energy/sports frais à seuil 1% -- cohérent avec un chiffre de
référence (le dépôt n'a pas encore de baseline LLM-only à comparer, App D.2,
pour situer ce taux relativement à l'alternative). Le coverage élevé (92,5%)
vient presque entièrement de 2 hypothèses à fort taux marginal
("conjunctions and introductory phrases" 85%, "spatial prepositions" 65%) --
pas d'une couverture homogène sur les 4 hypothèses valides.

**Limite connue** : n=10 hypothèses est petit (IC large, 16,8-68,7%) ; les
hypothèses proviennent d'un run archivé (juillet), pas régénérées sous le
code actuel ; pas de baseline LLM-only pour comparer verification_rate
(App D.2, LLM baseline de la Figure 11, toujours à construire) ; un seul
seed/tirage de corpus.

## 86. Clustering ciblé complet (App. F.1/§4.3) : mots-clés LLM + union top-k + Jaccard + accuracy + z-conductance, chaîne bout en bout pour la première fois

**Question** : `targeted_clustering_by_axis` (`saev5.py`) ne faisait que la
sélection de latents par UN SEUL `axis_query` fixe, sans étiquetage de
cluster, accuracy ni z-score de conductance (§1/§7,
`docs/archive/audits/AUDIT_SAE_2026-08.md`) — rien n'était comparable au papier pour le
clustering. `src/analysis/clustering_llm.py` (génération de mots-clés LLM +
union top-k, étiquetage de cluster, accuracy par réassignation LLM K.3,
z-conductance en espace dense) exercé ici en chaîne complète pour la
première fois, sur la requête "type de réclamation client".

**Écart à la configuration de référence** : latents = 68 features
d'extension déjà interprétables (`p1_top_extended_features.json`) plutôt que
le dictionnaire complet core+extension ; affinité de clustering = Jaccard
précalculée (corrigée au passage : le code par défaut utilisait
`SpectralClustering(affinity="cosine")` sur binaire, PAS Jaccard comme le
papier — deux métriques différentes sur un vecteur binaire, bug réel corrigé
dans `targeted_clustering_by_axis`, pas une divergence documentée) ;
embeddings denses = bge-m3 (pas les baselines dense/instruction-tuned du
papier, non reproduites ici) ; n_clusters=4 fixé (pas de sélection par
silhouette/gap statistic).

**Méthode statistique** : descriptive (accuracy et z-score par cluster,
`src/analysis/clustering_llm.py::compute_cluster_accuracy`/
`conductance_zscore`, null par 100 échantillons aléatoires de même taille).

**n** : 300 documents (train, échantillon aléatoire), 4 clusters (175/25/46/54
documents), 6 mots-clés générés, 61 latents sélectionnés (union).

**Résultat** (job 45622, h100, `clustering_llm_verification.json`) :

| Cluster | Taille | Description LLM | Accuracy | Z-conductance |
|---|---|---|---|---|
| 0 | 175 | Formal contract termination and billing dispute letters | 61,7% | -3,12 |
| 1 | 25 | New service activation requests and mixed service complaints | 40,0% | -2,91 |
| 2 | 46 | Urgent complaints regarding unexpected power outages | 56,5% | -4,32 |
| 3 | 54 | Mixed administrative requests for service activation and cancellation | 5,6% | -10,06 |

Mots-clés générés : "complaint about product defect", "billing dispute",
"service quality dissatisfaction", "delivery delay", "return and refund
request", "customer support responsiveness".

**Conclusion** : les 4 z-scores sont négatifs (clusters SAE plus "compacts"
en espace dense que des échantillons aléatoires de même taille, `conductance`
plus basse) — structure réelle, pas un artefact de hasard. L'accuracy varie
fortement entre clusters (5,6% à 61,7%) : cohérent qualitativement avec
l'observation du papier ("SAE clusters have comparable per-cluster
accuracies with embeddings, with generally higher variance across clusters",
§4.3/F.4) — le cluster 3 (accuracy la plus basse, z le plus négatif en
valeur absolue) porte lui-même le label "Mixed administrative requests",
cohérent avec une accuracy basse (cluster moins homogène malgré une
conductance dense très faible -- les deux métriques ne mesurent pas la même
chose : compacité géométrique dense vs cohérence sémantique jugée par LLM).

**Limite connue** : pas de baseline dense/instruction-tuned pour comparer
l'accuracy (le papier compare aux deux, §4.3) ; une seule requête testée, un
seul seed ; 68 features seulement (le papier opère sur des dictionnaires de
dizaines de milliers de latents) ; n_clusters fixé à 4 sans justification
empirique (silhouette/gap statistic).

## 85. NPMI_verified (App. E.1/E.3) : 5 paires vérifiées, seuil de dissimilarité de labels recalibré empiriquement

**Question** : sans NPMI_verified, `p1_interesting_correlations.json` reste
une liste de candidats (NPMI brut sur activations SAE + dissimilarité de
labels), pas un résultat comparable au papier (§1/§7,
`docs/archive/audits/AUDIT_SAE_2026-08.md`). `src/analysis/correlations_verified.py` exercé ici
pour la première fois via `scripts/npmi_verified_test.py`.

**Écart à la configuration de référence** : hypothèses = 150 features
d'extension interprétables déjà labellisées (`p1_top_extended_features.json`)
plutôt que le dictionnaire complet ; **seuil de dissimilarité de labels
recalibré** — le seuil absolu du papier (sim<0,2, App E.1) ne sélectionnait
JAMAIS aucune paire ici (diagnostic dédié, job 45630,
`scripts/npmi_similarity_diagnostic.py`) : sur les 165 paires à NPMI>0,3, la
similarité de label observée va de 0,369 à 1,000 (médiane 0,514) — le papier
calibre ce seuil sur des dictionnaires de milliers de features multi-domaines
(CivilComments/Pile), ce dépôt n'en a que 150 sur UN SEUL domaine (emails de
réclamation client EDF), donc même les paires les "moins reliées" y restent
sémantiquement proches en valeur absolue. Remplacé par un seuil RELATIF
(percentile 25 de la distribution observée à chaque run, `LABEL_SIM_PERCENTILE`)
plutôt qu'un nouveau chiffre absolu choisi à la main.

**Méthode statistique** : descriptive (NPMI_verified, CO, effectifs de
présence bruts rapportés explicitement — nécessaire ici, cf. Conclusion).

**n** : 5 paires (après filtres dissimilarité+syntaxique+trivial) × 60
documents train frais.

**Résultat** (job 45631, h100, `npmi_verified.json`) :

| Paire | NPMI (SAE) | NPMI_verified | CO | Présence i / j (sur 60) |
|---|---|---|---|---|
| Réclamation Client ↔ Problèmes énergie | 0,956 | 1,000 | 1,000 | 2 / 2 |
| Réclamation Client ↔ Coordonnées | 0,917 | 1,000 | 1,000 | 1 / 1 |
| Coordonnées ↔ Facture contestée | 0,885 | 1,000 | 1,000 | 1 / 1 |
| Réclamations Clients ↔ Facture élevée | 0,852 | 0,796 | 1,000 | 4 / 2 |
| Numéro Téléphone ↔ Facture contestée | 0,846 | 1,000 | 1,000 | 1 / 1 |

**Conclusion** : les 5 paires survivent la relabellisation indépendante par
juge (NPMI_verified élevé, 0,80-1,00, CO=1,000 partout) — le signal NPMI brut
n'est pas un artefact SAE pur sur ces paires précises. **Mais 4 des 5 ont une
présence de seulement 1-2 documents sur 60** : NPMI_verified=1,000 y reflète
un accord parfait sur un échantillon minuscule (perfection quasi automatique
dès que 1-2 documents co-occurrent et rien d'autre ne contredit), pas une
corrélation statistiquement robuste au sens où le papier l'entend sur des
échantillons de 1k-5k documents. Seule la paire "Réclamations Clients ↔
Facture élevée" (4/2 documents) offre un signal légèrement plus consistant
(NPMI_verified=0,796, en dessous du plafond).

**Limite connue** : `N_VERIFY_DOCS=60` est nettement plus petit que
l'échantillon du papier (1k sur 5k, ~20%) — beaucoup trop petit pour des
concepts rares (1-2 documents positifs), à augmenter avant de citer un
NPMI_verified individuel comme fiable ; le seuil de dissimilarité recalibré
(percentile 25) est spécifique à CE dictionnaire de 150 features, à
revérifier si le dictionnaire de features change (ex. passage au
dictionnaire complet core+extension) ; un seul seed/tirage de corpus.

## 87. N3 : le 89,3% stratifié (§79) n'est pas reconstructible a posteriori — sur-pondération des bins rares confirmée par le code, taux par bin non fiable sur cette tentative

**Question** : N3 (audit externe, docs/archive/audits/AUDIT_SAE_2026-08.md §8) soutient que 89,3%
(sélection stratifiée, §79) n'est pas plus comparable à un chiffre publié que
45,3% (magnitude), dans l'autre sens — le stratifié tire un nombre à peu près
fixe de features par bin de fréquence quel que soit l'effectif du bin,
sur-représentant les bins rares (nombreuses features spécifiques) par rapport
à toute distribution réellement utilisée dans le dépôt. Correctif proposé
sans rerun : publier le taux par bin + un scalaire repondéré
(Horvitz-Thompson) à partir des 150 features déjà sélectionnées et jugées
(`results_v10_emails_main/cache/b2_stratified_selection_rejudge.json`).

**Écart à la configuration de référence** : recalcul rétroactif visé (zéro
rerun de juge), sur les fragments déjà en cache de `results_v10_emails_main/`.

**Méthode statistique** : `feature_selection_stratified_by_frequency` étendue
d'un paramètre `return_bin_info=True` (`src/sae/judge.py`) pour exposer la
strate, la fréquence et la taille de bin de chaque feature sélectionnée ;
`horvitz_thompson_mean` ajouté à `src/analysis/stats.py` (poids = 1/π_i,
π_i ≈ bin_n_sampled/bin_population, constant par strate).

**n** : 150 features attendues (§79) ; 68 effectivement exploitables (cf.
Résultat).

**Résultat** : la ré-exécution de `feature_selection_stratified_by_frequency`
avec le même SEED (42) et les mêmes fragments que le run original **ne
reproduit PAS les 150 features de `b2_stratified_selection_rejudge.json`** —
seules 68/150 coïncident (job 45742, `scripts/n3_stratified_bin_rates.py`,
CPU-only, 39s). Une cause réelle mais mineure a été identifiée et corrigée en
cours de route : `b2_stratified_selection_rejudge.py` dérivait `d_total` de
`max(indices de la référence magnitude)+1` au lieu de la largeur réelle du
checkpoint frozen-core (`core_sae.W_dec.shape[0] + d_extra`) — 17404 au lieu
de 17408, 4 colonnes d'extension jamais vues par la sélection stratifiée
originale. Cet écart (4/1024 colonnes) est trop petit pour expliquer à lui
seul un désaccord sur 82/150 features ; la cause dominante reste
non identifiée (candidats non vérifiés : dérive du corpus reconstruit depuis
`Mails.tsv`/`augmented_mails.jsonl`, qui ont pu grossir depuis juillet, sans
qu'aucun fingerprint de corpus n'ait été enregistré pour ce `SAVE_DIR`
"legacy", antérieur au cache d'extraction partagé et à sa clé mécanique R5).
Sur les 68 features communes uniquement (population non représentative,
biaisée vers les bins déterministes de petit effectif — cf. Limite) :
taux brut 88,2% (60/68 — proche mais PAS égal au 89,3%/134/150 publié, deux
échantillons différents), taux repondéré Horvitz-Thompson **92,0%**.

**Conclusion** : la critique méthodologique de N3 (le stratifié sur-pondère
les bins rares) est confirmée par lecture directe du code —
`per_bin = n_features // n_bins_eff` alloue le même nombre de features par
strate indépendamment de son effectif réel (`src/sae/judge.py::feature_
selection_stratified_by_frequency`), un fait structurel, pas une hypothèse à
mesurer. En revanche, **produire un taux par bin ou repondéré FIABLE pour le
89,3% déjà publié n'est pas possible sans le rejouer** : la sélection
stratifiée n'est pas reconstructible a posteriori sur ce dépôt dans son état
actuel. `b2_stratified_selection_rejudge.py` est corrigé pour capturer
`bin_info` DIRECTEMENT au moment de la sélection (`return_bin_info=True`,
sauvé dans `OUT_PATH["bin_info"]`) — un futur rerun produira un taux par bin
fiable sans dépendre d'une reproduction ultérieure. Aucun rerun lancé cette
session (coût GPU non trivial, hors du "zéro rerun" visé par N3 tel que
reçu).

**Limite connue** : le taux 88,2%/92,0% ci-dessus porte sur un sous-ensemble
de 68 features dont la composition par bin est elle-même biaisée — les bins
à faible effectif (population ≤ per_bin, sélection déterministe
indépendante du seed) sont systématiquement présents dans l'intersection,
les bins à fort effectif (sélection aléatoire, sensible à toute divergence
de seed/état RNG/fragments) largement absents — donc pas un résultat à citer
comme le "vrai" taux repondéré de §79, seulement une démonstration que
`horvitz_thompson_mean`/le pipeline de bin_info fonctionnent. `n3_stratified_
bin_rates.json` (`results_v10_emails_main/cache/`) contient le détail complet
si une inspection supplémentaire est utile avant de rejouer b2.

## 88. Réplication du balayage layer (§51) sous le setup classique (K_EXTRA=5, sélection stratifiée) : layer 41 — cohérent avec le plateau déjà observé sur le sweep taille de modèle

**Question** : §82 sweep la taille du modèle extracteur (4B/12B/27B) sous le
setup classique. Ce sweep ne varie que `LAYER` à taille de modèle fixe (12B,
`gemma-3-12b-it`) pour situer layer 41 (~2/3 profondeur du palier 27B,
`_PRESETS`) par rapport à layer 31 (référence de facto du sweep taille
ci-dessus, déjà stratifié) et layer 24 (défaut historique du dépôt, mesuré
sous magnitude uniquement). Résultat déjà produit par le job 45367 (antérieur à l'introduction de
`JUDGE_MODEL_ID`/la décision de juger tout avec Qwen, cf. Limite) mais jamais
écrit dans ce journal.

**Écart à la configuration de référence** : uniquement `LAYER=41`
(`SAE_ID="layer_41_width_16k_l0_medium"`), reste identique au sweep §82
(`MODEL_ID=gemma-3-12b-it`, `K_EXTRA=5`, `D_EXTRA=1024`, 25M tokens, corpus
mixte, sélection stratifiée — défaut inchangé depuis §79).

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`).

**n** : 150 features.

**Résultat** (job 45367, h100,
`results_v33_ablation_classic_setup_k5_25m_layer41/cache/p1_judge_labels_extended.json`) :
82,7% (124/150, IC95% Wilson [75,8% ; 87,9%]) — comparé à la valeur déjà
publiée en référence du sweep taille de modèle (12B/layer 31, stratifié,
82,0%, 123/150, §82) : z=-0,15, **p=0,88, non significatif**.

**Conclusion** : confirme le plateau déjà observé entre 12B et 27B (§82,
p=0,76) — passer de layer 31 à layer 41 (toujours à taille de modèle fixe,
12B) ne déplace pas non plus le taux mesurable à n=150. La couche `resid_post`
choisie pour le core (24, historique) n'est donc pas identifiée comme
sous-optimale par ce balayage étendu ; le seul écart individuel qui reste
significatif contre elle est layer 31 (§51, non répliqué sous stratifié à ce
jour), mais layer 31 lui-même ne se distingue plus de layer 41 une fois la
sélection stratifiée appliquée aux deux. `layer 12` (job 45725, en file au
moment de l'écriture) complètera ce balayage à l'autre extrémité.

**Limite connue** : un seul seed ; **jugé par `gemma-3-12b-it` auto-jugeant
ses propres features** (job antérieur au découplage `JUDGE_MODEL_ID`), pas par
Qwen3.8-27B — sous la décision explicite de standardiser le juge à Qwen pour
tout le dépôt (pour éliminer le biais d'auto-préférence plutôt que de le
tolérer par confort), **ce chiffre est un point de repère provisoire, pas la
valeur finale à citer dans le rapport** : à rejuger avec Qwen avant citation
définitive (rejudge peu coûteux, réutilise `p1_top_extended_features.json`
déjà en cache, même patron que `judge_model_separation_test.py`) — comme le
reste du sweep taille de modèle (§82) et la référence layer 24
(`results_v10_emails_main`), tous mesurés sous le même juge auto-référent
avant cette décision.

## 89. B.1/B.2, arme mixte stratifiée rejugée Qwen : 94,7% — nouveau chiffre de référence, écart à Gemma resserré et à la limite de la significativité

**Question** : §79 (89,3%, 134/150) et §81 (B.1, résolu négativement contre
82,0%) sont tous deux jugés par gemma-3-12b-it auto-référent, antérieurs à la
décision de juger tout avec Qwen3.8-27B (docs/archive/audits/AUDIT_SAE_2026-08.md §9, N4). Ce
run rejuge les 150 features de l'arme MIXTE stratifiée (§79,
`b2_stratified_selection_rejudge.json`, déjà en cache) avec Qwen, mêmes
exemples, seul le juge change — même patron que §83
(`judge_model_separation_test.py`), isolation délibérée (pas de
déduplication par mail parent ici, cf. commit qui l'introduit : mélanger les
deux changements casserait la comparaison à exemples fixes).

**Écart à la configuration de référence** : uniquement le juge
(`scripts/b1_stratified_mixte_qwen_rejudge.py`, `ALT_JUDGE_MODEL_ID=Qwen3.8-27B`).

**Méthode statistique** : `paired_mcnemar_test` (`src/analysis/stats.py`,
exact) sur les 150 paires (même feature, deux juges).

**n** : 150 features.

**Résultat** (job 45735, h100,
`b1_stratified_mixte_qwen_rejudge_Qwen3.8-27B_seed42.json`) :

| Juge | Taux interp. |
|---|---|
| gemma-3-12b-it (référence, §79) | 89,3% (134/150) |
| Qwen3.8-27B | **94,7% (142/150)** |

Accord 90,7% (136/150) ; 11 features basculent non-interprétable→interprétable
avec Qwen contre 3 dans l'autre sens. McNemar exact sur les 14 paires
discordantes : statistique=3,0, **p=0,057** (non significatif, à la limite).

**Conclusion** : le sens de l'écart reproduit celui de §83 (Qwen plus
généreux que Gemma) mais l'AMPLEUR est très différente : +33,4 points sous
sélection par magnitude (§83, p=1,9e-8) contre +5,4 points sous sélection
stratifiée ici (non significatif). Combiné à §90 ci-dessous (layer 41,
stratifié : écart Qwen/Gemma non significatif, sens même légèrement
inversé), un pattern émerge : **l'effet du choix de juge est concentré sur
les features que la sélection par magnitude sur-échantillonne** (denses,
proches de directions génériques) — sous sélection stratifiée, qui évite
justement ces features, Gemma et Qwen s'accordent à 90%+ et l'écart résiduel
n'atteint pas la significativité à n=150. **94,7% devient le chiffre de
référence stratifié/Qwen à citer** en remplacement de 89,3% (Gemma) pour le
taux d'interprétabilité principal du rapport.

**Limite connue** : une seule graine ; le rejugement porte sur les EXEMPLES
déjà fixés par le run Gemma original (pas de déduplication par mail parent
ici, cf. §94 — un futur run "propre" combinant stratifié+Qwen+dédoublonnage
donnerait un chiffre encore plus définitif mais nécessiterait un rejugement
complet, pas un simple rejugement sur exemples figés) ; l'arme
originaux+filler (§81) reste non rejugeable sans extraction fraîche
(fragments supprimés) — la comparaison B.1 complète (deux bras, même juge)
reste hors de portée.

## 90. Sweep layer, suite (§88) : layer 41 rejugé Qwen — écart à Gemma non significatif, sens inversé par rapport à §83/§89

**Question** : complète §88 (layer 41, Gemma auto-référent, 82,7%) avec le
rejugement Qwen, même patron que §83/§89.

**Écart à la configuration de référence** : uniquement le juge, sur les 150
features de `results_v33_ablation_classic_setup_k5_25m_layer41/`.

**Méthode statistique** : `paired_mcnemar_test` (approximation — 40 paires
discordantes, au-delà du seuil habituel de l'exact binomial).

**n** : 150 features.

**Résultat** (job 45801, h100,
`results_v33_.../cache/p1_judge_model_separation_Qwen3.8-27B_seed42.json`) :

| Juge | Taux interp. | IC95% (Wilson) |
|---|---|---|
| gemma-3-12b-it (référence, §88) | 82,7% (124/150) | [75,8% ; 87,9%] |
| Qwen3.8-27B | 78,7% (118/150) | [71,4% ; 84,5%] |

Accord 73,3% (110/150) ; 17 features basculent non-interprétable→interprétable
avec Qwen, 23 dans l'autre sens. McNemar : statistique=0,625, **p=0,429**
(non significatif).

**Conclusion** : cohérent avec §89 — sous sélection stratifiée, l'écart
juge Gemma/Qwen n'atteint la significativité dans AUCUNE des deux
réplications testées (§89 : +5,4 points, p=0,057 ; ici : −4,0 points,
p=0,429, sens même légèrement inversé). Renforce l'hypothèse que l'écart
massif de §83 (magnitude, +33,4 points) est un artefact du biais de
sélection par magnitude, pas une propriété générale du choix de juge sur ce
dépôt.

**Limite connue** : IC95% larges qui se chevauchent fortement (une seule
graine par juge) ; comme §89, pas de déduplication par mail parent dans ce
rejugement à exemples figés.

## 91. App D.2 (diffing structuré, génération JSON par LLM) — première exécution : verification_rate=80%, nettement au-dessus du run K.1 informel (§84, 40%)

**Question** : App D.2 était la pièce la plus fraîche et la moins vérifiée
de la session précédente (génération structurée d'hypothèses par LLM,
risque de parsing raté ou d'hypothèses dégénérées comme pour NPMI_verified
au premier essai) — job 45750 lancé en fin de session, jamais confirmé
terminé ni le JSON inspecté en détail avant cette session.

**Écart à la configuration de référence** : hypothèses générées par
`src/analysis/diff_hypothesis_generator.py` (prompt structuré, top-200/seuil
0,03 par différence de fréquence, `select_top_diff_features_by_frequency`)
plutôt que reprises d'un CSV archivé comme en §84 ; corpus de vérification
identique (40 documents energy + 40 sports, FineWeb2-fr, frais) ; même
`HypothesisVerifier` (App K.1), même seuil 1%.

**Méthode statistique** : `compute_verification_metrics`
(`src/analysis/hypothesis_verifier.py`), descriptive (Figures 11/12 du
papier).

**n** : 10 hypothèses × 80 documents.

**Résultat** (job 45750, h100, `diffing_structured_hypotheses.json`) :

| Métrique | Valeur |
|---|---|
| `verification_rate` (seuil 1%) | **80,0% (8/10)** |
| `coverage` | 55,0% |

Détail : 8 hypothèses valides (|écart vérifié| > 1 point), 2 invalides
(différence nulle dans les deux groupes — "regulations, compliance limits,
penalties" et "promotional language, free codes, bonuses", confiance de
génération la plus basse du lot, 0,75 et 0,65 respectivement). Les 8
hypothèses valides couvrent les deux sens (`dataset: target/other` de la
génération) : `verification_rate` mesure `|rate_in_group - rate_out_group| >
seuil`, PAS que le sens prédit par le générateur soit confirmé — une
hypothèse "other" (ex. sport) validée signifie que l'écart existe, pas
nécessairement dans le sens énergie>sport (cohérent avec la définition du
papier, Figure 11, déjà utilisée par §84 — pas une divergence introduite
ici).

**Conclusion** : verification_rate=80% est nettement supérieur au 40% de
§84 — la génération structurée (JSON, confiance explicite, top-200 par
différence de fréquence) sélectionne des hypothèses plus robustes que le
CSV archivé (top par q-value NPMI, juillet, jamais régénéré) réutilisé en
§84. Aucun signe de dégénérescence (pas de parsing raté, hypothèses
lisibles et cohérentes sémantiquement avec leurs labels de features
sources) — App D.2 est fonctionnel et donne un signal net.

**Limite connue** : n=10 hypothèses, IC non calculé ici (comparaison
descriptive à §84, pas de test formé faute d'un n commun) ; toujours pas de
baseline LLM-only (Figure 11 du papier) pour situer ce taux en absolu ; un
seul tirage de corpus/seed.

## 92. Retrieval RRF + reranking LLM + RBO (App G), label remboursement resserré (N5) — le reranking domine uniformément, RBO confirme que TF-IDF et Latent Terms trouvent des documents différents

**Question** : complète §80 (Latent Terms vs TF-IDF, ancien label
remboursement bruité par "l'avoir"/"d'avoir") avec (a) le label resserré
(N5, docs/archive/audits/AUDIT_SAE_2026-08.md §8) et (b) la fusion RRF + reranking LLM + RBO,
jamais mesurés avant cette session (App G implémentée mais jamais câblée
sur un pipeline de requêtes réel avant `latent_retrieval_precision_eval.py`).

**Écart à la configuration de référence** : `INTENT_KEYWORDS_FR["remboursement"]`
resserré (`un\s+avoir|mon\s+avoir|notre\s+avoir`, "l'avoir"/"d'avoir"
retirés) ; sinon protocole identique à §80 (4 requêtes paraphrasées, corpus
complet 3480 mails).

**Méthode statistique** : descriptive (MAP/MP@k agrégées,
`src/analysis/metrics.py::mean_average_precision/mean_precision_at_k/
reciprocal_rank_fusion/rank_biased_overlap`).

**n** : 4 requêtes, corpus complet.

**Résultat** (job 45745, h100, `latent_retrieval_precision_results.json`) :

| Intention (taux de base) | TF-IDF P@10 | Latent Terms P@10 | RRF P@10 | RRF+rerank P@10 |
|---|---|---|---|---|
| réclamation (54,8%) | 1,00 | 0,50 | 0,90 | **1,00** |
| remboursement (10,3%) | 0,30 | 0,90 | 0,70 | **0,90** |
| information (60,7%) | 0,70 | 0,70 | 0,80 | **0,90** |
| urgence (33,8%) | 0,80 | 0,80 | 0,80 | **1,00** |

| Méthode | MAP | MP@10 | MP@20 |
|---|---|---|---|
| TF-IDF | 0,678 | 0,700 | 0,700 |
| Latent Terms | 0,775 | 0,725 | 0,712 |
| RRF | 0,761 | 0,800 | 0,675 |
| **RRF + rerank LLM** | **0,922** | **0,950** | **0,938** |

RBO (TF-IDF vs Latent Terms) moyen : **0,051** (0,012 à 0,122 selon la
requête) — très bas.

**Conclusion** : le label resserré (N5) ne change PAS le résultat déjà
publié de §80 sur `remboursement` (P@10 Latent Terms=0,90, TF-IDF=0,30
identiques au chiffre pré-N5 ; seul le taux de base recule légèrement,
11,1%→10,3%, cohérent avec le retrait d'un peu de bruit sans changer les
documents pertinents en tête de liste) — la conclusion de §80 était déjà
robuste à ce défaut. **Nouveau résultat, net et uniforme** : le reranking
LLM sur le RRF domine sur les 4 intentions (P@10 ≥ 0,90 partout, jamais
inférieur à RRF seul), MAP passe de 0,761 (RRF) à 0,922 — premier résultat
sans compromis de tout le chapitre retrieval (§80 opposait Latent Terms et
TF-IDF sans vainqueur uniforme). Le RBO très bas confirme que TF-IDF et
Latent Terms ne sont pas simplement deux réordonnancements du même ensemble
de documents : même quand leur P@10 est proche (`information`, 0,70 vs
0,70), les documents en tête diffèrent largement — les deux méthodes
apportent un signal réellement différent, ce que le reranking exploite.

**Limite connue** : toujours n=4 requêtes (une par intention, pas de
réplication) — la réserve méthodologique déjà documentée (vérité terrain
`INTENT_KEYWORDS_FR`, lexicale, favorise structurellement TF-IDF) s'applique
identiquement ici ; le prompt de reranking du dépôt diverge du prompt exact
du papier (non publié, `docs/references.md`) ; un seul juge (Qwen, défaut
actuel) pour le reranking, pas de comparaison à Gemma sur cette tâche
précise.

## 93. Pipeline 2, première mesure d'interprétabilité valide : 58,7% sous sélection stratifiée + juge Qwen + déduplication par mail parent

**Question** : jusqu'ici, le seul chiffre d'interprétabilité Pipeline 2
(74,7%, 112/150) datait du 17 juillet — sélection par magnitude pure (jamais
corrigée pour P2, contrairement à P1 depuis §79), juge gemma-3-12b-it
auto-référent, et sans déduplication par mail parent des exemples positifs
(le correctif de cette session). Jamais cité dans `RESULTS_TESTS.md` ni le
rapport (qui ne cite que NMSE/`clf_acc_email_axes` côté P2, jamais
l'interprétabilité). Ce run mesure le premier chiffre sous le protocole
actuel.

**Écart à la configuration de référence** : `FEATURE_SELECTION_METHOD=stratified`
(nouveau, `feature_selection_stratified_by_frequency_dense`) au lieu du
`topk` magnitude ; juge Qwen3.8-27B (défaut, `JUDGE_MODEL_ID`) au lieu de
gemma-3-12b-it ; déduplication par mail parent des exemples positifs
(`doc_groups=test_groups`, nouveau) ; `SAVE_DIR` inchangé
(`results_v10_emails_main`), réutilise les embeddings F2LLM-v2-80M déjà en
cache (pas de ré-extraction), réentraîne uniquement le `PhraseLevelSAE`
(checkpoint supprimé par le nettoyage disque).

**Méthode statistique** : descriptive (`proportion_with_ci`, Wilson).

**n** : 150 features.

**Résultat** (job 45805, h100, `results_v10_emails_main/cache/
p2_feature_labels.json`, sidecar `p2_feature_labels.json.meta.json` confirmant
`judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) :

**58,7% (88/150) interprétable, IC95% [50,7% ; 66,3%]** — 17/150 `dead_feature`
(moins de 3 exemples positifs valides après déduplication par mail parent).
Reconstruction (même job) : NMSE=0,0671 (meilleur que le 0,0745 déjà cité
dans le rapport pour F2LLM-v2-80M, §3.5), ρ_SAE=0,9614, dead%=0,4% (sur le
dictionnaire complet, 8192 features), `clf_acc_email_axes`=81,4% (vs 79,3%
cité dans le rapport).

**Conclusion** : à la différence de Pipeline 1 (où stratifier a PLUS QUE
DOUBLÉ le taux mesuré, 45,3%→89,3%), corriger la méthodologie de sélection/
jugement de Pipeline 2 fait BAISSER le taux mesuré (74,7%→58,7%) — les trois
changements (sélection, juge, déduplication) sont bundlés dans ce run, pas
décomposés individuellement, donc impossible d'attribuer la baisse à un
facteur précis sans rejugements supplémentaires. Ce que ça confirme sans
ambiguïté : **Pipeline 2 n'est PAS aussi mature que le rapport le laisse
entendre** ("les deux pipelines fonctionnent... avec des résultats
quantifiés" — vrai pour la reconstruction, pas pour l'interprétabilité tant
que le seul chiffre venait d'un protocole obsolète). 58,7% reste un taux
raisonnable (pas un effondrement), mais c'est désormais le chiffre à citer,
pas 74,7%.

**Limite connue** : les 3 facteurs de changement ne sont pas isolés (coût
d'un ablation complet non engagé cette session) ; une seule graine ; le
`dead_feature` du juge (17/150, "moins de 3 exemples positifs trouvés dans
l'échantillon") est une notion DIFFÉRENTE du `dead_pct` de reconstruction
(0,4% sur le dictionnaire complet) — un artefact d'échantillonnage possible
(peu de documents parmi les `sample_docs` tirés activent la feature) plutôt
qu'une feature réellement morte, pas décomposé ici. Même ambiguïté à trois
notions de "mort" repérée côté P1 (`dead_frac` d'entraînement toujours à 0,
`dead_pct` final de reconstruction, `dead_feature` du juge) — `dead_pct_core`/
`dead_pct_extension` (`src/analysis/metrics.py`) séparent déjà la première
confusion (core GemmaScope vs extension) mais pas encore celle-ci
(reconstruction vs échantillonnage du juge).

## 94. N3 résolu : taux par bin et repondération Horvitz-Thompson sur un rerun natif (stratifié + Qwen + déduplication par mail parent)

**Question** : §87 concluait que le taux repondéré fiable pour le 89,3%
(§79) n'était pas reconstructible a posteriori (sélection stratifiée non
reproductible sur les fragments/corpus actuels). `b2_stratified_selection_rejudge.py`
capture désormais `bin_info` nativement (`return_bin_info=True`) au moment
de la sélection — ce run l'exerce pour la première fois, en même temps que
le juge Qwen (`MODEL_ID` surchargé, cf. commit du correctif R5 sur `OUT_PATH`)
et la déduplication par mail parent (correctif de cette session).

**Écart à la configuration de référence** : sélection stratifiée refaite au
moment du run (pas de reproduction a posteriori), juge Qwen3.8-27B au lieu
de gemma-3-12b-it, déduplication par mail parent activée — trois
changements simultanés par rapport à §79, comme §93 (P2).

**Méthode statistique** : `horvitz_thompson_mean` (`src/analysis/stats.py`,
poids $1/\pi_i$, $\pi_i$ = `bin_n_sampled`/`bin_population` par strate).

**n** : 150 features, 9 strates non vides (sur 10 bins log-espacés — la
strate 3 n'a aucun membre vivant sur cette tentative).

**Résultat** (job 45887, h100,
`results_v10_emails_main/cache/b2_stratified_selection_rejudge_Qwen3.8-27B.json`) :

| Bin | n sélectionné | Interp. | Taux | Fréquence | Population du bin |
|---|---|---|---|---|---|
| 0 | 2 | 2 | 100,0% | 0,012–0,014 | 2 |
| 1 | 2 | 2 | 100,0% | 0,024 | 2 |
| 2 | 2 | 2 | 100,0% | 0,034–0,040 | 2 |
| 4 | 2 | 2 | 100,0% | 0,098–0,100 | 2 |
| 5 | 3 | 3 | 100,0% | 0,124–0,146 | 3 |
| 6 | 4 | 4 | 100,0% | 0,186–0,254 | 4 |
| 7 | 15 | 14 | 93,3% | 0,288–0,404 | 16 |
| 8 | 19 | 17 | 89,5% | 0,440–0,632 | 63 |
| 9 | 101 | 95 | 94,1% | 0,652–1,000 | 930 |

**Taux brut (non pondéré) : 94,0% (141/150).** **Taux repondéré
Horvitz-Thompson : 93,85%.**

**Conclusion** : la critique structurelle de N3 (le stratifié sur-échantillonne
les strates rares — ici les 4 premières strates ont une population de 2-4
features mais contribuent chacune 2-4 features à l'échantillon de 150,
sur-représentées d'un facteur ~40-75× par rapport à leur poids réel dans le
dictionnaire) est confirmée à nouveau par construction. **Mais son impact
empirique sur le taux agrégé est minime** : 94,0% brut contre 93,85%
repondéré, un écart de 0,15 point. Les strates rares sur-échantillonnées
sont toutes à 100% d'interprétabilité (features très spécifiques, peu
fréquentes mais faciles à juger une fois identifiées) — les repondérer à la
baisse ne change presque rien puisqu'elles ne tirent pas le taux vers le
haut de façon disproportionnée. **N3 est donc résolu : le biais est réel en
principe, négligeable en pratique sur ce dictionnaire.** 94,0% (brut) et
93,85% (repondéré) sont tous deux citables ; le brut suffit, la nuance
méthodologique n'a pas besoin d'être portée jusqu'au chiffre final du
rapport.

**Limite connue** : un seul seed/tirage ; la strate 9 (la plus peuplée, 930
features, 101 échantillonnées) domine numériquement le calcul — un tirage
différent dans cette seule strate pourrait faire bouger le taux repondéré
plus que les strates rares elles-mêmes ; comme §89/§93, trois changements de
protocole simultanés (sélection refaite, juge, déduplication) empêchent
d'attribuer l'écart avec 89,3%/94,7% à un facteur isolé.

## 95. Sweep taille de modèle, 1B sous méthodologie pleinement corrigée : l'effet d'échelle reste significatif mais son ampleur apparente était pour moitié un artefact de méthode

**Question** : §82 sweepait 4B/12B/27B sous sélection stratifiée mais juge
Gemma auto-référent, sans déduplication. Le 1B avait échoué deux fois (N1,
crash `KeyError: raw_acts` ; puis OOM sur `a100`, cf. `docs/ops.md`) avant
d'aboutir sous le code actuel — stratifié, juge Qwen (défaut), et
déduplication par mail parent nativement dans `saev5.py`, sans qu'aucune
configuration n'ait dû être forcée pour ça. C'est le premier point du sweep
mesuré sous méthodologie pleinement corrigée dès le premier passage,
pas par rejugement d'exemples déjà figés (contrairement à §89/§90/§94).

**Écart à la configuration de référence** : identique à §82 (K_EXTRA=5,
D_EXTRA=1024, 25M tokens, corpus mixte) sauf `MODEL_ID=gemma-3-1b-it`,
`LAYER=13`, `SAE_ID=layer_13_width_16k_l0_medium`.

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`).

**n** : 150 features.

**Résultat** (job 45874, h100, `results_v29_.../cache/p1_judge_labels_extended.json`,
sidecar confirmant `judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : **80,0% (120/150), 0 dead/insuffisant.** Comparé
au 12B sous la même méthodologie complète (§94, 94,0%, 141/150) :
$z=-3{,}61$, **$p=0{,}0003$** (significatif), $h$ de Cohen$=-0{,}43$ (effet
moyen). Comparé au 12B sous l'ANCIENNE méthodologie (magnitude, Gemma
auto-référent, référence historique du sweep, 45,3%, 68/150 — comparaison
non appariée, juge/sélection/corpus tous différents) :
l'écart brut est de +34,7 points au lieu de −33,3 points côté 1B/12B
historique (12,0% vs 45,3%, $z=-6{,}38$, $p=1{,}7\times10^{-10}$, $h=-0{,}77$,
effet large).

**Conclusion** : l'effet d'échelle du modèle **reste statistiquement
significatif** sous méthodologie pleinement corrigée, mais son **ampleur
apparente était pour une bonne moitié un artefact de méthode** :
$h=-0{,}77$ (effet large) sous l'ancienne méthodologie devient $h=-0{,}43$
(effet moyen) une fois sélection et juge corrigés — l'écart brut passe de
33,3 points (12,0%→45,3%) à 14,0 points (80,0%→94,0%). Cohérent avec
§89/§90 (l'effet de choix de juge est concentré sur les features que la
sélection par magnitude sur-échantillonne) : le 1B semble avoir été
pénalisé DEUX FOIS par l'ancienne méthodologie — une fois par la sélection
par magnitude (features denses/génériques, difficiles à interpréter quel
que soit le modèle), une fois par l'auto-jugement sévère d'un petit modèle
sur ses propres features. Le résultat central du stage ("l'échelle du
modèle domine tout effet architectural du SAE") tient toujours, mais le
chiffre qui l'illustre (33,3 points d'écart 1B/12B) doit être requalifié :
le vrai effet d'échelle, isolé des biais de méthode, est significatif mais
deux fois plus modeste (~14 points).

**Limite connue** : comparaison au 12B non appariée (features différentes
entre 1B et 12B, comme pour tout le sweep taille de modèle) ; 4B et 27B pas
encore remesurés sous méthodologie pleinement corrigée (seulement sous
stratifié+Gemma, §82 : 72,0%/83,3%) — la courbe complète 1B→4B→12B→27B sous
méthodologie corrigée reste à tracer avant de conclure sur la forme exacte
du plateau ; un seul seed.

## 96. Sweep layer, layer 12 sous méthodologie pleinement corrigée : pas d'écart significatif avec layer 31

**Question** : complète §90 (layer 41) pour le layer sweep sous méthodologie
complète — layer 12, jamais mesuré sous stratifié+Qwen+déduplication avant
ce run (crashé deux fois auparavant : N1 puis clé de cache orpheline,
`docs/archive/audits/AUDIT_SAE_2026-08.md` §9).

**Écart à la configuration de référence** : identique à §82/§88 sauf
`LAYER=12`.

**Méthode statistique** : `two_proportion_test`.

**n** : 150 features.

**Résultat** (job 45803, h100, `results_v32_.../cache/p1_judge_labels_extended.json`,
sidecar confirmant `judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : **88,7% (133/150), 0 dead**, IC95% [82,6% ; 92,8%].
Comparé à layer 31 sous la même méthodologie complète (§94, 94,0%) :
$z=-1{,}64$, $p=0{,}101$, non significatif.

**Conclusion** : cohérent avec §90 (layer 41) — sous méthodologie
pleinement corrigée, aucun layer testé (12, 41) ne se distingue
significativement de layer 31 à $n=150$. Le seul écart individuel
significatif du dépôt (layer 31 vs layer 24, §51, $z=2,20$) n'a toujours
pas été répliqué sous cette méthodologie ; layer 24 lui-même reste à
remesurer sous stratifié+Qwen+déduplication pour clore le sweep layer.

**Limite connue** : un seul seed ; comparaison non appariée (features
différentes entre layers).

**Correction post-hoc (§95/§96)** : les deux sections ci-dessus comparent
1B et layer 12 (famille "classic setup" K_EXTRA=5/25M tokens, `results_v29`/
`results_v32`) contre le "12B" de §94 — mais §94 porte sur
`results_v10_emails_main` (K_EXTRA=32, 500 000 tokens, **layer 24**), pas sur
`results_v27_.../layer31` (K_EXTRA=5, 25M tokens, layer 31), le véritable
comparateur à variable unique isolée pour cette famille de sweep (c'est lui
qui donne 82,0%/123 sous stratifié+Gemma dans la table §82, pas 89,3%). Les
tests statistiques de §95/§96 comparent donc deux configurations qui
diffèrent par K_EXTRA, volume ET layer, pas seulement par la taille du
modèle/le layer visé — confond plusieurs variables à la fois. `results_v27`
a également eu son cache purgé (comme les répertoires 4B/27B de la même
famille) et n'a jamais été remesuré sous méthodologie pleinement corrigée :
un rerun natif est nécessaire pour obtenir le bon point de comparaison. Les
taux bruts (80,0% pour 1B, 88,7% pour layer 12) restent valides tels quels ;
seuls les tests de significativité contre "12B" sont à refaire une fois
`results_v27` remesuré (job lancé,
cf. priorisation de la campagne dans `docs/archive/audits/AUDIT_SAE_2026-08.md` §9).

## 97. Run de référence (R0) : remplace `results_v27` purgé, comparateur unique de la campagne finale

**Question** : établir une référence unique sous méthodologie finale
(sélection stratifiée + juge Qwen3.8-27B + déduplication par mail parent) à
12B/layer 31/K_EXTRA=5/D_EXTRA=1024/25M tokens — le point de comparaison à
variable unique isolée que §95/§96 identifient comme manquant (`results_v27`,
seul candidat existant pour ce rôle, avait son cache purgé). Toute la
campagne d'ablations finale (C1, V1/V2, A1/A2, S1/S2, L1) compare contre ce
run.

**Écart à la configuration de référence** : aucun — ce run EST la
configuration de référence (`docs/archive/evaluation_protocol.md`).

**Méthode statistique** : `proportion_with_ci` (Wilson) pour le taux seul ;
`two_proportion_test`/`paired_mcnemar_test` pour toute comparaison contre ce
point (utilisé ci-dessous contre S1 et contre le 1B de §95).

**n** : 150 features jugées (voir Limite connue — la campagne prévoyait
n=300 pour ce run précis, le script de lancement n'a pas repris cette
valeur).

**Résultat** (job 45893, h100, `results_v27_ablation_classic_setup_k5_25m_layer31/`,
sidecar `p1_judge_labels_extended.json.meta.json` confirmant
`judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : **84,7% (127/150)**, IC95% Wilson [78,0% ; 89,6%].
Fidélité de reconstruction (`_fve_pair`, console — `fve_extended`/`fve_pretrained`
non encore persistés dans `results.json` pour ce run, corrigé depuis pour les
runs suivants) : FVE(core seul)=0,7451, FVE(core+extension)=0,8844,
**ΔFVE=+0,1393**.

Comparé au 1B pleinement corrigé (§95, 80,0%, 120/150) : $z=-1{,}06$,
$p=0{,}289$, **non significatif**. Comparé au 4B pleinement corrigé (§100,
81,3%, 122/150) : $z=0{,}77$, $p=0{,}442$, **non significatif**. Tendance de
Cochran-Armitage sur les trois points (1B/4B/12B, tous sous méthodologie
pleinement corrigée) : $z=1{,}05$, $p=0{,}293$, **non significative**.

**Conclusion** : **sous méthodologie totalement homogène (stratifié + Qwen +
dédup aux trois échelles), la progression 1B→4B→12B (80,0%/81,3%/84,7%) n'est
plus significative, ni par paire ni en tendance.** Ce n'est plus une
requalification à la baisse d'un facteur deux comme l'annonçait §95 (qui ne
comparait le 1B corrigé qu'au 12B *non* corrigé de §82/§94) — c'est,
provisoirement et sous réserve de S2 (27B, en attente), la **disparition de
la significativité de l'effet d'échelle une fois les trois points mesurés
sous le même protocole**. Le "résultat central du stage" tel qu'actuellement
formulé dans le rapport de stage (§model-scale) n'est plus soutenu
par ces trois chiffres et doit être révisé avant toute autre rédaction — pas
seulement requalifié à la baisse, sa significativité elle-même est
maintenant en question à ce sous-ensemble de tailles.

**Limite connue** : n=150 et non 300 comme prévu par la campagne (le script
de lancement n'a pas repris `N_FEATURES_TO_LABEL=300` ; n=150 retenu tel
quel comme référence définitive, décision utilisateur, pas de rerun à 300) ;
un seul seed (cf. §99/§100 pour le plancher de bruit une fois disponibles) ;
S2 (27B) manquant à ce jour pour clore le sweep sous méthodologie homogène —
la conclusion ci-dessus peut encore changer de sens si 27B rouvre un écart.

## 98. C1 — sanity check décodeur figé aléatoire sous méthodologie finale (Korznikov et al. 2026)

**Question** : le sanity check §19 (45,3% vs 29,3%, décodeur entraîné vs
figé aléatoire) mesurait sous sélection par magnitude + juge auto-référent —
les deux biais corrigés depuis (§79, §89). Sous méthodologie finale
(stratifié + Qwen3.8-27B + dédup), un décodeur figé à une initialisation
aléatoire (jamais entraîné, `SANITY_CHECK_FROZEN_DECODER=1`,
`FrozenDecoderExtendedSAE`) approche-t-il encore R0 (§97) ? Un rapprochement
marqué remettrait en cause la validité du protocole d'évaluation lui-même,
pas seulement le SAE.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
décodeur extra figé aléatoire au lieu d'entraîné (`W_dec_extra` gelé à son
init PCA-free, seuls `W_enc_extra`/`b_enc_extra`/seuil BatchTopK appris).

**Méthode statistique** : `two_proportion_test` contre R0 (§97).

**n** : 150 prévu — **11 obtenus** (voir Résultat pour la cause, pas un
raccourcissement délibéré).

**Résultat** (job 45953, h100, `results_v37_ablation_c1_sanity_frozen_decoder_stratified_qwen/`,
sidecar confirmant `judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : `dead_pct_extension` = **98,9%** (`results.json`,
contre **5,7%** pour R0 entraîné normalement, §97) — sur les 1024 directions
de l'extension, seules ~11 restent actives sous décodeur figé aléatoire, ce
qui borne mécaniquement le pool de features stratifiables à 11 (et non 150 :
`_select_features` ne peut pas stratifier sur des directions mortes). Sur ces
11 : **54,5% (6/11)**, IC95% Wilson [28,0% ; 78,7%] (large, n petit). Comparé
à R0 (84,7%, 127/150) : $z=2{,}54$, $p=0{,}011$, significatif mais à
interpréter avec prudence vu le $n$ effectif.

Fidélité de reconstruction (console, `_fve_pair`) : FVE(core seul)=0,7451
(identique à R0, même core figé), FVE(core+extension)=0,7452,
**ΔFVE=+0,0001** — contre **ΔFVE=+0,1393 pour R0** (§97). L'extension à
décodeur figé aléatoire n'améliore quasiment pas la reconstruction du tout.

**Conclusion** : trois métriques indépendantes convergent, et la troisième
est la plus propre statistiquement des trois. Le résultat qui **porte** ici
n'est pas le taux d'interprétabilité à $n=11$ (sous-alimenté, IC à ±25 points)
mais `dead_pct_extension` et **`ΔFVE`**, deux métriques continues mesurées
sans le bruit d'échantillonnage du juge : un décodeur figé à une
initialisation aléatoire laisse l'encodeur n'exploiter utilement qu'**~1% de
la capacité** nominale de l'extension (11/1024, contre 94,3% pour un
décodeur entraîné normalement), et cette capacité inutilisée ne sert
**quasiment à rien pour la reconstruction** : ΔFVE tombe de +0,1393 (R0) à
+0,0001 (C1), un facteur ~1400 sur un échantillon de 4096 tokens — bien plus
net et bien moins bruité que l'écart d'interprétabilité à $n=11$. Sous
méthodologie finale comme sous l'ancien protocole (§19), le SAE entraîné bat
donc nettement le décodeur aléatoire — mais la **raison** diffère de ce que
§19 laissait penser : ce n'est pas (seulement) que les quelques directions
aléatoires actives sont individuellement moins interprétables, c'est que la
quasi-totalité de la capacité disponible ne sert à rien, ni pour
l'interprétabilité ni pour la reconstruction, sans direction de décodage
apprise. **`ΔFVE` devrait être la métrique citée en premier pour ce sanity
check dans le rapport** — le taux d'interprétabilité à $n=11$ reste correct
mais sa faible puissance statistique en fait un second argument, pas le
premier.

**Limite connue** : un seul seed ; comparaison non appariée ; le taux
d'interprétabilité 54,5% (6/11) ne doit pas être cité seul dans le rapport
sans le $n=11$ et sans `dead_pct_extension` en regard — un rapporteur qui ne
verrait que "54,5% vs 84,7%, p=0,011" sans ce contexte surestimerait la
robustesse statistique de la comparaison à features individuelles.

## 99. V1 — variabilité run-à-run de R0, seed 123

**Question** : tout le dépôt mesure à n=1 graine d'entraînement SAE
(`SEED`) à ce jour (cf. `CLAUDE.md`, section Diagnostics, point 5). Ce run
rejoue R0 (§97) à l'identique, `SEED=123` au lieu de 42
(`CORPUS_SPLIT_SEED` inchangé, split corpus identique), pour donner un
plancher de bruit run-à-run à la campagne finale — cf. aussi V2 (§101,
seed 7) pour un second point.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`SEED=123`.

**Méthode statistique** : `two_proportion_test` contre R0 (§97) ; à
combiner avec V2 (§101) pour une amplitude de variation run-à-run plutôt
qu'un unique écart ponctuel.

**n** : 150 features.

**Résultat** (job 45954, h100, COMPLETED,
`results_v38_ablation_v1_seed123_classic_setup_k5_25m_layer31/`) :
**90,7% (136/150)**. Contre R0 (84,7%, 127/150) : $z=-1{,}58$, $p=0{,}114$,
non significatif. FVE(core+extension)=0,8849, ΔFVE=+0,1399 (R0 : +0,1393 —
quasi identique malgré 6 points d'écart sur l'interprétabilité : la
reconstruction est beaucoup moins sensible au seed que le jugement LLM).

**Conclusion** : voir §101 (V2) pour la lecture combinée du plancher de
bruit run-à-run — ce point seul montre déjà un écart brut de 6,0 points
(84,7%→90,7%) sans changer un seul hyperparamètre ni le seed du corpus,
uniquement l'init/l'ordre de mélange de l'entraînement SAE.

**Limite connue** : deux graines ne bornent qu'approximativement la
variabilité réelle ; cf. §100 note sur l'effet de seed restreint depuis le
cache d'extraction partagé, N9.

## 100. S1 — 4B sous méthodologie pleinement corrigée (courbe d'échelle, complète §95)

**Question** : §95 n'a rejugé que le point 1B sous méthodologie pleinement
corrigée (stratifié + Qwen + dédup) ; le point 4B restait sous
stratifié+Gemma (§82, 72,0%). Ce run mesure 4B sous méthodologie totalement
homogène avec R0 (§97) et le 1B de §95, pour reconstruire la courbe
d'échelle 1B/4B/12B à variable unique isolée.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`MODEL_ID=gemma-3-4b-it`, `MODEL_SIZE=4b`, `LAYER=17`,
GemmaScope `gemma-scope-2-4b-it` dédié.

**Méthode statistique** : `two_proportion_test`/`cochran_armitage_trend_test`
contre le 1B (§95) et R0/12B (§97) — calculs reportés dans la Conclusion de
§97 pour éviter la duplication.

**n** : 150 features.

**Résultat** (job 45894, h100, `results_v30_ablation_classic_setup_k5_25m_model_scale_4b/`,
COMPLETED 05:29:59, sidecar confirmant `judge_model_id=Qwen3.8-27B`,
`feature_selection_method=stratified`, `doc_groups_dedup=true`) :
**81,3% (122/150)**, IC95% Wilson [74,3% ; 86,8%]. FVE(core seul)=0,6323
(bien plus bas que R0/12B : 0,7451 — le core lui-même explique moins la
variance au layer 17/4B), FVE(core+extension)=0,9317, **ΔFVE=+0,2994**, plus
du double de R0 (+0,1393) : l'extension compense davantage un core plus
faible, cohérent avec la logique de l'architecture, mais ce gain de
reconstruction ne se traduit PAS en gain d'interprétabilité (81,3% contre
84,7% pour R0) — la fidélité de reconstruction et l'interprétabilité jugée
peuvent diverger nettement d'une échelle à l'autre.

**Conclusion** : voir §97 — avec ce point, les trois échelles 1B/4B/12B sous
méthodologie totalement homogène ne montrent plus d'effet d'échelle
significatif, ni par paire ni en tendance (Cochran-Armitage $p=0{,}293$).
Résultat inattendu par rapport à toutes les mesures antérieures de ce
dépôt (§28, §82, §95) qui montraient toutes un effet net — la différence
tient entièrement au fait que celles-ci comparaient des points mesurés sous
des protocoles différents (cf. §96, correction post-hoc).

**Limite connue** : un seul seed ; S2 (27B) manquant pour clore le sweep ;
comparaison non appariée (features différentes entre échelles de modèle).

## 101. V2 — variabilité run-à-run de R0, seed 7 (second point, complète §99)

**Question** : second point de variabilité run-à-run, en complément de V1
(§99, seed 123) — deux écarts indépendants donnent une amplitude de
variation plutôt qu'un unique point, cf. §99 pour la motivation complète.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`SEED=7`.

**Méthode statistique** : `two_proportion_test` contre R0 (§97) et contre V1
(§99).

**n** : 150 features.

**Résultat** (job 45956, h100, COMPLETED,
`results_v39_ablation_v2_seed7_classic_setup_k5_25m_layer31/`) :
**82,7% (124/150)**. Contre R0 (84,7%, 127/150) : $z=0{,}47$, $p=0{,}639$, non
significatif. Contre V1 (90,7%, 136/150) : $z=2{,}04$, $p=0{,}042$,
**significatif** (non corrigé). FVE(core+extension)=0,8852, ΔFVE=+0,1402
(R0 : +0,1393 ; V1 : +0,1399) — les trois graines donnent un ΔFVE quasi
identique à la 3e décimale, alors que le taux d'interprétabilité varie de
8 points entre elles : confirme le constat de §100, la reconstruction est
un signal beaucoup plus stable que le jugement LLM.

**Conclusion** : trois seeds de la même configuration exacte (R0/V1/V2 :
84,7\% / 90,7\% / 82,7\%, IC95\% Wilson [78,0-89,6] / [84,9-94,4] / [75,8-87,9])
donnent un **plancher de bruit run-à-run de ~8 points** (V1 vs V2, la paire la
plus écartée) — et ce plancher est lui-même statistiquement "significatif"
non corrigé ($p=0{,}042$) entre deux runs qui ne diffèrent que par l'init et
l'ordre de mélange de l'entraînement SAE, aucun hyperparamètre. **Conséquence
directe pour toute la campagne d'ablations (§102-108)** : un écart isolé à
$p<0{,}05$ contre R0 ne peut plus être lu comme un effet d'hyperparamètre sans
le comparer à cette amplitude de bruit intrinsèque — A1 (78,0\%, écart de
6,7 points) et A2 (88,0\%, écart de 3,3 points) sont tous deux **à l'intérieur**
de la fourchette V1-V2 (82,7-90,7\%), ce qui renforce leur non-significativité
plutôt que de la remettre en cause. Ceci répond à l'objection attendue d'un
jury sur un tableau à 15 comparaisons non corrigées (\S\ref{subsec:audit-stats}) :
la variabilité de seed, mesurée ici pour la première fois sous le cache
d'extraction partagé, est du même ordre de grandeur que les écarts
d'hyperparamètres testés — ce n'est donc pas seulement une lacune de
correction multi-tests, c'est un plancher de bruit réel et non négligeable
à $n=150$.

**Limite connue** : $n=2$ graines borne grossièrement l'amplitude réelle de
variabilité (l'écart-type effectif reste mal estimé à si peu de points) ;
comparaisons non appariées (features différentes tirées à chaque run).

**Conclusion** : *à compléter.*

**Limite connue** : *à compléter.*

## 102. A1 — `K_EXTRA=32` sous méthodologie finale (défaut historique du dépôt)

**Question** : `K_EXTRA=32` était le défaut du dépôt avant l'ablation
"setup classique du papier" qui a introduit `K_EXTRA=5` (recommandation SAE
Boost, §25/§45) — jamais retesté sous méthodologie finale. Ce run mesure
l'écart contre R0 (§97, `K_EXTRA=5`) sous stratifié+Qwen+dédup.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`K_EXTRA=32`.

**Méthode statistique** : `two_proportion_test` contre R0 (§97).

**n** : 150 features.

**Résultat** (job 45957, h100, COMPLETED,
`results_v40_ablation_a1_k_extra32_classic_setup_25m_layer31/`) :
**78,0% (117/150)**, `dead_pct_extension=0,0%`, `rho_sae=0,924`. Contre R0
(84,7%, 127/150) : $z=1{,}48$, $p=0{,}138$, non significatif. FVE(core+extension)=0,8933,
ΔFVE=+0,1483 (R0 : +0,1393) — reconstruction légèrement meilleure avec plus
de features actives par token, sans gain d'interprétabilité correspondant.

**Conclusion** : `K_EXTRA=32` (défaut historique, plus dense que le `K_EXTRA=5`
recommandé par SAE Boost et retenu par R0) ne se distingue pas significativement
de R0 sur le taux d'interprétabilité, malgré un écart brut de 6,7 points dans
le sens d'une dégradation. `dead_pct_extension=0,0%` confirme qu'une extension
plus dense (32 features actives/token au lieu de 5) ne meurt pas davantage —
la parcimonie plus faible n'est pas un problème d'entraînement, juste sans
bénéfice mesurable sur l'interprétabilité à ce $n$.

**Limite connue** : un seul seed ; $n=150$ laisse une puissance limitée pour
détecter un écart de cette taille (6,7 points) — ne pas lire "non significatif"
comme "prouvé équivalent".

## 103. A2 — `D_EXTRA=2048` sous méthodologie finale (capacité doublée de l'extension)

**Question** : `D_EXTRA=2048` (capacité doublée de l'extension résiduelle)
avait déjà été testé sous magnitude+Gemma (§27, pas d'écart) — jamais sous
méthodologie finale. Ce run mesure l'écart contre R0 (§97, `D_EXTRA=1024`)
sous stratifié+Qwen+dédup.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`D_EXTRA=2048`.

**Méthode statistique** : `two_proportion_test` contre R0 (§97).

**n** : 150 features.

**Résultat** (job 45958, h100, COMPLETED,
`results_v41_ablation_a2_d_extra2048_classic_setup_25m_layer31/`) :
**88,0% (132/150)**, `dead_pct_extension=14,7%`, `rho_sae=0,903`. Contre R0
(84,7%, 127/150) : $z=-0{,}84$, $p=0{,}401$, non significatif. FVE(core+extension)=0,8918,
ΔFVE=+0,1468 (R0 : +0,1393) — reconstruction légèrement meilleure malgré
14,7% de directions mortes, cohérent avec un budget de capacité en excès
plutôt qu'insuffisant.

**Conclusion** : doubler la capacité de l'extension (1024→2048 directions) ne
produit pas d'écart significatif, cohérent avec §27 (déjà sans effet sous
l'ancienne méthodologie). `dead_pct_extension` passe de 5,7% (R0) à 14,7% —
une capacité plus grande laisse mécaniquement plus de directions inutilisées,
sans que cela dégrade `rho_sae` (0,903 contre 0,883) ni le taux
d'interprétabilité mesuré.

**Limite connue** : un seul seed ; comme A1, $n=150$ limite la puissance pour
détecter un écart modéré.

## 104. D1 — sondes d'intention sur les codes de R0 : baseline TF-IDF+LogReg honnête

**Question** : les labels d'intention (`INTENT_KEYWORDS_FR`, `src/data/dataset.py`)
sont construits par mots-clés, donc structurellement lexicaux — une baseline
TF-IDF+LogReg est-elle battue par les codes SAE sur cette tâche, ou seulement
égalée (ce qu'un jury industriel demande en premier,
`docs/archive/audits/AUDIT_SAE_2026-08.md`) ? Les deux sondes sont évaluées sur des plis de CV
**identiques** (même `StratifiedKFold`, même ordre document) pour permettre un
McNemar apparié sur les prédictions hors-pli plutôt qu'une simple comparaison
de moyennes.

**Écart à la configuration de référence** : aucun calcul GPU — réutilise les
codes SAE déjà extraits par R0 (§97, `p1_all_doc_acts_ext_d1024.pt`) sur les
3300 mails originaux (hors augmentation). TF-IDF (`max_features=20000`,
`sublinear_tf`) refit à chaque pli sur le train du pli uniquement.

**Méthode statistique** : `paired_mcnemar_test` par intention (`b`=SAE
correct/TF-IDF faux, `c`=SAE faux/TF-IDF correct) ; `fdr_bh` sur les 5 tests.

**n** : 3300 mails originaux par intention (5 intentions testées).

**Résultat** (job 45972, h100, `results_v27_.../cache/intent_urgency_probe_results.json`) :

| Intention | Base. majoritaire | acc SAE | acc TF-IDF | McNemar $p$ (BH) |
|---|---|---|---|---|
| Réclamation (55,2\% pos.) | 55,2\% | 98,0\% | 97,6\% | 0,074 (0,092) |
| Résiliation (24,6\% pos.) | 75,4\% | 98,2\% | 96,0\% | $1{,}2\times10^{-13}$ ($2{,}9\times10^{-13}$) |
| Remboursement (10,3\% pos.) | 89,7\% | 94,8\% | 90,4\% | $1{,}4\times10^{-22}$ ($7{,}0\times10^{-22}$) |
| Information (60,8\% pos.) | 60,8\% | 87,9\% | 85,6\% | 0,00051 (0,00084) |
| Urgence (33,8\% pos.) | 66,2\% | 95,8\% | 95,8\% | 1,000 (1,000) |

**Conclusion** : les codes SAE **battent ou égalent** TF-IDF+LogReg sur les 5
intentions, jamais l'inverse (`mcnemar_b_sae_only` > `mcnemar_c_tfidf_only`
dans les 5 cas). L'écart est significatif après correction BH pour 3
intentions sur 5 (résiliation, remboursement, information) et non significatif
pour 2 (réclamation, urgence — TF-IDF quasi à égalité). C'est le résultat que
l'audit demandait explicitement de produire avant tout autre : la baseline
lexicale honnête ne dépasse les codes SAE sur aucune des 5 tâches testées, et
perd nettement sur les 3 intentions les moins fréquentes/les plus
structurées (résiliation, remboursement) — cohérent avec l'idée que le SAE
capture une structure sémantique qui dépasse la simple présence de
mots-clés, alors même que les labels eux-mêmes sont construits par
mots-clés (un avantage structurel pour TF-IDF que le SAE surmonte quand même).

**Limite connue** : les labels d'intention restent des labels FAIBLES (regex
sur mots-clés, pas une annotation humaine) — un désaccord SAE/TF-IDF peut
refléter un bruit d'étiquetage partagé par les deux sondes plutôt qu'un signal
réel. Corpus mails originaux uniquement (hors variantes augmentées, cohérent
avec `intent_urgency_probe.py` historique) ; un seul split train/test
(`CORPUS_SPLIT_SEED` fixe).

## 105. C1b — sanity check décodeur figé, init `cov` (schéma principal de Korznikov et al.)

**Question** : lecture de `pdf/sanitychecks.pdf` (Korznikov et al. 2026) après
C1 (§98) : leur baseline Frozen Decoder est testée sous deux schémas d'init,
`iso` (Gaussien isotrope normalisé, ce que C1 utilise) et `cov` (Gaussien de
covariance réelle, puis normalisé) --- et c'est **`cov`** qu'ils utilisent
pour tous leurs résultats publiés (Fig. 1, Tables 2-4, Annexe E) car
empiriquement plus difficile à battre (Explained Variance 0,570 contre 0,430
sur BatchTopK/Gemma-2-2B layer 12, à configuration comparable). C1 (`iso`)
sous-estime donc potentiellement la sévérité du sanity check tel que publié.
`cov_init` ajouté à `FrozenDecoderExtendedSAE`
(`src/sae/frozen_core.py::_reinit_decoder_cov` : tirage $\mathcal{N}(0,\Sigma)$,
$\Sigma$ = covariance empirique de $x$ sur 8192 échantillons, régularisation
ridge, puis normalisation --- PAS une PCA, les directions restent aléatoires,
seule leur distribution d'ensemble épouse la forme du nuage réel).

**Écart à la configuration de référence** : identique à C1 (§98) sauf
`SANITY_CHECK_FROZEN_DECODER_INIT=cov` au lieu du défaut `iso`.

**Méthode statistique** : `two_proportion_test` contre R0 (§97) et contre C1
(§98) ; comparaison de `dead_pct_extension` entre les deux schémas d'init
(métrique continue, pas de test de proportion nécessaire).

**n** : 150 prévu (borné par le nombre de directions actives sous décodeur
figé, cf. §98 --- peut être très inférieur, comme pour C1).

**Résultat** (job 45988, h100, COMPLETED,
`results_v42_ablation_c1b_sanity_frozen_decoder_cov_init/`, sidecar confirmant
`judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : `dead_pct_extension` = **97,0%** (contre 98,9%
pour C1/`iso`, 5,7% pour R0) --- `cov` laisse un peu plus de directions
vivantes (31 contre 11) mais reste massivement dégénéré. Sur les 31 :
**80,6% (25/31)**. Contre R0 (84,7%, 127/150) : $z=0{,}56$, $p=0{,}578$, non
significatif. Contre C1 (54,5%, 6/11) : $z=-1{,}69$, $p=0{,}091$, non
significatif (mais dans le sens attendu : `cov` moins dégradé que `iso`).

FVE (console, `_fve_pair`, domaine + hors domaine désormais disponibles pour
ce run) : domaine --- FVE(core)=0,7451, FVE(core+extension)=0,7536,
**ΔFVE=+0,0086** (R0 : +0,1393 ; C1/`iso` : +0,0001) ; hors domaine ---
FVE(core)=0,6879, FVE(core+extension)=0,6891, ΔFVE=+0,0012.

**Conclusion** : `cov` confirme être, comme chez Korznikov et al., une
baseline **plus dure à battre** que `iso` (ΔFVE 86x plus grand que C1,
dead_pct_extension 2 points plus bas) --- mais R0 la bat quand même
nettement : ΔFVE $+0{,}1393$ contre $+0{,}0086$, un facteur **~16**. Même sous
le schéma d'initialisation que Korznikov et al. utilisent pour leurs
résultats publiés (Fig. 1, Tables 2-4), le SAE entraîné produit une extension
qui explique ${\sim}16\times$ plus de variance supplémentaire qu'un décodeur
figé à une covariance réaliste mais non apprise. Le taux d'interprétabilité
brut (80,6\%, proche de R0) est à interpréter avec la même prudence que pour
C1 : $n=31$ reste sous-alimenté, et **c'est le $\Delta$FVE qui tranche ici**,
pas le taux jugé.

**Limite connue** : un seul seed ; comparaison non appariée ; $n=31$ toujours
sous-alimenté pour la comparaison de taux (IC large, non calculé faute de
gain d'information au-delà du $\Delta$FVE).

## 106. A3 — largeur du SAE core 65k sous méthodologie finale

**Question** : le balayage de largeur du core (§17 : 16k/65k/262k) n'a jamais
été mesuré sous méthodologie finale ni au layer de référence 31 (le 65k
disponible localement ne couvrait que layer 24). SAE GemmaScope-2
`layer_31_width_65k_l0_medium` téléchargé pour ce run. Contre R0 (§97, core
16k).

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`SAE_ID` (largeur du core figé ; `D_EXTRA`/`K_EXTRA` de l'extension
inchangés). **Ré-extraction complète requise** (pas de partage de cache
avec R0) : `compute_activation_cache_key` inclut `sae_id`, les activations
core max-poolées dépendent de la largeur du dictionnaire figé —
contrairement à ce que supposait la catégorisation initiale de cette
campagne (largeur core classée à tort "cache-partagé, ~1h" au même titre que
`K_EXTRA`/`D_EXTRA`/seed/`BATCH_SIZE_EXTRA`, qui eux n'entrent réellement pas
dans la clé).

**Méthode statistique** : `two_proportion_test` contre R0 (§97).

**n** : 150 features.

**Résultat** (job 46026, h100, COMPLETED,
`results_v46_ablation_a3_core_width65k_classic_setup_25m_layer31/`) :
**83,3% (125/150)**, `dead_pct_extension=5,1%`, `rho_sae=0,878`. Contre R0
(84,7%, 127/150) : $z=0{,}31$, $p=0{,}753$, non significatif.

**Conclusion** : quadrupler la largeur du core figé (16k→65k) ne produit
aucun écart mesurable — cohérent avec le fait que l'extension résiduelle
(D_EXTRA/K_EXTRA inchangés) fait tout le travail de spécialisation domaine,
la largeur du core ne conditionnant que sa couverture générique.

**Limite connue** : un seul seed ; ré-extraction complète (sae_id dans la
clé de cache), donc corpus vu par le modèle rigoureusement identique à R0
mais représentation core différente — pas un simple recalcul sur cache
partagé.

## 107. A4 — `BATCH_SIZE_EXTRA=16384` sous méthodologie finale

**Question** : le budget top-k de BatchTopK est partagé sur tout le batch
aplati (k·B pré-activations) — un batch 16x plus large (16384 contre 1024
par défaut) change le régime de parcimonie effectif par échantillon.
Trade-off `dead_pct`/`rho_sae` observé par l'audit externe mais jamais
tranché à pleine puissance statistique. Contre R0 (§97, `BATCH_SIZE_EXTRA=1024`).

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`BATCH_SIZE_EXTRA=16384` (hors clé de cache, extraction partagée avec R0).

**Méthode statistique** : `two_proportion_test` contre R0 (§97) ; comparaison
`dead_pct_extension`/`rho_sae` (`results.json`) pour trancher le trade-off.

**n** : 150 features.

**Résultat** (job 46027, h100, COMPLETED,
`results_v45_ablation_a4_batch_size_extra16384_classic_setup_25m_layer31/`) :
**89,3% (134/150)**, `dead_pct_extension=6,7%`, `rho_sae=0,898`. Contre R0
(84,7%, 127/150) : $z=-1{,}20$, $p=0{,}230$, non significatif.

**Conclusion** : le trade-off `dead_pct`/`rho_sae` suspecté par l'audit
externe ne se matérialise pas en écart d'interprétabilité : `dead_pct_extension`
(6,7% contre 5,7% pour R0) et `rho_sae` (0,898 contre 0,883) bougent à peine,
dans le sens d'une légère amélioration des deux. Batch 16x plus large,
aucun signal de dégradation.

**Limite connue** : un seul seed ; écart brut (+4,6 points) dans la
fourchette du plancher de bruit V1/V2 (§101).

## 108. A6 — `EPOCHS_EXTRA=40` (×4) sous méthodologie finale

**Question** : l'ancienne ablation "Époques ×4 (40/100)" (§synthese-ablations,
41,3\%, $z=0{,}70$, non significatif) ne testait que sous sélection par
magnitude et juge auto-référent. Contre R0 (§97, `EPOCHS_EXTRA=10`).

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`EPOCHS_EXTRA=40` (hors clé de cache, extraction partagée avec R0 —
entraînement 4x plus long, jugement inchangé).

**Méthode statistique** : `two_proportion_test` contre R0 (§97).

**n** : 150 features.

**Résultat** (job 46028, h100, COMPLETED,
`results_v47_ablation_a6_epochs40_classic_setup_25m_layer31/`) :
**88,0% (132/150)**, `dead_pct_extension=6,9%`, `rho_sae=0,888`. Contre R0
(84,7%, 127/150) : $z=-0{,}84$, $p=0{,}401$, non significatif.

**Conclusion** : quadrupler les époques (10→40) ne produit aucun écart
significatif, cohérent avec §97 (dead_frac/loss déjà stables dès l'epoch 2
sur R0, cf. l'analyse de convergence de C1/R0) — le SAE est déjà convergé
à 10 époques, 4x plus d'entraînement n'apporte rien de mesurable.

**Limite connue** : un seul seed ; écart brut (+3,3 points) dans la
fourchette du plancher de bruit V1/V2 (§101).

**Limite connue** : *à compléter.*

## 109. L1 — layer 24 sous méthodologie finale (clôture partielle du sweep layer)

**Question** : layer 24 était le layer par défaut historique du dépôt
(couverture Neuronpedia) avant l'adoption de layer 31 comme référence
(§51, écart significatif sous magnitude+auto-jugement, jamais répliqué
sous stratifié). Contre R0 (§97, layer 31), sous méthodologie totalement
homogène. Complète, avec layer 12 (§96, 88,7%) et layer 41 (rerun en
cours, job 45973 -- l'ancien résultat §90 datait d'un cache incompatible,
cf. docs/archive/audits/AUDIT_SAE_2026-08.md), le sweep layer à K5/25M sous Qwen.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`LAYER=24`, `SAE_ID=layer_24_width_16k_l0_medium`.

**Méthode statistique** : `two_proportion_test` contre R0 (§97).

**n** : 150 features.

**Résultat** (job 45896, h100, COMPLETED,
`results_v34_ablation_classic_setup_k5_25m_layer24/`, sidecar confirmant
`judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : **80,7% (121/150)**. Contre R0 (84,7%, 127/150) :
$z=0{,}92$, $p=0{,}360$, non significatif.

**Conclusion** : cohérent avec §88 (layer 41 vs 31, magnitude) et §96
(layer 12 vs 31, méthodologie complète) -- aucun des trois layers
alternatifs testés (12, 24, 41) ne se distingue significativement de layer
31 une fois la sélection stratifiée et le juge Qwen appliqués aux deux
bras. Le seul écart individuel jamais significatif du dépôt (§51, layer 31
vs 24 sous magnitude+auto-jugement) ne réplique donc toujours pas sous
méthodologie corrigée, cette fois avec les DEUX layers directement
comparés à la config de référence retenue (pas seulement l'un des deux).

**Limite connue** : un seul seed ; comparaison non appariée ; layer 41 sous
méthodologie finale encore en cours (job 45973) pour clore complètement le
sweep.

## 110. M1 — sélection par magnitude, juge auto-référent, à la configuration de R0 (25M tokens)

**Question** : la comparaison stratifié/magnitude et Gemma/Qwen ne
disposait jusqu'ici de tous ses points qu'à 500k tokens
(`results_v10_emails_main`, K$_\text{extra}=32$, layer 24, §correction-methode)
--- un volume sous le plancher de crédibilité retenu pour ce rapport (25M).
À la configuration de R0 (K$_\text{extra}=5$, layer 31, 25M tokens), on
dispose déjà de stratifié+Gemma (82,0\%, §82) et stratifié+Qwen (84,7\%,
R0/§97) ; ce run ajoute magnitude+Gemma pour compléter la grille méthode
$\times$ juge à un volume de tokens crédible, sans dépendre du run 500k.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`FEATURE_SELECTION_METHOD=magnitude` et `JUDGE_MODEL_ID=gemma-3-12b-it`
(auto-référent, ablation ponctuelle documentée -- pas un changement du
comportement par défaut, qui reste juge découplé de `MODEL_ID`).

**Méthode statistique** : `two_proportion_test` contre §82 (stratifié+Gemma,
même config, même juge) et contre R0 (stratifié+Qwen).

**n** : 150 features.

**Résultat** (job 46067, h100, COMPLETED,
`results_v48_ablation_m1_magnitude_gemma_classic_setup_25m_layer31/`) :
**89,3% (134/150)** --- plus haut que stratifié+Gemma (§82, 82,0%,
$z=-1{,}81$, $p=0{,}070$, tendanciel non significatif) et que R0
stratifié+Qwen (84,7%, $z=-1{,}20$, $p=0{,}229$, non significatif).

**Conclusion — À TRAITER AVEC PRUDENCE, cf. §115.** Ce résultat inverse le
sens de l'effet de sélection observé à 500k tokens (magnitude nettement
*sous* stratifié, §correction-methode) : ici magnitude+Gemma est
numériquement le taux le plus haut de toute la campagne R0. Avant d'écrire
quoi que ce soit dans le rapport à partir de ce chiffre, noter que §115
documente une contamination de `neg_example` (contexte négatif trivial,
souvent 1-2 mots, corrélée à la densité d'activation de la feature) qui
n'a pas encore été mesurée spécifiquement sur cette sélection par magnitude
(features plus denses par construction, §82 B.2 -- le pool de négatifs
`f_acts <= threshold_neg` pourrait se comporter différemment). **Ne pas
conclure "magnitude bat stratifié à 25M" avant d'avoir vérifié que ce run
n'est pas plus contaminé que R0, ou de l'avoir rejugé avec le correctif de
§115.**

**Limite connue** : un seul seed ; complète 3 cellules sur 4 de la grille
méthode$\times$juge à 25M tokens (magnitude+Qwen non mesuré) ; résultat
non encore validé sous le correctif de négatif de §115.

## 112. S2 — 27B sous méthodologie pleinement corrigée (clôture de la courbe d'échelle)

**Question** : dernier point de la courbe 1B/4B/12B/27B sous méthodologie
totalement homogène (§97 le déclarait décisif). Job initial (45895) échoué
après 7h37 (fuite mémoire GPU, `_decoder_stack` non libéré avant le
chargement du juge — corrigé dans `saev5.py`, cf. commit) ; relancé sur
h100-bis (job 46022) une fois le correctif appliqué.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`MODEL_ID=gemma-3-27b-it`, `MODEL_SIZE=27b`, `LAYER=40`, GemmaScope
`gemma-scope-2-27b-it` dédié.

**Méthode statistique** : `two_proportion_test`/`cochran_armitage_trend_test`
sur les quatre points 1B/4B/12B/27B.

**n** : 150 features.

**Résultat** (job 46022, h100-bis, COMPLETED,
`results_v31_ablation_classic_setup_k5_25m_model_scale_27b/`, sidecar
confirmant `judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : **85,3% (128/150)**. Contre R0/12B (84,7%,
127/150) : $z=-0{,}16$, $p=0{,}872$, non significatif.

**Conclusion** : **la courbe d'échelle sous méthodologie totalement
homogène est maintenant complète : 1B 80,0% / 4B 81,3% / 12B 84,7% / 27B
85,3%.** Le palier 12B→27B confirme le plateau déjà observé sous
magnitude+Gemma (§82, 82,0%→83,3%) : doubler encore la taille du modèle
au-delà de 12B n'apporte aucun gain mesurable. Combiné à §97 (Cochran-Armitage
1B/4B/12B, $p=0{,}293$), l'issue que §97 présentait comme ouverte est
tranchée : **aucun effet d'échelle significatif ne survit au retrait des
biais de sélection/juge, sur les quatre tailles testées.** Ce résultat
lui-même reste soumis à la même réserve que tout le reste de la campagne
(§115) : les quatre points partagent la même construction de `neg_example`.

**Limite connue** : un seul seed par palier ; layer confondu avec la taille
du modèle (chaque palier utilise son propre layer "~2/3 profondeur", jamais
isolé, cf. §82) ; MDE à $n=150$ non chiffré ici (voir audit externe reçu
28/08, à traiter).

## 113. Layer 41 — rejugé sous méthodologie finale (clôture du sweep layer)

**Question** : l'ancien §90 (78,7%) reposait sur un cache de fragments
incompatible avec le schéma actuel (`KeyError: 'raw_acts'`, répertoire
`results_v33` purgé et relancé proprement). Complète le sweep layer à
K5/25M sous Qwen : 12 (§96, 88,7%), 24 (§109, 80,7%), 31 (R0, 84,7%), 41.

**Écart à la configuration de référence** : identique à R0 (§97) sauf
`LAYER=41`, `SAE_ID=layer_41_width_16k_l0_medium`.

**Méthode statistique** : `two_proportion_test` contre R0 (§97).

**n** : 150 features.

**Résultat** (job 45973, h100, COMPLETED,
`results_v33_ablation_classic_setup_k5_25m_layer41/`, sidecar confirmant
`judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : **81,3% (122/150)**. Contre R0 (84,7%, 127/150) :
$z=0{,}77$, $p=0{,}442$, non significatif.

**Conclusion** : le sweep layer est maintenant complet sous méthodologie
finale (12/24/31/41, tous non significatifs contre 31) — le seul écart
individuel jamais significatif du dépôt (§51, layer 31 vs 24 sous
magnitude+auto-jugement) ne réplique sur aucun layer testé une fois la
sélection stratifiée et le juge Qwen appliqués partout.

**Limite connue** : un seul seed ; comparaison non appariée.

## 114. Volume 50M tokens à 1B (interaction volume × taille de modèle)

**Question** : R0/S1/S2 (25M tokens) ne montrent aucun effet de volume ni
d'échelle détecté ; teste si un petit modèle (1B) bénéficie davantage de
plus de tokens que le 12B (§23, aucun effet de volume détecté pour ce
dernier). Contre le point 1B/25M (§95, 80,0%).

**Écart à la configuration de référence** : identique au point 1B de §95
sauf `N_TOKENS_EXTRA_TRAIN=50000000` (au lieu de 25M) --
**ré-extraction complète** (le budget de tokens fait partie de la clé de
cache).

**Méthode statistique** : `two_proportion_test` contre le point 1B/25M
(§95).

**n** : 150 features.

**Résultat** (job 46024, h100-bis, COMPLETED,
`results_v43_ablation_volume50m_model_scale_1b/`, sidecar confirmant
`judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : **76,0% (114/150)**. Contre 1B/25M (80,0%,
120/150) : $z=0{,}84$, $p=0{,}403$, non significatif.

**Conclusion** : doubler le volume de tokens à 1B ne produit aucun gain
mesurable (76,0% contre 80,0%, écart dans le sens inverse de l'attendu
mais non significatif, dans la fourchette du plancher de bruit V1/V2).
Aucune interaction volume×taille détectée à cette puissance. Le point
50M/270M n'a pas pu être mesuré (échec de chargement du tokenizer,
`transformers`/format de checkpoint incompatible avec la version installée
-- rabbit hole non poursuivi faute de temps).

**Limite connue** : un seul seed ; puissance limitée ($n=150$, MDE ~13
points) pour détecter un effet d'interaction probablement plus petit que
l'effet principal déjà non détecté.

## 115. ⚠️ Contamination critique du négatif odd-one-out : le juge peut résoudre la tâche par longueur de contexte, pas par concept

**Question** (soulevée par inspection manuelle, confirmée empiriquement) :
le protocole odd-one-out présente 9 exemples positifs (contexte gauche
jusqu'à 60 tokens, `extract_causal_context`) et 1 négatif censé ne pas
partager le concept. Le négatif est choisi par magnitude d'activation la
plus faible (`build_feature_examples_with_control`), le contexte affiché
étant centré sur l'argmax d'activation de la feature sur ce document —
un argmax de bruit pur pour un document qui n'active pas la feature. Rien
dans le protocole ne garantit que ce contexte soit visuellement comparable
aux positifs.

**Mesure** (R0, `results_v27_.../cache/p1_judge_labels_extended.json`,
150 features ; A2/`results_v41_...`, même mesure) :

| Statistique | R0 | A2 |
|---|---|---|
| `neg_example` à $\le$ 3 mots | 150/150 (100%) | 150/150 (100%) |
| `neg_example` = `<<,>>` exactement | 60/150 (40%) | 62/150 (41%) |
| `neg_example` = `<<,\n\nJe>>` exactement | 34/150 (23%) | 26/150 (17%) |
| `neg_example` distincts sur 150 | 43 | 48 |
| features où le négatif est plus court que LES 9 positifs | 147/150 (98%) | — |
| longueur moyenne positifs vs négatif (mots) | 32,9 vs 1,4 | — |

**Méthode statistique** : aucune (mesure descriptive directe sur le champ
`neg_example`, pas de test d'hypothèse nécessaire face à un écart de cette
taille).

**Résultat** : sur R0, **147 features sur 150 (98%)** ont un négatif
strictement plus court que chacun de leurs 9 positifs — souvent un unique
mot de salutation ou de ponctuation (`<<,>>`, `<<Bonjour,>>`,
`<<,\n\nJe>>`) sans aucun contexte gauche. Root cause tracée dans
`src/sae/judge.py::build_feature_examples_with_control` : le négatif est
choisi par magnitude d'activation minimale sur un pool de documents
quasi-inactivants, mais le contexte affiché reste centré sur l'argmax
*de cette feature sur ce document* — un signal de bruit pur pour un
document qui n'active pas la feature, qui atterrit de façon disproportionnée
près du début du document une fois les tokens spéciaux/sink masqués
(`valid_token_mask`/`norm_outlier_mask`, `saev5.py:1240-1244`, correctement
appliqués **avant** l'écriture des fragments — ce n'est donc pas une fuite
de token BOS/sink brut, mais un effet résiduel de position sur les
tokens juste après le masquage, plausible mais non pleinement isolé ici).

**Conclusion** : **un juge (LLM ou humain) peut résoudre l'écrasante
majorité des 150 tâches odd-one-out de ce dépôt en choisissant simplement
le stimulus le plus court/le plus pauvre en contexte, sans lire le concept
partagé par les 9 autres.** Ceci ne prouve pas que le taux d'interprétabilité
mesuré (84,7% pour R0) est entièrement un artefact — les positifs doivent
toujours partager un concept réel pour que le juge produise un label
cohérent, et le taux d'interprétabilité repose aussi sur la qualité du
label généré, pas seulement sur l'identification correcte de l'intrus —
mais **la composante "identification de l'intrus" du score, telle que
construite, ne teste pas ce qu'elle prétend tester** pour au moins 98% des
features. Ce biais est probablement **présent de façon comparable dans
TOUTES les mesures de ce dépôt utilisant `odd_one_out_judge`** (R0, S1,
S2, C1, C1b, V1, V2, A1-A6, L1, M1, layer 41, 50M/1B, et tout l'historique
antérieur) puisqu'aucune n'a jamais modifié cette construction du négatif.

**Correctif appliqué** (`src/sae/judge.py`, B.6) : le négatif est
maintenant retenu en priorité s'il offre un contexte gauche d'au moins
`MIN_NEG_CONTEXT_TOKENS=20` tokens (pool de candidats élargi de 20 à 40
pour donner à cette contrainte une chance réelle d'aboutir), avec repli
explicite et loggé (`neg_context_truncated`, message `[WARN]`) sur
l'ancien critère si aucun candidat du pool ne qualifie. R0 relancé avec ce
correctif (job 46151, cache SAE/extraction réutilisé, seul le jugement est
refait) pour mesurer l'ampleur réelle de la contamination sur le chiffre de
référence du rapport avant toute décision de rejugement plus large.

**Limite connue** : le correctif change le **choix** du négatif (candidat
avec contexte suffisant) mais pas l'argmax lui-même — si le bruit de
position affecte aussi les candidats à contexte riche de façon plus subtile,
une partie de l'effet pourrait persister sous une forme moins visible.
Root cause du biais positionnel (pourquoi l'argmax de bruit atterrit-il
préférentiellement près du début du document même après masquage
BOS/sink ?) non pleinement élucidée — hypothèse de travail : effet
résiduel d'"attention sink" sur les tokens 2-4 (juste après le token masqué),
non couvert par `sigma_clip=4.0`. À creuser si le temps le permet.

## 116. Rejugement R0 sous le correctif B.6 (§115) : le garde-fou de longueur n'engage quasiment jamais

**Question** : le correctif B.6 (`MIN_NEG_CONTEXT_TOKENS=20`, pool élargi à
`_NEG_POOL_SCAN=40`) change-t-il le taux d'interprétabilité de référence
(R0, 84,7%) et résout-il la contamination décrite en §115 ?

**Écart à la configuration de référence** : aucun — même SAE entraîné
(`results_v27_ablation_classic_setup_k5_25m_layer31/`, cache d'extraction et
checkpoint réutilisés), seul le jugement est refait (job 46151).

**Méthode statistique** : test z à deux proportions appariées sur le même
jeu de 150 features (`two_proportion_test`, `src/analysis/stats.py`) entre
`p1_judge_labels_extended.json` avant (`.bak_pre_neg_context_fix`) et après
correctif ; mesure descriptive répliquant exactement le tableau de §115 sur
le fichier post-correctif.

**Résultat** :

| Statistique | R0 pré-correctif (§115) | R0 post-correctif (job 46151) |
|---|---|---|
| `interp_score` | 127/150 = 84,7% | 127/150 = 84,7% |
| features où `interp_score` change de valeur | — | 0/150 |
| `[WARN] neg_context_truncated=True` (repli déclenché) | — | **148/150 (98,7%)** |
| `neg_example` à $\le$ 3 mots | 150/150 (100%) | 148/150 (98,7%) |
| `neg_example` = `<<,>>` exactement | 60/150 (40%) | 59/150 (39%) |
| `neg_example` distincts sur 150 | 43 | 45 |
| features où le négatif est plus court que LES 9 positifs | 147/150 (98%) | 145/150 (96,7%) |
| `neg_example` textuellement différent du pré-correctif | — | 2/150 |

`two_proportion_test(127, 150, 127, 150)` : diff = 0,0 point, z = 0,0,
p = 1,0.

**Conclusion — le correctif B.6 n'a quasiment pas engagé.** Sur les 150
features de R0, le garde-fou `MIN_NEG_CONTEXT_TOKENS=20` ne trouve un
candidat qualifiant que pour 2 features (16604, 16733 — négatifs
authentiquement riches en contexte après correctif) ; pour les 148 autres,
aucun des 40 candidats du pool négatif n'atteint 20 tokens de contexte
gauche, et le code replie explicitement sur l'ancien critère (magnitude
seule) — le négatif produit est alors, textuellement, **identique** à avant
correctif. L'identité du taux d'interprétabilité (84,7% des deux côtés,
0 feature dont le score change) n'est donc PAS une preuve que le biais de
construction du négatif est bénin : c'est la conséquence mécanique du fait
que le correctif change presque partout... rien. Ceci confirme empiriquement
l'hypothèse de travail de §115 (biais positionnel systématique de l'argmax
de bruit, pas un phénomène rare) : si le contexte pauvre n'affectait qu'une
poignée de documents du pool négatif, élargir le balayage de 20 à 40
candidats aurait dû en trouver un correct pour l'écrasante majorité des
features — ce n'est pas le cas, ce qui indique que la quasi-totalité du pool
de documents quasi-non-activants pour une feature donnée produit un argmax
de bruit proche du début du document, pas seulement une minorité de "cas
difficiles" contournables par un plus grand pool.

**Limite connue** : un correctif qui résout réellement le biais devrait
découpler le contexte affiché pour le négatif de l'argmax de la feature
cible sur ce document (bruit par construction sur un document non-activant)
— par exemple un segment à position fixe ou aléatoire du document négatif,
comme c'est implicitement le cas pour les positifs (dont le contexte est
centré sur un argmax réellement informatif). Ce changement de conception
n'a pas été implémenté ni mesuré ici ; la contamination documentée en §115
reste donc essentiellement intacte sur R0 et, par extension plausible, sur
tout run antérieur utilisant `odd_one_out_judge` sans ce correctif plus
profond (§115 pour la liste : S1, S2, C1, C1b, V1, V2, A1-A6, L1, M1, layer
41, 50M/1B).

## 117. Correctif profond : position aléatoire au lieu d'argmax pour un négatif au signal nul

**Question** : §116 identifie précisément la cause mécanique du non-effet
de B.6 --- `np.argmax` sur un vecteur d'activation EXACTEMENT nul (le cas
normal pour un candidat du pool négatif : BatchTopK, la feature est
hard-zero hors de son top-k) renvoie déterministiquement l'index 0
(convention numpy sur les ex-aequo), pas une position de bruit distribuée.
Vérifié directement sur les fragments (feature 16511, 200/200 documents du
pool négatif à activation exactement nulle partout, `argmax=0` à chaque
fois) : ce n'est pas un biais positionnel résiduel post-masquage BOS/sink,
c'est un artefact déterministe de `argmax` sur un vecteur constant. Élargir
le pool balayé (B.6, 20→40) ne peut structurellement rien changer : tous
les candidats produisent la même position 0. §116 recommande explicitement
de découpler le contexte affiché de l'argmax quand le signal est nul --- ce
correctif l'implémente.

**Écart à la configuration de référence** : aucun changement de config,
correctif de code uniquement (`src/sae/judge.py::build_feature_examples_with_control`,
commentaire B.3 révisé). Quand `candidate_magnitude <= threshold_pos` (signal
nul, le cas normal pour neg_pool), la position à surligner n'a plus de sens
mécanique --- tirage aléatoire déterministe (graine `(f_idx, d_idx)`,
replay-stable comme l'exige le protocole) d'un début de mot avec au moins
`MIN_NEG_CONTEXT_TOKENS=20` tokens de contexte gauche disponibles (repli sur
n'importe quel début de mot si aucun ne qualifie, pour les documents
authentiquement courts). Quand un candidat porte un signal réel non nul
(rare dans neg_pool mais possible, cf. B.5), l'argmax reste utilisé --- il
est alors légitimement informatif.

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`) sur
le même jeu de 150 features, entre le pré-correctif de §115
(`.bak_pre_neg_context_fix`) et le rejugement sous ce correctif (job 46152
a échoué sur une `TypeError`, cf. commit `4dc198c` ; relancé job 46153,
`p1_judge_labels_extended.json`) ; mesure descriptive répliquant le tableau
de §115/§116 sur le fichier post-correctif.

**n** : 150 features.

**Résultat** (job 46153, COMPLETED, 1h32) :

| Statistique | §115 pré-correctif | §116 (B.6) | §117 (correctif profond) |
|---|---|---|---|
| `interp_score` | 127/150 = 84,7% | 127/150 = 84,7% | **94/150 = 62,7%** |
| `neg_example` à $\le$ 3 mots | 150/150 (100%) | 148/150 (98,7%) | **0/150 (0%)** |
| `neg_example` = `<<,>>` exactement | 60/150 (40%) | 59/150 (39%) | **0/150 (0%)** |
| `neg_example` distincts sur 150 | 43 | 45 | **150 (tous distincts)** |
| négatif plus court que LES 9 positifs | 147/150 (98%) | 145/150 (96,7%) | **25/150 (16,7%)** |

`two_proportion_test(127, 150, 94, 150)` (pré-correctif vs. correctif
profond, et B.6 vs. correctif profond — identique, B.6 n'avait rien changé
au fond) : diff = **−22,0 points**, z = 4,33, **p = 1,5×10⁻⁵**, h de Cohen =
0,51 (effet moyen-large). 45/150 features changent de score
(`interp_score`) entre le pré-correctif et le correctif profond.

**Conclusion — le correctif profond engage réellement (0/150 négatifs
courts, contre 148/150 sous B.6) et fait chuter le taux d'interprétabilité
de référence de 84,7% à 62,7%, un écart massif et hautement significatif.**
Contrairement à B.6 (§116), qui ne changeait presque rien parce que le
garde-fou de longueur ne pouvait jamais être satisfait par un argmax
dégénéré, ce correctif change la construction du négatif pour la quasi-
totalité des features et fait donc un test réel de l'hypothèse de §115 :
une fraction substantielle du taux de 84,7% mesurait la capacité du juge à
repérer un négatif trivialement pauvre en contexte, pas un concept partagé
absent. **84,7% cesse d'être défendable comme taux d'interprétabilité de
référence de ce rapport** ; 62,7% (job 46153, correctif profond, même SAE
entraîné que R0) en est la meilleure estimation actuelle sous ce protocole
corrigé. Cet écart dépasse de loin le MDE de 9,6 points (n=150, puissance
80%, `src/analysis/stats.py::minimum_detectable_effect`) et le plancher de
bruit empirique de 8 points (V1 vs V2) : ce n'est pas dans le bruit
méthodologique déjà caractérisé ailleurs dans ce dépôt.

**Limite connue** : mesuré uniquement sur R0 (12B, layer 31, $K_\text{extra}=5$,
25M tokens) à ce stade. Par construction (§115 : aucune mesure historique de
ce dépôt n'a jamais découplé le contexte du négatif de l'argmax de la
feature cible), le même écart est attendu, mais non mesuré, sur S1, S2, C1,
C1b, V1, V2, A1-A6, L1, M1, layer 41, 50M/1B — toute comparaison relative
entre ces runs (ablations d'hyperparamètres, sweep de layer, sweep d'échelle
du modèle) reste valide en tant que comparaison relative sous un protocole
partagé (mêmes deux biais des deux côtés de chaque ablation), mais leurs
valeurs absolues de taux d'interprétabilité surestiment très probablement le
taux réel dans la même proportion que R0. Le sweep d'échelle du modèle
(1B/4B/12B/27B, `subsec:model-scale`) est le plus exposé : ses points
extrêmes (1B à 80,0%, §95 ; 27B à 85,3%, §112) sont mesurés sous l'ancien
protocole non corrigé et ne peuvent pas être comparés en valeur absolue au
62,7% de R0 sous le protocole corrigé sans rejugement équivalent des autres
échelles.

## 118. Pipeline 2 : le biais de négatif de §115 ne s'y réplique pas

**Question** : `build_phrase_examples_with_control` (Pipeline 2) partage-t-il
le biais de construction du négatif identifié pour Pipeline 1 (§115) ? La
construction est structurellement différente (pas d'argmax token-level ni de
fenêtre de contexte -- la phrase candidate entière est affichée), donc le
mécanisme précis ne s'y applique pas mécaniquement, mais un biais analogue
(phrases à faible activation systématiquement plus courtes/génériques) restait
non vérifié.

**Écart à la configuration de référence** : aucun calcul GPU -- réplique
`build_phrase_examples_with_control` en rechargeant les embeddings F2LLM-v2
déjà en cache et le checkpoint `PhraseLevelSAE` déjà entraîné
(`results_v10_emails_main/p2_sae_dim320_d8192_k16.pt`), sur les 150 features
exactement jugées pour le chiffre 58,7% de §93 (`scripts/p2_negative_length_audit.py`,
job 46190, `a100`, CPU-only, aucun modèle chargé).

**Méthode statistique** : descriptive (longueurs moyennes/médianes, taux de
négatifs plus courts que tous les positifs) + corrélation de Spearman entre
longueur du négatif et `interp_score`.

**n** : 150 features.

**Résultat** (`results_v10_emails_main/cache/p2_negative_length_audit.json`) :

| Statistique | Pipeline 1 pré-correctif (§115) | Pipeline 2 |
|---|---|---|
| longueur négatif (mots), moyenne | 1,4 | **16,8** |
| longueur positifs (mots), moyenne | 32,9 | **15,6** |
| négatif plus court que TOUS les positifs | 147/150 (98%) | **36/150 (24,0%)** |
| négatif $\le$ 3 mots | 150/150 (100%) | **3/150 (2,0%)** |
| négatifs distincts sur 150 | 43 | **149/150** |

Spearman(longueur du négatif, `interp_score`) = **0,069 (p=0,40)** :
aucune corrélation détectable entre la longueur du négatif affiché et le
succès du juge.

**Conclusion** : **le biais de §115 ne se réplique pas sur Pipeline 2.**
Les négatifs et les positifs ont des distributions de longueur
statistiquement indissociables (16,8 contre 15,6 mots en moyenne, contre un
facteur ${\sim}23\times$ pour Pipeline 1 avant correctif), le taux de
négatifs plus courts que tous les positifs est proche de ce qu'on
attendrait par simple variabilité (24,0%, pas 98%), et la longueur du
négatif ne prédit pas le succès du juge. Le chiffre 58,7% (§93) n'a donc pas
besoin d'être requalifié pour ce motif -- l'hypothèse d'un artefact
analogue, non testée au moment de la rédaction du rapport, est falsifiée.

**Limite connue** : ne couvre que le biais de LONGUEUR spécifiquement ; ne
teste pas d'autres formes possibles de "tell" (registre, ponctuation
caractéristique d'un négatif générique/passe-partout) que le juge pourrait
exploiter sans que la longueur seule les capture.

## 119. Campagne de rejugement n=300 sous le correctif profond : R0 final et clôture complète (ablations, layers, échelle de modèle)

**Question** : §117 mesure le correctif profond à $n=150$ sur R0 seul
(84,7%→62,7%). Cette campagne rejuge R0 et ses ablations à $n=300$ (précision
resserrée) sous ce même correctif, sur toutes les configurations de la
"campagne finale" (§97-114) -- cache d'extraction et checkpoints réutilisés,
seul le jugement est refait (jobs 46191-46207, `h100`/`h100-bis`, 17/17
terminés).

**Écart à la configuration de référence** : aucun -- rejugement à l'identique,
$N\_FEATURES\_TO\_LABEL=300$ au lieu de 150.

**Méthode statistique** : `two_proportion_test`/`proportion_with_ci`
(`src/analysis/stats.py`).

**n** : 300 par run (11/31 pour C1/C1b -- cf. limite structurelle du décodeur
figé aléatoire, §97 et ci-dessous).

**Résultat (campagne complète, 17/17 jobs terminés)** :

| Run | $n$ | Résultat | $z$/$p$ vs R0 | $p_\text{BH}$ |
|---|---|---|---|---|
| **R0** (job 46191, layer 31) | 300 | **65,7% (197/300)**, IC95% [60,1% ; 70,8%] | référence | --- |
| V1 seed123 (46193) | 300 | 63,7% (191/300) | $z=-0{,}51$, $p=0{,}608$ | 0,84 |
| V2 seed7 (46195) | 300 | 64,0% (192/300) | $z=-0{,}43$, $p=0{,}669$ | 0,84 |
| A1 $K_\text{extra}=32$ (46196) | 300 | **54,3% (163/300)** | $z=-2{,}83$, $p=0{,}0046$ | **0,044** |
| A2 $D_\text{extra}=2048$ (46197) | 300 | 67,0% (201/300) | $z=0{,}35$, $p=0{,}730$ | 0,84 |
| A3 core 65k (46199) | 300 | 59,0% (177/300) | $z=-1{,}69$, $p=0{,}092$ | 0,28 |
| A4 batch\_extra=16384 (46200) | 300 | 65,0% (195/300) | $z=-0{,}17$, $p=0{,}864$ | 0,86 |
| A6 epochs\_extra=40 (46201) | 300 | 64,7% (194/300) | $z=-0{,}26$, $p=0{,}797$ | 0,85 |
| L1 layer24 (46202) | 300 | 60,3% (181/300) | $z=-1{,}35$, $p=0{,}176$ | 0,44 |
| L41 layer41 (46204) | 300 | 61,7% (185/300) | $z=-1{,}02$, $p=0{,}308$ | 0,63 |
| layer12 (46207) | 300 | 74,0% (222/300) | $z=2{,}22$, $p=0{,}026$ | 0,098 |
| 1B (46206) | 300 | 63,3% (190/300) | $z=-0{,}60$, $p=0{,}550$ | 0,84 |
| 4B (46194) | 300 | **54,7% (164/300)** | $z=-2{,}75$, $p=0{,}0059$ | **0,044** |
| 27B (46203) | 300 | 55,7% (167/300) | $z=-2{,}51$, $p=0{,}012$ | 0,061 |
| C1 sanity iso (46192) | 11 | 72,7% (8/11) | $z=0{,}49$, $p=0{,}627$ | 0,84 |
| C1b sanity cov (46198) | 31 | 74,2% (23/31) | $z=0{,}96$, $p=0{,}338$ | 0,63 |
| V50M\_1B volume 50M @1B (46205) | 300 | 55,3% (166/300) | vs R0 : n.s. ; **vs 1B (63,3%) : $z=-1{,}99$, $p=0{,}046$** | non incluse dans la famille BH ci-dessus |

$p_\text{BH}$ : correction Benjamini-Hochberg (`fdr_bh`) sur les 15
comparaisons contre R0 (hors V50M\_1B, comparé à 1B, pas à R0). **Seuls A1
et 4B survivent** ($p_\text{BH}<0{,}05$) ; 27B et layer12, significatifs en
brut, ne survivent pas à la correction.

R0 à $n=300$ (65,7%) contre l'estimation à $n=150$ de §117 (62,7%) :
$z=-0{,}63$, $p=0{,}53$ -- statistiquement indiscernables, comme attendu
d'un même run rejugé à une taille d'échantillon différente. **65,7%
(197/300) devient la valeur de référence définitive de ce rapport.**

**Résultat (sweep d'échelle du modèle, 1B/4B/12B/27B)** : Cochran-Armitage
sur les 4 points ordonnés (`cochran_armitage_trend_test`) : $z=-0{,}95$,
$p=0{,}34$ -- **aucune tendance monotone détectable.** Séquence observée :
63,3%→54,7%→65,7%→55,7%, un motif non ordonné (1B$\approx$R0 > 4B$\approx$27B),
pas la progression 80,0%→81,3%→84,7%→85,3% mesurée sous l'ancien protocole
(négatif biaisé, §113 zone). **L'effet d'échelle du modèle, présenté dans
les versions précédentes de ce rapport comme le résultat central du stage,
ne réplique pas sous protocole intégralement corrigé.** Seul un
hyperparamètre du SAE ($K_\text{extra}$, A1) reste significatif après
correction BH -- pas l'échelle du modèle extracteur/juge.

**Conclusion (C1/C1b)** : sous l'ancien protocole (négatif biaisé), le
décodeur figé aléatoire était mesuré significativement en dessous du SAE
entraîné sur le taux d'interprétabilité seul (29,3%, $n=150$ magnitude+
auto-jugement, $z=2{,}86$ vs 45,3%). Rejugées sous le correctif de négatif
et sous méthodologie stratifiée+Qwen, les features vivantes des deux sanity
checks (72,7% $n=11$ iso ; 74,2% $n=31$ cov) sont statistiquement
indiscernables de R0, et numériquement légèrement au-dessus. **Le taux
d'interprétabilité seul ne distingue plus le décodeur entraîné du décodeur
aléatoire à la puissance disponible sur ces échantillons (11/31 features)**
-- cohérent avec Korznikov et al. (les métriques proxy usuelles, dont
l'auto-interprétation, peuvent être trompées par un décodeur non entraîné).
Le $\Delta$FVE ($+0{,}1393$ R0 contre $+0{,}0001$/$+0{,}0086$) reste, lui,
inchangé par ce correctif (métrique continue, indépendante du juge et de la
construction du négatif) et demeure le test de falsification net de ce
sanity check.

**Limite connue** : $n=11$/$31$ reste structurellement faible (contrainte du
nombre de directions vivantes d'un décodeur non entraîné, pas un choix de
protocole) -- l'absence de significativité ne prouve pas l'absence d'écart,
seulement qu'il n'est pas détectable à cette puissance ; le $\Delta$FVE reste
la mesure porteuse de cette conclusion. Le sweep d'échelle (1B/4B/12B/27B)
n'a qu'une seule graine par point -- le motif non monotone observé
(1B$\approx$R0 > 4B$\approx$27B) n'a pas de mécanisme identifié ; une
réplication à seconde graine par échelle trancherait si ce motif est stable
ou un artefact de bruit sur un tirage unique. V50M\_1B (volume 50M à
l'échelle 1B) n'a été comparé qu'à 1B (25M, même échelle) et à R0 (12B,
échelle différente) -- pas de bras 50M à l'échelle 12B pour isoler
pleinement l'effet volume de l'effet échelle à ce point de la courbe.
M1 (magnitude+juge auto-référent, §110) n'a délibérément pas été inclus dans
cette campagne (comparaison intentionnellement non corrigée, cf. politique
juge Qwen-partout). Campagne complète (17/17 jobs), aucune section restante
à compléter pour ce rejugement.

## 120. Recouvrement de labels inter-graines sous R0 (audit de conformité du rapport) : plus faible que la mesure historique, pas plus élevé

**Question** : §21 mesure un recouvrement EXACT de labels de features
interprétables de 22/78 = 28,2% entre `SEED=42` et `SEED=123`, mais sous
sélection par magnitude, juge Gemma-3-12b-it auto-référent, négatif non
corrigé et n=150 (`results_v10_emails_main` vs
`results_v13_ablation_seed123`) — un protocole non R0, cité tel quel dans
le rapport de stage (§reproductibilite) comme s'il caractérisait
la stabilité des features sous R0. Ce même recouvrement, mesuré sous
méthodologie pleinement corrigée, donne-t-il un ordre de grandeur
comparable ?

**Écart à la configuration de référence** : aucun calcul GPU/LLM — les trois
graines R0 (`SEED=42`, R0 lui-même, §97/§119 ; `SEED=123`, V1, §99/§119 ;
`SEED=7`, V2, §101/§119) sont déjà entraînées et jugées à n=300 sous
stratifié + Qwen3.8-27B + négatif corrigé + déduplication par mail parent
— seule une comparaison d'ensembles de chaînes de caractères déjà en cache
est effectuée (`scripts/seed_label_overlap_r0_test.py`, réutilise la même
métrique que `feature_group_reproducibility_test.py::exact_label_overlap`,
qui avait servi à produire le chiffre de §21).

**Méthode statistique** : descriptive — indice de Jaccard
$|A \cap B| / |A \cup B|$ sur l'ensemble des libellés exacts (chaînes non
normalisées) des features avec `interp_score==1`, pour chaque paire de
graines. Aucun test de significativité formel : comme pour la mesure
historique de §21, il n'existe pas de témoin par permutation associé à
cette métrique (limite déjà présente en §21, non résolue ici).

**n** : 197 features interprétables (R0, seed 42), 191 (V1, seed 123), 192
(V2, seed 7), sur 300 features jugées par graine.

**Résultat** (`results_v27_ablation_classic_setup_k5_25m_layer31/cache/
seed_label_overlap_r0_results.json`) :

| Paire de graines | Labels distincts A | Labels distincts B | Intersection | Union | Jaccard |
|---|---|---|---|---|---|
| R0 (seed 42) vs V1 (seed 123) | 162 | 164 | 41 | 285 | **14,4%** |
| R0 (seed 42) vs V2 (seed 7) | 162 | 164 | 41 | 285 | **14,4%** |
| V1 (seed 123) vs V2 (seed 7) | 164 | 164 | 35 | 293 | **11,9%** |

**Conclusion** : sous méthodologie R0, le recouvrement exact de labels entre
graines (11,9% à 14,4% selon la paire) est **plus faible**, pas plus élevé,
que le chiffre historique de §21 (28,2%) — la conclusion qualitative
(instabilité individuelle des features malgré un taux agrégé stable, R0 =
65,7%/63,7%/64,0% pour les trois graines) **se confirme et se renforce**
sous protocole corrigé, elle ne s'inverse pas. La baisse du chiffre par
rapport à §21 est cohérente avec le passage à une sélection stratifiée
(qui échantillonne davantage les régions rares/périphériques du dictionnaire,
plus idiosyncrasiques d'une graine à l'autre) et à n=300 (population de
labels distincts mécaniquement plus grande, ce qui réduit un indice de
Jaccard à taux d'accord individuel égal) plutôt qu'un changement réel de
robustesse du SAE lui-même. **Le rapport de stage doit remplacer
28,2% par ce chiffre** (14,4%, R0 vs V1 — même paire de valeurs de seed que
la mesure historique) et préciser que la conclusion qualitative est
inchangée, pas seulement mettre à jour le nombre.

**Limite connue** : la métrique elle-même reste sensible à la sélection
stratifiée (deux graines différentes n'examinent pas nécessairement les
mêmes indices de feature, un artefact déjà noté pour la comparaison
inter-échelle) et à la longueur/normalisation du label (comparaison en
chaîne exacte, sans tolérance aux paraphrases proches — un jugement humain
pourrait considérer deux libellés synonymes comme "le même concept" alors
que cette métrique les compte comme distincts) ; ne mesure que le
recouvrement EXACT, pas un recouvrement sémantique (cf. §34/§66 pour le
volet groupe-à-groupe par similarité bge-m3, non repris ici — hors périmètre
de cet audit, cf. décision utilisateur).

## 121. Domaine vs volume sous R0 (audit de conformité du rapport) : l'effet domaine confirmé en §46 ne réplique pas sous protocole intégral

**Question** : §12/§46 comparent le corpus générique
(energy/sports/support) au corpus emails à effectif apparié (n=150 chacun)
mais sous sélection par magnitude, juge Gemma-3-12b-it auto-référent,
négatif non corrigé et hyperparamètres historiques
(`results_v9_full`/`results_v10_emails_main`, $K_\text{extra}=32$, layer 24,
500k tokens) — un protocole non R0, cité dans
le rapport de stage (§diagnostic-initial) comme la démonstration
que "le facteur qui a déterminé l'interprétabilité était la nature du
corpus d'entraînement, pas sa taille". Cette conclusion résiste-t-elle à la
méthodologie R0 intégrale ?

**Écart à la configuration de référence** : corpus principal = générique
(energy/sports/support, `CONFIRMATORY_DOMAIN_BASELINE=1`,
`src/sae/saev5.py`, comportement inchangé sinon) au lieu du corpus emails ;
sinon identique à R0 (§97) : sélection stratifiée, juge Qwen3.8-27B,
négatif corrigé, $K_\text{extra}=5$, $D_\text{extra}=1024$, 25M tokens,
layer 31, $n=300$. Pas de déduplication par mail parent pertinente ici
(`train_groups=None`, aucune notion de "mail d'origine" dans ce corpus) —
écart structurel assumé, pas un oubli. Contrairement à §46 (rejugement seul
sur un checkpoint déjà entraîné), aucun checkpoint sous ces hyperparamètres
n'existait pour le corpus générique — extraction et entraînement complets
requis (job 46946, h100, `results_v40_confirmatory_domain_baseline_r0_setup/`).

**Méthode statistique** : `two_proportion_test` (`src/analysis/stats.py`)
contre R0 (§97/§119, 65,7%, 197/300). Comparaison isolée, pas incluse dans
la famille de 15 comparaisons corrigée Benjamini-Hochberg de §119 (axe
différent : corpus, pas hyperparamètre SAE ou échelle de modèle) — le $p$
brut (0,60) est de toute façon si loin du seuil qu'une correction conjointe
ne changerait pas la conclusion qualitative.

**n** : 300 features jugées (corpus générique), comparé à R0 (300, corpus
emails).

**Résultat** (`results_v40_confirmatory_domain_baseline_r0_setup/cache/
p1_judge_labels_extended.json`, sidecar confirmant
`judge_model_id=Qwen3.8-27B`, `feature_selection_method=stratified`,
`doc_groups_dedup=true`) : **67,7% (203/300)**, IC95% Wilson
[62,2% ; 72,7%] pour le corpus générique, contre **65,7% (197/300)** pour
R0 (emails) — $z=0{,}52$, $p=0{,}60$, **non significatif**. Diagnostics de
fiabilité avant lecture du taux : `dead_pct_extension`=2,05% (sain,
comparable au 5,7% de R0), $\rho_\text{SAE}$=0,8389, FVE(core
seul)=0,7058, FVE(core+extension)=0,8067, $\Delta$FVE=+0,1009 (du même
ordre de grandeur que R0, +0,1393 — reconstruction saine, pas de signe de
sous-entraînement).

**Conclusion** : **l'effet domaine mesuré en §46 (45,3% vs 30,0%, écart de
15,3 points, $z=2{,}74$, $p=0{,}006$, significatif sous magnitude + juge
auto-référent) ne réplique pas du tout sous protocole R0** : le point
estimé s'inverse même légèrement (générique 67,7% $>$ emails 65,7%), sans
que cet écart soit lui-même significatif. **Le rapport de stage
doit corriger sa conclusion, pas seulement ses chiffres** : l'affirmation
"le facteur qui a déterminé l'interprétabilité était la nature du corpus
d'entraînement, pas sa taille" ne tient plus sous méthodologie corrigée —
ni le domaine ni le volume (déjà établi sans effet, Table~\ref{tab:synthese})
ne montrent d'effet détectable sur le taux d'interprétabilité une fois la
sélection stratifiée et le juge indépendant appliqués. Ce résultat rejoint
la liste déjà longue d'effets mesurés comme significatifs sous
magnitude + auto-jugement qui ne survivent pas à la correction complète du
protocole (effet d'échelle du modèle, §119 ; ancien biais de négatif,
§115-117) — un motif structurel du protocole historique plutôt qu'un
hasard isolé à cette comparaison précise.

**Limite connue** : un seul seed pour chaque bras (comme pour l'essentiel
de la campagne finale) ; le corpus générique (energy/sports/support,
Wikipédia/fineweb2 filtré par mots-clés) reste qualitativement très
différent du corpus emails augmenté (registre, longueur de document,
structure) — l'absence d'effet mesurable ici ne dit rien sur un domaine
plus proche des emails mais tout de même hors distribution ; deux domaines
aussi extrêmes que possible donnant un résultat statistiquement
indiscernable est en soi le résultat le plus fort disponible contre
l'hypothèse d'un effet domaine détectable à ce protocole et ce $n$. La
taille d'effet minimale détectable à $n=300$ par bras et à ce taux de base
reste ${\sim}10$ points (cf. §119) : un effet domaine réel mais inférieur à
ce seuil resterait indétectable ici, comme pour toute comparaison de cette
campagne.

## 122. ΔCE entraîné contre décodeur aléatoire sous R0 (audit de conformité du rapport) : la vraie comparaison que le texte prétendait déjà avoir, corrigée après un premier essai invalidé par un artefact de masquage

**Question** : les deux rapports décrivaient le $\Delta$CE de §58/§61 comme
une comparaison décodeur entraîné contre décodeur aléatoire, alors que cette
mesure compare en réalité core seul contre core+extension entraînée, sur le
checkpoint historique $K_\text{extra}=32$ (pas R0). Une fois le texte corrigé
pour refléter honnêtement ce qui a été mesuré (cf. le rapport de stage),
la vraie comparaison — décodeur entraîné (R0) contre décodeurs aléatoires
figés (C1/C1b), sous $K_\text{extra}=5$ — reste-t-elle à produire, et
confirme-t-elle la lecture qu'en donne le $\Delta$FVE (R0 largement
supérieur aux deux témoins aléatoires, §98/§105/§119) ?

**Écart à la configuration de référence** : aucun réentraînement — réutilise
trois checkpoints déjà entraînés sous $K_\text{extra}=5$/$D_\text{extra}=1024$/
layer 31 : R0 (décodeur entraîné, `results_v27.../p1_frozen_core_d1024_k5.pt`,
§97), C1 (décodeur aléatoire figé, init \emph{iso}, `results_v37.../p1_frozen_core_d1024_k5.pt`,
§98) et C1b (décodeur aléatoire figé, init \emph{cov}, `results_v42.../p1_frozen_core_d1024_k5.pt`,
§105) — même core GemmaScope gelé partagé entre les trois. $\Delta$CE
(`ce_loss_increase`, `src/sae/compare/crosslingual.py`) calculé sur les
mêmes 60 documents tenus à l'écart (test split, `CORPUS_SPLIT_SEED`) pour
les trois conditions.

**Premier essai (job 46994) invalidé** : le patch s'appliquait à TOUTES les
positions du document, y compris le premier token de contenu et les
outliers de norme intra-document (\og{}massive activations\fg{} de Gemma-3,
déjà documentées dans le rapport) — des positions sur lesquelles l'extension
n'a jamais été entraînée ni évaluée (`extract_residual_acts`, utilisé pour
tout l'entraînement et le $\Delta$FVE officiels, les exclut systématiquement
via `valid_token_mask`/`norm_outlier_mask`, `src/analysis/activations.py`).
Résultat de ce premier essai : $\Delta$CE(R0\_trained)=5,16, **supérieur**
aux deux témoins aléatoires (1,18 et 1,26) — l'inverse de ce que montre le
$\Delta$FVE. Vérifié empiriquement que ce n'est PAS un problème de classe de
chargement (`SAEBoostResidualSAE`/`FrozenDecoderExtendedSAE` partagent
exactement les mêmes paramètres ; `load_state_dict` confirme `missing=[]`,
`unexpected=[]` pour les trois checkpoints dans les deux essais). Diagnostic
retenu : un décodeur entraîné et calibré sur la distribution normale du
résidu produit une reconstruction largement fausse sur une position hors
distribution (BOS/outlier) qu'il n'a jamais vue, alors qu'un décodeur
aléatoire quasi mort (§98 : `dead_pct_extension`=98,9\% pour C1) reste
largement inoffensif faute de réagir à quoi que ce soit — un artefact de
protocole de mesure, pas une propriété réelle du SAE. Corrigé
(`scripts/ce_loss_r0_trained_vs_random_test.py`) : le patch ne s'applique
plus qu'aux positions valides (même masque que l'extraction officielle),
les positions exclues gardant leur activation d'origine.

**Méthode statistique** : test de Wilcoxon signed-rank apparié par document
(H1 : $\Delta$CE[aléatoire] $>$ $\Delta$CE[entraîné]), une comparaison pour
chaque témoin aléatoire (iso, cov).

**n** : 60 documents appariés (même échantillon pour les trois conditions).

**Résultat** (job 47197, h100, `results_v27_ablation_classic_setup_k5_25m_layer31/cache/
ce_loss_r0_trained_vs_random_results.json`, 0,9\% des tokens masqués en
moyenne par document, cohérent avec la rareté documentée des positions
outliers/BOS) :

| Condition | $\Delta$CE moyen | vs R0\_trained ($H_1$ : aléatoire $>$ entraîné) |
|---|---|---|
| R0\_trained | **0,301** | --- |
| C1\_random\_iso | 0,525 | $W=1785$, $p=7{,}5\times10^{-11}$, 56/60 documents en faveur du SAE entraîné |
| C1b\_random\_cov | 0,510 | $W=1769$, $p=1{,}6\times10^{-10}$, 54/60 documents en faveur du SAE entraîné |

**Conclusion** : une fois le patch restreint aux positions sur lesquelles
l'extension a réellement été entraînée/évaluée, le décodeur entraîné dégrade
nettement moins la cross-entropy que les deux témoins aléatoires figés
($\Delta$CE 0,301 contre 0,51-0,53), effet hautement significatif et
quasi unanime par document (56/60 et 54/60) — **cette fois, la lecture du
$\Delta$CE converge réellement avec celle du $\Delta$FVE** (R0 largement
supérieur aux témoins aléatoires, §98/§105/§119) : le rapport de stage
peut désormais citer un vrai $\Delta$CE entraîné-vs-aléatoire sous R0, plutôt
que la comparaison historique core-seul-vs-core+extension sous
$K_\text{extra}=32$ mal décrite dans les versions précédentes.

**Limite connue** : $n=60$ documents, un seul seed d'entraînement par
condition ; le masquage (`valid_token_mask`/`norm_outlier_mask`) réutilise
les fonctions de l'extraction officielle mais n'a pas été audité
indépendamment pour ce script précis (risque résiduel que le masque
lui-même diffère subtilement de celui utilisé à l'entraînement, ex. ordre
d'application par rapport au padding) ; les deux témoins aléatoires (iso,
cov) restent structurellement à faible capacité active (§98 : 11/1024 et
31/1024 directions vivantes) — le test porte sur l'effet agrégé au niveau
document, pas sur une décomposition par direction active/morte.

## 123. Robustesse au réordonnancement sous R0 (audit de conformité du rapport) : le protocole final est plus stable individuellement que la mesure historique, pas moins

**Question** : §13.1 mesure le taux de décision unanime sur 5
réordonnancements (30,7%) et le taux de bascule individuelle (31,3%) sur
`results_v10_emails_main` — sélection par magnitude, juge Gemma-3-12b-it
auto-référent rechargé comme juge (biais que CLAUDE.md interdit
explicitement de réintroduire), sans déduplication par mail parent. Le
même protocole, sous R0 intégral, donne-t-il un ordre de grandeur
comparable ?

**Écart à la configuration de référence** : aucun calcul GPU au-delà du
rejugement — réutilise les activations/fragments déjà en cache de R0.
Corrige trois écarts du script historique
(`scripts/judge_robustness_check_r0_test.py`, adapté de
`judge_robustness_check.py`) : juge = `JUDGE_MODEL_ID` (Qwen3.8-27B,
découplé de `MODEL_ID`) au lieu de Gemma-3-12b-it auto-référent ;
`doc_groups=train_groups` (déduplication par mail parent, comme l'appel
officiel `odd_one_out_judge` dans `saev5.py`) ; 300 features stratifiées de
R0 au lieu de 150 par magnitude. **Piège rencontré et corrigé en cours de
route** : un premier essai donnait 0% de vote majoritaire (100% de réponses
non parsables, `predicted=-1` sur toutes les répétitions) — cause
identifiée : les templates de chat Qwen3(.x) activent par défaut un
préambule `<think>...</think>` qui dépasse à lui seul `max_new_tokens=8`
sans jamais atteindre la réponse, un piège déjà documenté dans
`src/sae/judge.py::_batched_generate` (`enable_thinking=False`) mais absent
de l'appel `_apply_chat_and_extract` utilisé par ce script — corrigé en
ajoutant `enable_thinking=False`, script relancé.

**Méthode statistique** : `paired_mcnemar_test` (single-shot vs vote
majoritaire, mêmes 300 features).

**n** : 300 features (5 répétitions chacune).

**Résultat** (job 47362, h100-bis, `p1_judge_robustness_r0.json`) :

| Métrique | R0 (n=300) | Historique §13.1 (n=150) |
|---|---|---|
| Taux single-shot | 65,7% | 45,3% |
| Taux vote majoritaire | 63,7% | 48,7% |
| McNemar single-shot vs majoritaire | $\chi^2=0{,}43$, $p=0{,}51$ | $p=0{,}560$ |
| Décisions unanimes (5/5) | **53,3% (160/300)** | 30,7% (46/150) |
| Features changeant de statut | **19,3% (58/300)** | 31,3% (47/150) |

**Conclusion** : sous R0, le protocole reste globalement stable en agrégé
(65,7%→63,7%, non significatif, comme historiquement) mais est
**individuellement plus stable** que sous l'ancien protocole : 53,3%
de décisions unanimes contre 30,7%, et 19,3% de bascules individuelles
contre 31,3% — près de 12 points de moins. La conclusion qualitative de
§13.1 (le protocole odd-one-out à décision unique est bruyant au niveau
de la feature individuelle, un vote majoritaire est préférable en
production) **se confirme**, mais son ampleur exacte était surestimée sous
l'ancien protocole — cohérent avec le motif déjà observé ailleurs dans ce
rapport (sélection par magnitude + juge auto-référent amplifiant les
artefacts de mesure). Le rapport de stage doit citer ce chiffre
comme référence, pas celui de §13.1.

**Limite connue** : un seul seed de réordonnancement (5 répétitions, pas de
réplication de la campagne elle-même) ; comme pour §13.1, aucun témoin par
permutation n'existe pour cette métrique.

## 124. Biais multilingue du juge sous R0 (audit de conformité du rapport) : même lecture qualitative, bascule individuelle plus faible

**Question** : §22 mesure le biais multilingue (FR 46,9% vs EN 45,5%,
38,6% de bascule individuelle) sur `results_v10_emails_main`, avec les
mêmes écarts que §13.1 (magnitude, juge/traducteur Gemma-3-12b-it
auto-référent, sans déduplication). Réplique-t-il sous R0 ?

**Écart à la configuration de référence** : mêmes trois corrections que
§123 (`scripts/multilingual_judge_bias_test_r0.py`, adapté de
`multilingual_judge_bias_test.py`) : juge/traducteur = Qwen3.8-27B
découplé, `doc_groups=train_groups`, 300 features R0. **Même piège
`enable_thinking`** rencontré et corrigé (le premier essai est resté
bloqué ${\sim}2$h sans produire la moindre ligne de progression — généra-
tion emballée dans le préambule de raisonnement à chaque appel, jamais
arrivée à la réponse malgré un budget de tokens généreux pour la
traduction ; job annulé avant terme, aucun résultat de ce premier essai
n'a été utilisé).

**Méthode statistique** : `paired_mcnemar_test` (FR vs EN, mêmes features).

**n** : 296 features (4 échecs de traduction exclus, sur 300).

**Résultat** (job 47363, h100-bis, `multilingual_judge_bias_r0_results.json`) :

| Métrique | R0 (n=296) | Historique §22 (n=145) |
|---|---|---|
| Taux FR | 65,5% | 46,9% |
| Taux EN | 63,5% | 45,5% |
| McNemar FR vs EN | $\chi^2=0{,}31$, $p=0{,}58$ | $p=0{,}894$ |
| Features changeant de statut | **27,0% (80/296)** | 38,6% (56/145) |

**Conclusion** : même lecture que §22 — aucune différence agrégée
significative entre français et anglais traduit (non-significatif dans les
deux protocoles), mais un bruit de bascule individuelle substantiel. Comme
pour le réordonnancement (§123), ce bruit individuel est **plus faible**
sous R0 qu'historiquement (27,0% contre 38,6%) — le motif se confirme,
son ampleur était surestimée sous l'ancien protocole.
le rapport de stage doit citer ce chiffre comme référence.

**Limite connue** : la traduction reste faite par le même modèle que le
juge (Qwen, pas un traducteur neutre) ; 4/300 échecs de parsing JSON
exclus, non attribuables à une catégorie de features en particulier
(vérification rapide, pas d'analyse systématique du biais d'exclusion).

## 125. Clustering ciblé sous R0 (audit de conformité du rapport) : structure géométrique confirmée, accuracy qualitativement différente sur un jeu de features distinct

**Question** : §86 exerce la chaîne complète de clustering ciblé
(mots-clés LLM, union top-k, Jaccard, étiquetage, accuracy, z-conductance)
sur les 68 features interprétables de la campagne historique (sélection
par magnitude, juge auto-référent). Le résultat qualitatif (clusters plus
compacts que le hasard, accuracy hétérogène) tient-il sur les 197 features
interprétables de R0 ?

**Écart à la configuration de référence** : aucun changement de code —
`scripts/clustering_llm_test.py` était déjà paramétré par `SAVE_DIR` et
utilisait déjà `load_judge_model()` (Qwen3.8-27B, jamais `MODEL_ID`) ;
seul `SAVE_DIR` est repointé sur R0. Latents = 197 features interprétables
de R0 (`p1_top_extended_features.json`) au lieu des 68 historiques.

**Méthode statistique** : descriptive (accuracy et z-conductance par
cluster, témoin par permutation à 100 tirages déjà intégré au script).

**n** : 300 documents (même requête \og{}type de réclamation client\fg{},
4 clusters).

**Résultat** (job 47339, h100-bis, `clustering_llm_verification.json`) :

| Cluster | Taille | Description LLM | Accuracy | $z$-conductance |
|---|---|---|---|---|
| 0 | 41 | Aggressive complaints, power outages | 29,3% | −1,39 |
| 1 | 56 | Formal/mixed-language service+billing complaints | 64,3% | −5,09 |
| 2 | 111 | Standard formal power-cut complaints | 27,9% | −10,86 |
| 3 | 92 | Contract termination / new service requests | 90,2% | −15,94 |

**Conclusion** : les quatre $z$-conductances restent négatives (clusters
géométriquement plus compacts que des échantillons aléatoires) — la
structure géométrique **se confirme** sur le jeu de features R0, avec un
effet même plus marqué qu'historiquement ($z$ jusqu'à −15,94 contre −10,06
au plus négatif en §86). L'accuracy reste hétérogène (27,9% à 90,2%,
contre 5,6% à 61,7% historiquement) — toujours cohérent qualitativement
avec la littérature citée (accuracy par cluster plus variable qu'une
baseline dense), mais la répartition par cluster n'est pas directement
comparable : les mots-clés générés, la taille des clusters (41/56/111/92
contre 175/25/46/54) et le contenu même diffèrent, puisque le jeu de 197
features R0 n'est pas un sur-ensemble des 68 historiques (sélection
stratifiée contre magnitude). Le rapport de stage doit citer ce
résultat comme référence, en conservant la mise en garde déjà présente
(une seule requête testée, pas de baseline dense/instruction-tuned,
$n_\text{clusters}$ non sélectionné empiriquement).

**Limite connue** : identique à §86 (une seule requête, un seul seed, pas
de baseline dense) ; la comparaison cluster-à-cluster avec §86 n'a pas de
sens direct (jeux de features disjoints, tailles de cluster différentes) —
seule la propriété qualitative agrégée (compacité systématique, accuracy
hétérogène) est comparable entre les deux mesures.

## 126. Fidélité du steering sous R0 (audit de conformité du rapport) : hétérogénéité confirmée, catégorisation par intention non stable entre configurations

**Question** : §24/§68/§69 mesurent la fidélité du steering
(`steer_and_decode`) exclusivement sur le checkpoint historique
$K_\text{extra}=32$/layer 24. Réplique-t-elle sous R0
($K_\text{extra}=5$/layer 31) ?

**Écart à la configuration de référence** : `scripts/steering_fidelity_test_r0.py`,
adapté de `steering_fidelity_test.py` — seuls `D_EXTRA`/`K_EXTRA`/`SAE_ID`
(1024/5/`layer_31_width_16k_l0_medium`, au lieu de 1024/32/`layer_24_...`)
et `SAVE_DIR` changent ; le correctif `random_state=SEED` de §69 était déjà
en place. Aucun appel au juge LLM dans ce script (sonde logistique +
encode/decode du SAE seulement) — non affecté par le piège
`enable_thinking` de §123/§124.

**Méthode statistique** : descriptive (ratio round-trip/ablation en place,
comme §24/§68/§69).

**n** : jusqu'à 200 documents par intention (5 intentions).

**Résultat** (job 47360, h100-bis, `steering_fidelity_r0_results.json`) :

| Intention | Chute en place | Chute round-trip | Ratio | R0 | Historique (§69, K32/layer24) |
|---|---|---|---|---|---|
| Réclamation | 0,141 | 0,569 | **4,03×** | amplifié | 1,87× (amplifié) |
| Résiliation | 0,569 | 0,871 | **1,53×** | amplifié | 0,82× (partiel) |
| Remboursement | 0,998 | 0,019 | **0,02×** | neutralisé | 0,37× (partiel) |
| Information | 0,938 | 0,544 | **0,58×** | partiel | 0,99× (quasi préservé) |
| Urgence | 0,370 | 0,981 | **2,65×** | amplifié | 0,11× (neutralisé) |

**Conclusion** : la conclusion qualitative générale (comportement
hétérogène selon l'intention, pas de mécanisme causal uniforme, le steering
n'est pas un outil d'intervention fiable et prévisible) **se confirme**
sous R0. En revanche, **la catégorisation précise par intention change
substantiellement entre K32/layer24 et K5/layer31** — urgence passe de
fortement neutralisée (0,11×) à amplifiée (2,65×), information de quasi
préservée (0,99×) à partiellement préservée (0,58×), remboursement de
partiellement préservée (0,37×) à quasi neutralisée (0,02×) : comme pour
le correctif d'intention de §68/§69, les affirmations spécifiques par
intention ne sont pas stables d'une configuration à l'autre, seule la
conclusion générale (hétérogénéité, absence de mécanisme uniforme) l'est.
le rapport de stage doit citer les chiffres R0 comme référence
et ne plus présenter la répartition K32/layer24 comme le résultat final.

**Limite connue** : un seul seed de classifieur par intention (contrairement
à §69 qui en a confirmé deux pour K32/layer24) ; les vecteurs restent des
codes SAE poolés par MAX documentaire, pas des codes de token réels (même
limite qu'à l'origine, §24).
