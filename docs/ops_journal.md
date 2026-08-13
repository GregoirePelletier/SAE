# Journal d'infrastructure

Incidents d'ingénierie sans valeur informative résiduelle pour l'interprétation
des résultats (jobs SLURM, chemins, mémoire) — déplacés hors de
`RESULTS_TESTS.md` pour ne pas mélanger le suivi opérationnel avec le journal
d'expériences. Chaque section garde son numéro d'origine ; une ligne de renvoi
reste à sa place dans `RESULTS_TESTS.md`.

## 4. Bugs trouvés dans les 3 autres `.slurm` (jamais fonctionnels tels quels)

Trois scripts n'avaient jamais réussi à s'exécuter (`augmentation_38743.log`,
`test_massive_38845.log` : échec immédiat).

- **Commun** : `uv run python ...` tente de re-résoudre l'environnement contre
  le lockfile et de télécharger torch, timeout systématique sur un nœud sans
  accès Internet direct. Fix : `uv run python X` → `.venv/bin/python X` dans
  les 3 scripts.
- `slurm/validation/run_test_massive.slurm` : chemin de script faux (préfixe
  `scripts/` manquant) ; `LOCAL_SAE_DIR` par défaut pointait un répertoire
  vide au lieu du répertoire réel des poids. Job 38948 : COMPLETED après fix.
- `slurm/augmentation/run_augmentation.slurm` : même chemin faux ; `SAVE_DIR`
  non défini tombait sur un défaut incohérent avec le reste du pipeline. Job
  38949 : soumis.
- `slurm/baseline_diffing/run_baseline.slurm` : `scripts/baseline_gemmascope.py`
  attend 2 arguments positionnels jamais passés (`IndexError` garanti) ; même
  bug `LOCAL_SAE_DIR` vide ; `SAVE_DIR`/`CACHE_DIR` désalignés avec le script
  d'augmentation. Job 38950 : soumis avec dépendance sur 38949.

## 5. Budget de temps de l'augmentation complète

Job 38949 (augmentation complète, 45 240 générations visées, `--time=2:00:00`)
: TIMEOUT après 2h, 1440/45240 générations écrites (reprise automatique
possible, aucune perte). Au rythme observé (~12 générations/min), le corpus
complet demande ~63h GPU cumulées. Job 38950 (dépendant) annulé
(`DependencyNeverSatisfied`).

Validation retenue : sous-échantillonnage déterministe (`AUGMENT_SAMPLE_N=60`,
`random_state=SEED`) pour valider le pipeline baseline avant d'engager le
volume complet.

## 6. OOM CUDA dans `slurm/baseline_diffing/run_baseline.slurm`

Job 38988 (baseline sur échantillon test) : `CUDA out of memory`. Cause
initiale : `output_hidden_states=True` sans `logits_to_keep=1` calcule par
défaut les logits sur toute la séquence et tout le vocabulaire (Gemma-3 :
~262k tokens) alors que seul `hidden_states` est utilisé — passait inaperçu
sur un nœud à 85 Go VRAM, sature un A100 40 Go. Fix appliqué
(`logits_to_keep=1`) ; job 38999 toujours FAILED, OOM plus tardif.

Cause racine réelle : `extract_residual_acts` (`src/analysis/activations.py`)
était décorée `@torch.no_grad()` sur une fonction génératrice — ce décorateur
n'entoure que la création de l'objet générateur, pas les itérations réelles
faites ensuite via `next()`/`for`. L'autograd tournait donc actif sur tous les
forwards, retenant le graphe de calcul des ~48 couches sans qu'aucun
`.backward()` ne soit jamais appelé. Fix : `with torch.no_grad():` explicite
entourant le corps de la boucle. Job 39003 : COMPLETED (11min24s, contre 3
échecs OOM successifs).

**Observation retenue pour l'interprétation des résultats** (pas un incident
d'infrastructure) : sur le pipeline baseline validé par ce run, le feature le
plus discriminant par axe/niveau est très souvent un label Neuronpedia
générique/structurel ("Subject: followed by email subject lines", listes,
ponctuation) plutôt qu'un concept sémantique lié à l'axe de perturbation —
première observation du biais de formatage résiduel ("Objet :"/"Subject :")
mesuré et corrigé par la suite (`RESULTS_TESTS.md` §14.1).

## 7. Suivi des jobs (session d'audit initiale)

| Job ID | Script | But | Résultat |
|---|---|---|---|
| 38948 | `slurm/validation/run_test_massive.slurm` | Diagnostic massive activations | COMPLETED |
| 38949 | `slurm/augmentation/run_augmentation.slurm` (corpus complet) | Génère `augmented_mails.jsonl` | TIMEOUT (2h) — 1440/45240, reprise possible |
| 38950 | `slurm/baseline_diffing/run_baseline.slurm` (dépendait de 38949) | Baseline SAE natif | annulé (dépendance jamais satisfaite) |
| 38987 | `slurm/augmentation/run_augmentation.slurm` (test, 60 mails) | Génère `augmented_mails_test.jsonl` | COMPLETED (780/780, 694 acceptées) |
| 38988 | `slurm/baseline_diffing/run_baseline.slurm` (test) | Baseline SAE natif sur échantillon test | FAILED (OOM, `lm_head` sur tout le vocab) |
| 38999 | idem, retry 1 | avec fix `logits_to_keep=1` | FAILED (OOM plus tardif) |
| 39000 | idem, retry 2 | avec fix fragmentation (`expandable_segments`) | FAILED (pas la fragmentation) |
| 39003 | idem, retry 3 | avec le fix réel (`no_grad` sur générateur) | COMPLETED (11min24s) |

## 8. Correctifs de code de cette passe

- Appels `.venv/bin/python` au lieu de `uv run python` dans les 3 `.slurm`.
- Chemins de script corrigés (préfixe `scripts/` manquant).
- `LOCAL_SAE_DIR` explicite (le défaut était vide).
- Arguments positionnels manquants ajoutés à `run_baseline.slurm`.
- `scripts/run_augmentation.py` : ajout `AUGMENT_SAMPLE_N`/`AUGMENT_OUT_NAME`.
- `src/analysis/activations.py::extract_residual_acts` : `logits_to_keep=1` +
  `with torch.no_grad():` explicite sur générateur (cf. §6).

## 9. Suivi (session d'audit initiale)

Pipeline baseline validé de bout en bout sur échantillon. Décisions prises par
la suite : augmentation complète lancée en plusieurs jobs successifs (reprise
automatique via skip des `aug_id` déjà écrits) ; biais de formatage vérifié
puis corrigé (§14.1).

## 20. Suppression accidentelle d'un lien symbolique lors d'un nettoyage disque

Lors d'une réorganisation du dépôt, le dossier racine `saes/` (30 Go) a été
identifié comme un doublon legacy de `local_data/saes/` et supprimé après
confirmation. Erreur : `local_data/saes/gemma-scope-2-12b-it` était en
réalité un lien symbolique vers `saes/gemma-scope-2-12b-it-res`, pas un
dossier réel — sa suppression a effacé la seule copie physique des poids SAE
GemmaScope, laissant un lien cassé.

Impact réel : nul sur les résultats déjà produits (artefacts déjà écrits sur
disque indépendamment des poids sources). Seul job affecté :
`results_v12_sanity_frozen_decoder` (échec immédiat au chargement du SAE).
Corrigé : lien cassé supprimé, poids des 3 largeurs retéléchargés directement
vers le chemin canonique (dossier réel, plus de lien symbolique). Job
relancé (41082).

## 47. Échecs d'infrastructure sur 5 jobs (42687/42688/42694/42696-42698/42703-42704)

Deux causes distinctes, constatées après une interruption de session ayant
empêché le monitoring en temps réel :

- **42687/42688/42694** : `sacct` montre `FAILED, ExitCode 1:0`, aucun
  traceback Python — le process s'arrête net au milieu d'une barre de
  progression. Cause probable au niveau infrastructure (nœud), pas un bug du
  pipeline. Resoumis tels quels.
- **42696/42697/42698, 42703/42704** : échec immédiat, poids GemmaScope pour
  les layers 12/31/41 et les hook-points `attn_out`/`mlp_out` jamais
  téléchargés localement (seul `resid_post/layer_24` l'était). Nœuds de
  calcul offline, le fallback Hub échoue aussi. Corrigé : les 5
  configurations manquantes téléchargées (`download_sae.py --sae-only`,
  ~6,5 Go total) depuis un shell avec accès réseau. Jobs resoumis
  (42812-42816).
