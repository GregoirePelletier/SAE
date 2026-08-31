# Opérations — cluster, réseau, environnement

## Cluster SLURM

Trois partitions GPU (`a100`, `h100`, `h100-bis`, 8 GPU/nœud chacune). Les
nœuds de calcul n'ont pas d'accès réseau direct (`HF_HUB_OFFLINE=1`
systématique dans les scripts `.slurm`, `.venv/bin/python` plutôt que `uv run`
qui tenterait de re-résoudre l'environnement).

Si un job reste `PD` avec la raison `ReqNodeNotAvail, UnavailableNodes:...`,
ce n'est pas de la congestion (un simple changement de partition ne résout
rien) : le nœud lui-même est down. Vérifier avec `sinfo -p <partition> -N -l`
avant de basculer vers une autre partition — un `STATE` à `down*`/raison
`Not responding` confirme la panne.

**`a100` n'a plus la marge pour charger le juge par défaut du pipeline
principal.** Depuis que `JUDGE_MODEL_ID` défaut à Qwen3.8-27B bf16 (~52 Go de
poids), tout run P1/P2 qui atteint l'étape de labellisation LLM sur `a100`
(39,49 Go de capacité sur ce cluster) OOM à ce moment précis —
`torch.OutOfMemoryError` sur `caching_allocator_warmup`, *après* que
l'extraction/entraînement aient tourné jusqu'au bout (job 45724, `a100`,
1B/layer13 : 2h23 d'extraction complètes puis crash sur le chargement du
juge, seule l'extraction — coûteuse mais réutilisable via le cache partagé —
a été sauvée). `h100` obligatoire pour tout `.slurm` de `slurm/pipeline_runs/`
qui ne pin pas explicitement `JUDGE_MODEL_ID`/`ALT_JUDGE_MODEL_ID` sur un
juge plus petit — vérifier la partition AVANT de soumettre, pas après
observation d'un crash tardif. Conséquence sur la course de doublons
a100/h100 (juste en dessous) : une course n'a plus de sens pour un job qui
atteint la labellisation — la copie `a100` est condamnée à échouer au
dernier pas, quel que soit le résultat de la course elle-même ; ne soumettre
que sur `h100`/`h100-bis` pour ce type de job.

### Arborescence des scripts de soumission

`slurm/<catégorie>/*.slurm`, sortie (`--output`) configurée vers
`logs/<catégorie>/` (même nom de sous-dossier), jamais à la racine du dépôt :

| Catégorie | Contenu |
|---|---|
| `pipeline_runs/` | runs `saev5.py` (Pipeline 1/2) : run principal, ablations volume/largeur/époques/capacité |
| `baseline_diffing/` | `scripts/baseline_gemmascope.py` : diffing SAE natif originaux vs augmentés |
| `augmentation/` | `scripts/run_augmentation.py` : génération des variantes de mails augmentées |
| `analysis/` | tests post-hoc sur activations déjà en cache (robustesse du juge, fidélité/plausibilité, sondes, corrélations, comparaison d'embeddings) |
| `validation/` | smoke-tests ad hoc |

```bash
sbatch slurm/pipeline_runs/<script>.slurm
squeue -u $USER
tail -f logs/pipeline_runs/<nom>_<jobid>.log
```

`logs/` est gitignoré (`*.log`) — seuls les `.slurm` sont versionnés. Le
suivi des résultats de chaque run vit dans `RESULTS_TESTS.md`, pas dans les
logs bruts.

### Dette de configuration (N11, AUDIT_SAE_2026-08.md §8)

Le nombre de `.slurm` croît plus vite qu'il n'est nettoyé (63 → 87 constaté,
malgré une passe de suppression explicite) — deux règles pour freiner ça :

1. **Archiver, pas laisser traîner.** Dès que le `§N` de `RESULTS_TESTS.md`
   correspondant à un `.slurm` est écrit (le run est terminé, cité, conclu),
   déplacer le script vers `slurm/archive/<catégorie>/` (même sous-arborescence
   que l'original). Le script reste lisible/versionné pour reproduire le run
   plus tard, mais sort de l'arborescence "active" que l'on parcourt pour
   lancer de nouveaux jobs.
2. **Réutiliser avant de créer.** Un nouveau `.slurm` ne se crée que si aucun
   `.slurm` existant (actif OU archivé) ne couvre déjà la même config à un
   export d'environnement près — dans ce cas, réutiliser le script existant
   avec `export VAR=... sbatch script.slurm` (ou un variant `_h100`/`_a100`
   de partition, déjà la convention pour la course entre partitions) plutôt
   que dupliquer un fichier quasi identique.

### Reprise après coupure (checkpoint, code de sortie 64)

Chaque `.slurm` de `pipeline_runs/` déclare `--signal=B:USR1@600` : SLURM
envoie `SIGUSR1` 600s avant la limite de temps (`--time`), `GracefulShutdown`
(`src/storage/checkpoint.py`) l'intercepte et écrit un checkpoint atomique
avant de sortir avec le code **64** (`EXIT_CODE_GRACEFUL_CHECKPOINT`), pas 0
— distinct d'un run réellement terminé, pour qu'un `sacct`/
`--dependency=afterok:<jobid>` en aval puisse le distinguer et ne pas
enchaîner sur un résultat incomplet. Un resoumission du même `.slurm` sur le
même `SAVE_DIR` reprend automatiquement depuis ce checkpoint.

### Disque

Le disque partagé (`/home`) est souvent proche de la capacité — vérifier
`df -h .` avant tout téléchargement/cache volumineux. Les artefacts d'un
ancien run (fragments token-level, activations brutes) ne sont pas garantis
présents sur disque : ils sont parfois purgés après coup pour l'espace, seuls
les JSON légers de résultats survivent systématiquement.

Avant de supprimer un dossier qui a l'air d'un doublon lors d'un nettoyage,
vérifier qu'il n'est pas un **lien symbolique** vers la copie physique réelle
(`ls -la`/`readlink`) — un dossier `local_data/saes/...` qui ressemblait à un
doublon d'un dossier racine `saes/` était en réalité l'inverse : le lien
pointait vers l'unique copie physique des poids SAE, supprimée par erreur.

**Avant de télécharger un modèle de base (LM) volumineux, vérifier
`/mnt/gvd/modeles_ia/`** (même filesystem `10.52.209.65:/DGX` que `/home`,
donc même quota) : dépôt partagé large (Llama, Mistral, Qwen, Gemma, GPT-OSS,
etc.) alimenté par d'autres utilisateurs du cluster, pas indexé nulle part
dans ce projet. `gemma-3-27b-it` (52 Go) y était déjà présent en entier
lorsqu'un téléchargement personnel redondant a été lancé faute de l'avoir
vérifié d'abord (nettoyé depuis) — `MODEL_ID` peut pointer un chemin absolu
sous ce répertoire directement, `from_pretrained(..., local_files_only=True)`
n'a besoin d'aucune étape d'import. Les SAE GemmaScope-2 n'y sont en revanche
jamais présents (spécifique à ce projet), `download_sae.py` reste nécessaire
pour eux.

**Cache d'extraction partagé entre runs** (`local_data/activation_cache/`,
`sae_shared.py::compute_activation_cache_key`/`shared_activation_cache_dir`) :
les résidus bruts, activations core et fragments token-level sont partagés
entre tous les runs de même modèle/couche/hook/corpus/budget de tokens, quel
que soit `K_EXTRA`/`D_EXTRA`/`EPOCHS_EXTRA` (downstream, sans effet sur
l'extraction) — via des liens symboliques créés sous `SAVE_DIR/cache/`, pas
une redirection directe (une dizaine de scripts d'analyse lisent ces chemins
sous `SAVE_DIR/cache` directement). **Conséquence sur la course de doublons
a100/h100** (pratique établie pour réduire le temps de file d'attente,
`AUDIT_SAE_2026-08.md`) : deux jobs qui partagent la MÊME clé d'extraction
(même modèle/couche/hook/corpus/budget) écrivant simultanément dans le même
cache partagé peuvent se marcher dessus (deux `ShardedFragmentWriter`, deux
écritures memmap concurrentes sur le même fichier). Annuler le doublon perdant
dès qu'un des deux passe en état `R` (politique déjà en vigueur) reste
suffisant SI la vérification a lieu avant que les deux jobs atteignent la
phase d'extraction (quelques minutes après le démarrage, le temps de charger
le modèle) — ne pas laisser une course avec clé d'extraction identique tourner
sans surveillance au-delà de cette fenêtre.

### Réseau et portabilité

`CLUSTER_OFFLINE_MODE=1` (`src/config.py`) désactive la vérification SSL et
force `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`/`HF_DATASETS_OFFLINE` — reproduit
l'environnement cluster. Désactivé par défaut (`0`) pour permettre le premier
téléchargement en local. `MODEL_ID` pointe un repo HuggingFace (pas un chemin
disque figé) : une fois `download_sae.py` exécuté, il est résolu depuis le
cache HF local, portable entre machines.

Les labels Neuronpedia sont récupérés par téléchargement direct des lots
`.jsonl.gz` publics du bucket S3 `neuronpedia-datasets` (pas via l'API REST),
et mis en cache localement (`local_data/neuronpedia_labels/`, partagé entre
tous les runs) — voir `fetch_neuronpedia_labels()`.

`download_sae.py` ne télécharge par défaut que la configuration SAE utilisée par
le run principal (`resid_post`, couche 24). Tout balayage de couche ou de
hook-point (`attn_out`/`mlp_out`, autres couches) nécessite un
`download_sae.py --sae-only` explicite pour cette configuration avant de
soumettre le job — les nœuds de calcul étant offline, un poids manquant y échoue
immédiatement, sans repli possible vers le Hub.

## Installation locale

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate   |   Linux/Mac : source .venv/bin/activate
pip install -e .
```

### Accès HuggingFace (obligatoire)

`google/gemma-3-*-it` et `google/gemma-scope-2-*-it` sont des repos gated :

1. Créer un compte sur [huggingface.co](https://huggingface.co).
2. Accepter la licence Gemma sur la page du modèle ciblé.
3. Générer un token sur <https://huggingface.co/settings/tokens>.
4. Copier `.env.example` en `.env` et y placer `HF_TOKEN=hf_...` (gitignored).

### Windows

- Nom d'utilisateur accentué ou chemin `HOME` long : le cache HuggingFace peut
  dépasser `MAX_PATH` (260 caractères) sur les fichiers de verrou → `OSError
  [Errno 22]`. Solution : `HF_HOME=C:\hfcache` (chemin court, sans accent).
- Liens symboliques non autorisés (mode développeur désactivé) :
  `HF_HUB_DISABLE_SYMLINKS=1`.
- Affichage console UTF-8 (le code utilise des caractères comme `→`) :
  `PYTHONUTF8=1` et `PYTHONIOENCODING=utf-8`.
