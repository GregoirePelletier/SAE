# Opérations — cluster, réseau, environnement

## Cluster SLURM

Trois partitions GPU (`a100`, `h100`, `h100-bis`, 8 GPU/nœud chacune). Les
nœuds de calcul n'ont pas d'accès réseau direct (`HF_HUB_OFFLINE=1`
systématique dans les scripts `.slurm`, `.venv/bin/python` plutôt que `uv run`
qui tenterait de re-résoudre l'environnement).

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

### Disque

Le disque partagé (`/home`) est souvent proche de la capacité — vérifier
`df -h .` avant tout téléchargement/cache volumineux. Les artefacts d'un
ancien run (fragments token-level, activations brutes) ne sont pas garantis
présents sur disque : ils sont parfois purgés après coup pour l'espace, seuls
les JSON légers de résultats survivent systématiquement.

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
