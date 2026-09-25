# Exploitation : cluster, stockage, installation

## Cluster Slurm

- Trois partitions GPU : `a100`, `h100`, `h100-bis`, 8 GPU par nœud. Les A100 ont 39,5 Go
  utilisables, les H100 environ 80 Go.
- Les nœuds de calcul n'ont pas accès à Internet : les recettes positionnent `HF_HUB_OFFLINE=1` et
  appellent `.venv/bin/python` directement (pas `uv run`, qui tenterait de résoudre les
  dépendances).
- Aucun calcul sur le nœud frontal : tout passe par `sbatch`.
- Les logs vont dans `logs/<catégorie>/`, du nom du sous-dossier de `slurm/` ; ils ne sont pas
  versionnés.

**Choisir la partition.** Le modèle juge Qwen3.8-27B (environ 52 Go en bf16) ne tient pas sur une
A100. Tout job qui l'utilise, y compris un run de `saev5.py` qui nomme ses features, doit tourner
sur `h100` ou `h100-bis`, sinon il échoue au chargement du juge après avoir fait tout le reste.
Si les deux nœuds H100 sont pleins (`scontrol show node dgx-h100 dgx-h100-bis`), les recettes
`06b` et `07b` répartissent le juge sur 2 A100.

**Job bloqué.** Un job en attente avec la raison `ReqNodeNotAvail` indique un nœud en panne, pas
une file chargée : vérifier avec `sinfo -p <partition> -N -l` avant de changer de partition.

**Reprise après interruption.** Les recettes demandent un signal `USR1` 10 minutes avant la fin
du temps alloué. `src/storage/checkpoint.py` enregistre alors l'état et le processus se termine
avec le code 64 (différent de 0, pour qu'une dépendance `afterok` ne s'enchaîne pas sur un run
incomplet). Resoumettre la même recette avec le même `SAVE_DIR` reprend là où le run s'était
arrêté. La reprise n'est pas reproductible au bit près (l'ordre des lots change).

**Nouvelles recettes.** Avant d'en créer une, vérifier qu'une recette existante ne fait pas déjà
la même chose à des variables d'environnement près (`sbatch --export=ALL,VAR=valeur ...`).

## Stockage

- `/home` est souvent presque plein : vérifier `df -h .` avant tout téléchargement ou calcul qui
  produit un gros cache.
- Avant de supprimer un dossier qui ressemble à un doublon, vérifier qu'il n'est pas la cible d'un
  lien symbolique (`ls -la`, `readlink`). Des poids de SAE ont déjà été perdus ainsi.
- Avant de télécharger un modèle de langue, regarder `/mnt/gvd/modeles_ia/` : ce dépôt partagé du
  cluster contient de nombreux modèles (Llama, Mistral, Qwen, Gemma…), et `MODEL_ID` peut y
  pointer directement. Les SAE GemmaScope-2 n'y sont pas : `download_sae.py` reste nécessaire.
- Les caches volumineux des anciens runs (fragments de tokens, activations) ont parfois été
  supprimés pour libérer de la place ; seuls les JSON de résultats sont toujours conservés.

**Cache d'extraction partagé.** Les activations extraites (`local_data/activation_cache/<clé>/`)
sont partagées par tous les runs de même modèle, couche, corpus et budget de tokens, quels que
soient les paramètres de l'extension. La clé est calculée par
`sae_shared.py::compute_activation_cache_key` ; chaque run y accède par des liens symboliques dans
`SAVE_DIR/cache/`. Deux jobs de même clé qui extraient en même temps écrivent dans les mêmes
fichiers : si l'on soumet le même job sur deux partitions pour gagner du temps d'attente, annuler
l'un des deux dès que l'autre démarre, avant la phase d'extraction.

## Réseau et HuggingFace

- `CLUSTER_OFFLINE_MODE=1` (`src/config.py`) reproduit l'environnement des nœuds (pas de réseau,
  pas de vérification SSL). Par défaut à 0, pour permettre un premier téléchargement.
- `MODEL_ID` peut être un identifiant HuggingFace (résolu depuis le cache local après
  téléchargement) ou un chemin sous `models/`.
- Derrière le proxy du cluster, les gros téléchargements HuggingFace échouent en cours de route
  (« CAS Client Error ») sans `HF_HUB_DISABLE_XET=1`.
- Les noms Neuronpedia des features CORE sont téléchargés depuis les fichiers publics du bucket
  `neuronpedia-datasets` et mis en cache dans `local_data/neuronpedia_labels/`
  (`fetch_neuronpedia_labels()`).
- `download_sae.py` télécharge le modèle et le SAE de la configuration par défaut de
  `MODEL_SIZE`. Pour une autre couche ou un autre point d'accroche, lancer
  `download_sae.py --sae-only` avec cette configuration avant de soumettre le job : sans réseau,
  un poids manquant fait échouer le job immédiatement.

## Installation

```bash
uv sync --locked --python 3.12
```

Accès aux modèles Gemma et GemmaScope-2 (dépôts à accès restreint) : créer un compte HuggingFace,
accepter la licence Gemma sur la page du modèle, créer un token et le placer dans `.env`
(`HF_TOKEN=...`, fichier non versionné, à partir de `.env.example`).

Sous Windows :

- si le nom d'utilisateur contient un accent ou si le chemin est long, le cache HuggingFace peut
  dépasser la limite de 260 caractères : `HF_HOME=C:\hfcache` ;
- si les liens symboliques ne sont pas autorisés : `HF_HUB_DISABLE_SYMLINKS=1` ;
- pour l'affichage UTF-8 : `PYTHONUTF8=1` et `PYTHONIOENCODING=utf-8`.
