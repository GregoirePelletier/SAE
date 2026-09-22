#!/bin/bash
#SBATCH --job-name=SAE_post_e00_profile_v2
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@600
#SBATCH --output=logs/post_stage/e00_profile_%j.out
#SBATCH --error=logs/post_stage/e00_profile_%j.err

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# E00 -- validation du correctif C1 (docs/post_stage/memory_diagnosis.md),
# apres le premier profilage (job 48515, correctif pas encore applique a
# l'epoque). Meme config qu'alors (1B/layer13, 250k tokens residus, 20k
# filler) SAUF MAX_AUGMENTED_PER_MAIL=1 : le premier profilage avait laisse
# le defaut (13, saev5.py:298) sans le savoir, traitant en realite les 36 602
# documents de la production complete au lieu d'un sous-echantillon --
# corrige ici pour qu'un run complet (jusqu'aux sondes en aval) tienne dans
# la fenetre de 2h. But : confirmer sur un run reel que le pic transitoire
# mesure au moment de torch.stack(all_doc_sae_acts) a disparu/est reduit a
# l'echelle desormais reellement utile (train+test+diff, sans filler).
#
# Pas de --signal=B:USR1 (avertissement explicite du plan, §16.4) : le "B:"
# viserait le shell batch, pas necessairement python. Ici le signal va au
# process lance par srun (scripts/post_stage/profile_run.py), qui le
# retransmet explicitement a saev5.py (cf. profile_run.py, signal.signal) --
# verifie par tests/post_stage/test_profile_run.py::test_forwards_sigusr1_to_child.

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export CLUSTER_OFFLINE_MODE=1

export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-1b-it"
export MODEL_SIZE=1b
export DTYPE=bf16
export PIPELINES=p1

# §6.1/§5.7 du plan : 1B, layer13 (preset MODEL_SIZE=1b), 250k tokens residus,
# meme D_EXTRA/K_EXTRA/EPOCHS_EXTRA que le protocole E01 pour que ce profil
# soit reutilisable comme reference memoire de ce run precis, pas seulement
# comme test synthetique.
export N_TOKENS_EXTRA_TRAIN=250000
export N_VOLUME_FILLER_TARGET_CHUNKS=20000
export MAX_AUGMENTED_PER_MAIL=1
export D_EXTRA=1024
export K_EXTRA=5
export EPOCHS_EXTRA=10
export N_FEATURES_TO_LABEL=10
export SAVE_DIR="${SAE_ROOT}/results_post_stage_e00_profile_1b_v2/"

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH:-}"

mkdir -p "${SAE_ROOT}/logs/post_stage" "${SAVE_DIR}/cache"
cd "${SAE_ROOT}"

srun --ntasks=1 .venv/bin/python scripts/post_stage/profile_run.py \
  --out "${SAVE_DIR}/cache/resources_profile.jsonl" \
  --log "${SAVE_DIR}/cache/pipeline_stdout.log" \
  --interval 20 \
  --fs-check-path "${SAVE_DIR}" \
  -- .venv/bin/python -u src/sae/saev5.py
