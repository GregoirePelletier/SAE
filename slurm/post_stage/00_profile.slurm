#!/bin/bash
#SBATCH --job-name=SAE_post_e00_profile
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@600
#SBATCH --output=/home/h21486/SAE/logs/post_stage/e00_profile_%j.out
#SBATCH --error=/home/h21486/SAE/logs/post_stage/e00_profile_%j.err

# E00 -- premier profilage reel, borne (Plan_execution_SAE_15_jours_Claude_Code.md
# §5.7) : 1B/layer13, 250k tokens residus, filler MODESTE (20k chunks, pas les
# ~1,2M d'un run de production) pour observer un premier ordre de grandeur du
# cout memoire des lignes filler dans all_doc_sae_acts (C1) sans faire courir
# ~1,2M forwards Gemma-3-1B supplementaires des la premiere mesure. Aucun
# correctif applique encore -- ce job MESURE le code actuel tel quel.
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

export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-1b-it"
export MODEL_SIZE=1b
export DTYPE=bf16
export PIPELINES=p1

# §6.1/§5.7 du plan : 1B, layer13 (preset MODEL_SIZE=1b), 250k tokens residus,
# meme D_EXTRA/K_EXTRA/EPOCHS_EXTRA que le protocole E01 pour que ce profil
# soit reutilisable comme reference memoire de ce run precis, pas seulement
# comme test synthetique.
export N_TOKENS_EXTRA_TRAIN=250000
export N_VOLUME_FILLER_TARGET_CHUNKS=20000
export D_EXTRA=1024
export K_EXTRA=5
export EPOCHS_EXTRA=10
export N_FEATURES_TO_LABEL=10
export SAVE_DIR="/home/h21486/SAE/results_post_stage_e00_profile_1b/"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH:-}"

mkdir -p /home/h21486/SAE/logs/post_stage "${SAVE_DIR}/cache"
cd /home/h21486/SAE

srun --ntasks=1 .venv/bin/python scripts/post_stage/profile_run.py \
  --out "${SAVE_DIR}/cache/resources_profile.jsonl" \
  --log "${SAVE_DIR}/cache/pipeline_stdout.log" \
  --interval 20 \
  --fs-check-path "${SAVE_DIR}" \
  -- .venv/bin/python -u src/sae/saev5.py
