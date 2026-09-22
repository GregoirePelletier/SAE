#!/bin/bash
#SBATCH --job-name=SAE_post_encode_confirm
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@600
#SBATCH --output=logs/post_stage/e01_encode_confirm_%j.out
#SBATCH --error=logs/post_stage/e01_encode_confirm_%j.err

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# Encode CONFIRM a travers le SAE deja entraine (checkpoint fige, job 48585)
# -- necessaire pour evaluer E03/E04 sur CONFIRM sans jamais l'entrainer.
# Reutilise le "slot" diff_texts existant (POST_STAGE_INCLUDE_CONFIRM_AS_DIFF,
# cf. saev5.py) : p1_frozen_core_d1024_k5.pt/p1_extended_sae.pt DEJA PRESENTS
# dans SAVE_DIR -> charges tels quels, aucun reentrainement. Le hash de cache
# d'extraction change (diff_texts non-vide desormais) : une extraction
# fraiche de TOUT le corpus (FIT+DEV+CONFIRM) est attendue ici, pas une
# reutilisation partielle -- attendu et sans risque (garde-fou de forme
# existant), pas un signe d'erreur.
#
# Sous-produit cosmetique connu : le pipeline etiquette son UMAP diff comme
# "energy/sports/support" (nom historique du slot) alors que diff_texts
# contient ici CONFIRM avec ses vrais labels d'axe -- fichier HTML local
# uniquement, sans consequence sur les resultats numeriques exportes.

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

export POST_STAGE_SPLIT_ASSIGNMENTS_PATH="${SAE_ROOT}/configs/post_stage/split_assignments.json"
export POST_STAGE_INCLUDE_CONFIRM_AS_DIFF=1

export MODEL_ID="${SAE_ROOT}/models/gemma-3-1b-it"
export MODEL_SIZE=1b
export DTYPE=bf16
export PIPELINES=p1

export N_TOKENS_EXTRA_TRAIN=8000000
export D_EXTRA=1024
export K_EXTRA=5
export EPOCHS_EXTRA=10
export N_FEATURES_TO_LABEL=10
export SAVE_DIR="${SAE_ROOT}/results_post_stage_e01_fit_1b_layer13_k5/"

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH:-}"

mkdir -p "${SAE_ROOT}/logs/post_stage" "${SAVE_DIR}/cache"
cd "${SAE_ROOT}"

srun --ntasks=1 .venv/bin/python scripts/post_stage/profile_run.py \
  --out "${SAVE_DIR}/cache/resources_profile_confirm.jsonl" \
  --log "${SAVE_DIR}/cache/pipeline_stdout_confirm.log" \
  --interval 30 \
  --fs-check-path "${SAVE_DIR}" \
  -- .venv/bin/python -u src/sae/saev5.py
