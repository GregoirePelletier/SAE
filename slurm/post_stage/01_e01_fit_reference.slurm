#!/bin/bash
#SBATCH --job-name=SAE_post_e01_fit_ref
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@600
#SBATCH --output=logs/post_stage/e01_fit_reference_%j.out
#SBATCH --error=logs/post_stage/e01_fit_reference_%j.err

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# E01 -- run de reference sur FIT SEUL (Plan_execution_SAE_15_jours_Claude_Code.md
# §6.1) : 1B/layer13 (preset MODEL_SIZE=1b), coeur 16k, extension D_EXTRA=1024/
# K_EXTRA=5, MEME BATCH_SIZE_EXTRA/EPOCHS_EXTRA que le defaut du depot (pas de
# changement silencieux). DEV joue le role "test" de saev5.py (validation
# interne/quality gates, §6.2) -- CONFIRM est totalement absent de ce run,
# jamais charge (POST_STAGE_SPLIT_ASSIGNMENTS_PATH, cf. dataset_contract.py).
#
# Pas de filler (N_VOLUME_FILLER_TARGET_CHUNKS=0, defaut) : ce premier run
# propre mesure ce que FIT offre reellement en tokens (§6.1 -- "si le corpus
# emails offre 4M tokens pertinents, les conserver comme tels"), rapporte a la
# fin (n_residuals_collected dans les logs). Accessoirement, l'absence de
# filler evite le second sink memoire encore non corrige
# (memory_diagnosis.md §7, p1_all_doc_acts_ext_d*.pt) -- sans filler, ce
# tableau n'est deja plus surdimensionne.

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

export MODEL_ID="${SAE_ROOT}/models/gemma-3-1b-it"
export MODEL_SIZE=1b
export DTYPE=bf16
export PIPELINES=p1

# §6.1 du plan : configuration de reference E01. MAX_AUGMENTED_PER_MAIL au
# defaut du depot (13, src/config.py) -- continuite avec la config historique
# de RESULTS_TESTS.md, pas un nouveau choix. N_TOKENS_EXTRA_TRAIN large
# (borne haute, pas une cible) : le reservoir se tronque naturellement a ce
# que FIT fournit reellement si c'est moins -- lire n_residuals_collected
# dans les logs plutot que supposer 8000000 atteint.
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
  --out "${SAVE_DIR}/cache/resources_profile.jsonl" \
  --log "${SAVE_DIR}/cache/pipeline_stdout.log" \
  --interval 30 \
  --fs-check-path "${SAVE_DIR}" \
  -- .venv/bin/python -u src/sae/saev5.py
