#!/bin/bash
#SBATCH --job-name=SAE_post_e05_randinit46
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@600
#SBATCH --output=logs/post_stage/e05_randinit46_%j.out
#SBATCH --error=logs/post_stage/e05_randinit46_%j.err

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# E05 -- bras TEMOIN "init aleatoire" (independance a l'initialisation) : meme config
# que 10_e05_seed43 mais EXTRA_DECODER_INIT=random (directions gaussiennes
# isotropes normalisees sous SEED au lieu des directions PCA du reservoir,
# echelles/biais calibres a l'identique). SAVE_DIR DISTINCT obligatoire (jamais
# le dossier d'un run pca : load_or_train_extended_sae ne valide pas la config
# d'un checkpoint existant). Rappel du bras principal (plan §10.2 :
# "Trois checkpoints d'extension, memes donnees FIT, meme architecture,
# memes hyperparametres, trois etats de hasard documentes"). MEME config
# que 01_e01_fit_reference.slurm (SAVE_DIR different, seed differente) --
# le cache d'extraction partage (compute_activation_cache_key, cf. CLAUDE.md
# "Seeds") ne depend PAS de SEED/SAVE_DIR : ce run reutilise le reservoir
# de tokens deja extrait par job 48585 (meme corpus/modele/couche/hook),
# seul l'entrainement de l'extension (init + ordre de melange) varie.

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

export N_TOKENS_EXTRA_TRAIN=8000000
export D_EXTRA=1024
export K_EXTRA=5
export EPOCHS_EXTRA=10
export N_FEATURES_TO_LABEL=10
export SEED=46
export EXTRA_DECODER_INIT=random
export SAVE_DIR="${SAE_ROOT}/results_post_stage_e05_randinit46${RUN_SUFFIX:-}/"

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH:-}"

mkdir -p "${SAE_ROOT}/logs/post_stage" "${SAVE_DIR}/cache"
cd "${SAE_ROOT}"

srun --ntasks=1 .venv/bin/python scripts/post_stage/profile_run.py \
  --out "${SAVE_DIR}/cache/resources_profile.jsonl" \
  --log "${SAVE_DIR}/cache/pipeline_stdout.log" \
  --interval 30 \
  --fs-check-path "${SAVE_DIR}" \
  -- .venv/bin/python -u src/sae/saev5.py
