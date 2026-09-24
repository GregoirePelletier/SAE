#!/bin/bash
#SBATCH --job-name=SAE_batchsweep
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/extraction_batch_sweep_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Balayage EXTRACTION_BATCH_SIZE (benchmarks/extraction_batch_size_sweep.py) --
# à relancer à chaque changement de GPU, de modèle ou de couche.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"
export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export LAYER=24
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"

cd "${SAE_ROOT}/"
.venv/bin/python benchmarks/extraction_batch_size_sweep.py
