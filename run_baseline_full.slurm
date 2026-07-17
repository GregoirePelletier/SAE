#!/bin/bash
#SBATCH --job-name=SAE_baseline_full
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_full_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# cf. run_baseline.slurm (test) : fix OOM CUDA (logits_to_keep + no_grad réel dans
# activations.py) déjà validé sur job 39003 ; gardé en défense supplémentaire.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export LOCAL_MAILS_PATH="/home/h21486/SAE/Mails.tsv"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_SAE_DIR="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res"
export SAVE_DIR="./results_v9_test/"
export CACHE_DIR="${SAVE_DIR}cache_baseline_full"

mkdir -p logs "${CACHE_DIR}"

cd /home/h21486/SAE/
# Corpus complet : ${SAVE_DIR}augmented_mails.jsonl doit avoir été généré et fusionné
# au préalable via run_augmentation_full.slurm (array 8 shards) + merge_augmentation_shards.sh.
.venv/bin/python scripts/baseline_gemmascope.py "${LOCAL_MAILS_PATH}" "${SAVE_DIR}augmented_mails.jsonl"
