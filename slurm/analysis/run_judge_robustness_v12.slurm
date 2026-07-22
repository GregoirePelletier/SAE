#!/bin/bash
#SBATCH --job-name=SAE_judge_robust_v12
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=judge_robustness_v12_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Même protocole que run_judge_robustness.slurm, rejoué sur le run v12 (65k, 600
# features labellisées) -- réutilise les activations/fragments déjà calculés par
# run_sae_v12_scaled.slurm, aucune réextraction Gemma-3.
export SAVE_DIR="./results_v12_scaled_65k/"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_65k_l0_medium"
export N_REPEATS=5
export SEED=42
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/judge_robustness_check.py
