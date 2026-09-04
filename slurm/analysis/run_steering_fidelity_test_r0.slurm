#!/bin/bash
#SBATCH --job-name=SAE_steerfid_r0
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analysis/steering_fidelity_r0_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

# Fidélité du steering sous R0 (RESULTS_TESTS.md §24/§68/§69 : toujours mesuré
# sur K_EXTRA=32/layer 24, jamais sur R0) -- scripts/steering_fidelity_test_r0.py
# ne change que D_EXTRA/K_EXTRA/SAE_ID (1024/5/layer_31) et SAVE_DIR.
export SAVE_DIR="./results_v27_ablation_classic_setup_k5_25m_layer31/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/steering_fidelity_test_r0.py
