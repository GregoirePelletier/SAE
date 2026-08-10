#!/bin/bash

#SBATCH --job-name=SAE_judge_sample
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/judge_sampling_ensemble_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Vote majoritaire par ÉCHANTILLONNAGE (temperature>0), distinct du réordonnancement
# déjà testé par run_judge_robustness.slurm. Réutilise les activations/fragments déjà
# calculés (job 39660) -- aucune réextraction Gemma-3.
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export N_REPEATS=5
export JUDGE_TEMPERATURE=0.7
export SEED=42
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/judge_sampling_ensemble_test.py
