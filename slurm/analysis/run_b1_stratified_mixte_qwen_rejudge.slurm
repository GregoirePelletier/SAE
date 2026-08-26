#!/bin/bash

#SBATCH --job-name=SAE_b1_qwen
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/b1_stratified_mixte_qwen_rejudge_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# N4 (AUDIT_SAE_2026-08.md §8) : rejuge l'arme mixte stratifiée de B.1 (§79,
# §81, b2_stratified_selection_rejudge.json) avec Qwen3.8-27B au lieu de
# gemma-3-12b-it -- même juge/patron que run_judge_model_separation_qwen.slurm.
export SAVE_DIR="./results_v10_emails_main/"
export ALT_JUDGE_MODEL_ID="/home/h21486/SAE/models/Qwen3.8-27B"
export DTYPE=bf16
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/b1_stratified_mixte_qwen_rejudge.py
