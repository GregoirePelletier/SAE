#!/bin/bash
#SBATCH --job-name=SAE_multilingual
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/analysis/multilingual_judge_bias_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Biais multilingue du juge (Resck et al. 2025 ; SAEs Can Capture
# Language-Specific Concepts Across Diverse Languages, arXiv:2507.11230) ────────
# Teste si le juge (Gemma-3-12B-it) interprète MIEUX les mêmes 150 features déjà
# jugées quand les exemples/le prompt sont traduits en anglais -- hypothèse que les
# LLM représentent souvent les concepts multilingues via une structure interne
# dominée par l'anglais. Réutilise les activations/fragments déjà en cache
# (results_v10_emails_main), aucune réextraction Gemma-3, aucun réentraînement.
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SEED=42
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/multilingual_judge_bias_test.py
