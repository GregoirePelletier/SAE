#!/bin/bash
#SBATCH --job-name=SAE_augmentation_test
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=1:30:00
#SBATCH --output=logs/augmentation_test_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export LOCAL_MAILS_PATH="/home/h21486/SAE/Mails.tsv"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
# Doit matcher le SAVE_DIR utilisé par run_baseline.slurm.
export SAVE_DIR="./results_v9_test/"
# Run de validation rapide : 60 mails × 13 axes/niveaux ≈ 780 générations
# (≈65 min au rythme observé sur job 38949, vs ~63h GPU pour le corpus complet
# 3480 mails × 13 -- cf. RESULTS_TESTS.md). Sortie dans un fichier distinct pour ne
# pas mélanger avec un futur run complet sur augmented_mails.jsonl.
export AUGMENT_SAMPLE_N="60"
export AUGMENT_OUT_NAME="augmented_mails_test.jsonl"

mkdir -p logs "${SAVE_DIR}"

cd /home/h21486/SAE/
# run_augmentation.py est dans scripts/, pas à la racine ; uv run tentait aussi de
# re-résoudre l'env via le réseau (bloqué sur le cluster) -> échec systématique.
.venv/bin/python scripts/run_augmentation.py