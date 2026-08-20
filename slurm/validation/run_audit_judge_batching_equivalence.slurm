#!/bin/bash
#SBATCH --job-name=SAE_judgebatch
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/audit_judge_batching_equivalence_%j.log

# Vérifie le correctif de batching du juge (AUDIT_SAE_2026-08.md §2.6, item 1)
# avant de le considérer validé pour la production : équivalence stricte
# séquentiel vs batché, puis gain réel. Indépendant de tout run en cours.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_judge_batching_equivalence.py
