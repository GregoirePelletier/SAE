#!/bin/bash
#SBATCH --job-name=SAE_e9_sizematch
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/audit_e9_size_matched_embedding_compare_%j.log

# Audit 2026-08, E.9 : refait la comparaison bge-m3 vs F2LLM (job 40730,
# select_latents_by_similarity) a taille de modele bien plus proche
# (F2LLM-v2-330M au lieu de F2LLM-v2-80M, ecart reduit de ~26x a ~6,6x).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_e9_size_matched_embedding_compare.py
