#!/bin/bash
#SBATCH --job-name=interp_embed_tutorial
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/validation/interp_embed_tutorial_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Aucun GPU demandé : les .pkl du tutoriel (nickjiang/feature_labels,
# artifacts/gpt52_gemini3_tutorial/) contiennent déjà les activations SAE
# précalculées -- ce job ne fait que désérialiser + calculer une diff de
# fréquences (numpy/pandas), zéro inférence modèle.

cd /home/h21486/SAE/
/home/h21486/SAE/external/interp_embed/.venv/bin/python \
    /home/h21486/SAE/scripts/validate_interp_embed_tutorial.py
