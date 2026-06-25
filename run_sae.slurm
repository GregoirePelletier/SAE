#!/bin/bash
#SBATCH --job-name=SAE_Dual_Pipeline
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3:00:00
#SBATCH --output=sae_v7_%j.log

# 1. Isolation réseau totale (On interdit aux libs de chercher le proxy)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 2. Variables applicatives
export SEED=42
export SAVE_DIR="./results_v7/"
export USE_FROZEN_CORE="1"

# 3. Exécution locale
cd /home/h21486/SAE/
uv run saev5.py