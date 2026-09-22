#!/bin/bash
#SBATCH --job-name=SAE_embed_cmp
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analysis/embed_cmp_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/embedding_model_comparison_test.py
