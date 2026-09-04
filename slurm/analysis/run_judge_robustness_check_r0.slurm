#!/bin/bash
#SBATCH --job-name=SAE_robust_r0
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/analysis/sae_judge_robustness_r0_%j.log

# Robustesse au réordonnancement sous R0 (RESULTS_TESTS.md §13.1 réutilisait
# le juge auto-référent MODEL_ID=Gemma-3-12b-it sur la sélection historique par
# magnitude -- corrigé ici : juge Qwen3.8-27B découplé, doc_groups=train_groups
# (déduplication mail parent), 300 features stratifiées de R0). Zéro
# réextraction -- réutilise les activations/fragments déjà en cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SAVE_DIR="./results_v27_ablation_classic_setup_k5_25m_layer31/"
export N_REPEATS=5
export SEED=42

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/judge_robustness_check_r0_test.py
