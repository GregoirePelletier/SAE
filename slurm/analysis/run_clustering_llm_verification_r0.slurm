#!/bin/bash

#SBATCH --job-name=SAE_cluster_r0
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/clustering_llm_verification_r0_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Clustering ciblé (App. F.1/§4.3) sous R0 : RESULTS_TESTS.md §86 utilisait
# les 68 features interprétables de la campagne historique (sélection par
# magnitude, juge auto-référent) -- ce run réutilise scripts/clustering_llm_test.py
# à l'identique (déjà paramétré par SAVE_DIR, déjà juge Qwen3.8-27B via
# load_judge_model), seulement pointé sur R0 : 197 features interprétables
# stratifiées + Qwen (p1_top_extended_features.json de
# results_v27_ablation_classic_setup_k5_25m_layer31/).
export SAVE_DIR="./results_v27_ablation_classic_setup_k5_25m_layer31/"
export AXIS_QUERY="type de réclamation client"
export N_CLUSTERS=4
export N_DOCS=300

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/clustering_llm_test.py
