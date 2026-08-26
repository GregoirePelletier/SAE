#!/bin/bash

#SBATCH --job-name=SAE_cluster_llm
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/clustering_llm_verification_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Appendix F.1/§4.3 : génération de mots-clés LLM + union top-k de latents,
# clustering Jaccard, étiquetage de cluster LLM, accuracy par réassignation
# LLM (K.3), z-score de conductance en espace dense (bge-m3) -- première
# mesure de ce dépôt comparable au papier pour le clustering
# (AUDIT_SAE_2026-08.md §1/§7).
export SAVE_DIR="./results_v10_emails_main/"
export AXIS_QUERY="type de réclamation client"
export N_CLUSTERS=4
export N_DOCS=300

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/clustering_llm_test.py
