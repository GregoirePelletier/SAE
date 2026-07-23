#!/bin/bash
#SBATCH --job-name=SAE_baseline_v12
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_diffing/baseline_v12_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_SAE_DIR="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res"
export SAE_ID="layer_24_width_65k_l0_medium"

# Diffing par axe (original vs chaque axe d'augmentation) dans le MÊME SAVE_DIR
# que run_sae_v12_scaled.slurm, pour que l'onglet Diffing du dashboard ait à la
# fois la démo cross-domaine (p1_diff_energy_sports.csv, produite par saev5.py)
# ET le diffing par axe email (diff_*.csv ici) dans un seul run. Indépendant du
# run P1/P2 (SAE GemmaScope natif rechargé directement) -> peut tourner en
# parallèle. Cache dédié pour éviter toute collision avec CACHE_DIR de saev5.py.
export SAVE_DIR="./results_v12_scaled_65k/"
export CACHE_DIR="${SAVE_DIR}cache_baseline_full"

mkdir -p logs "${CACHE_DIR}"

cd /home/h21486/SAE/
.venv/bin/python scripts/baseline_gemmascope.py "${LOCAL_MAILS_PATH}" "${LOCAL_AUGMENTED_MAILS_PATH}"
