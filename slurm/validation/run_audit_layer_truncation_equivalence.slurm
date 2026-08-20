#!/bin/bash
#SBATCH --job-name=SAE_layertrunc
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/audit_layer_truncation_equivalence_%j.log

# Vérifie le correctif G1 (AUDIT_SAE_2026-08.md §2.2) avant de le considérer
# validé pour la production : équivalence stricte de hidden_states[LAYER]
# avec/sans troncature de layers[:LAYER], puis débit/VRAM réels. Sur a100,
# en parallèle du job 44211 (h100) -- aucune contention.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LAYER=24
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_layer_truncation_equivalence_and_speedup.py
