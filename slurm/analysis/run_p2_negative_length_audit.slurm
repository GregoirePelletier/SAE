#!/bin/bash

#SBATCH --job-name=SAE_p2_neg_audit
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analysis/p2_negative_length_audit_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# CPU-only : aucun modèle chargé, aucun GPU réservé (--gres=gpu absent
# volontairement) -- lecture de tenseurs déjà en cache (embeddings F2LLM-v2,
# checkpoint PhraseLevelSAE) et réplication de la sélection pos/neg déjà
# jugée (RESULTS_TESTS.md §93/§115). Soumis sur `a100` (idle, sans lien avec
# la campagne de rejugement en h100/h100-bis) pour ne pas rivaliser avec elle.
export SEED=42
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/p2_negative_length_audit.py
