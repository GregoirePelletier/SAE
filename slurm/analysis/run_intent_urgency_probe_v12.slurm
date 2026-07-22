#!/bin/bash
#SBATCH --job-name=SAE_intent_v12
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=intent_urgency_v12_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

# Zéro calcul GPU réel (réutilise p1_all_doc_acts_ext_d1024.pt déjà en cache),
# rejoué sur le run v12 pour alimenter l'onglet Urgence/Robustesse.
export SAVE_DIR="./results_v12_scaled_65k/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/intent_urgency_probe.py
