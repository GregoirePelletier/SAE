#!/bin/bash

#SBATCH --job-name=SAE_contrastive
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/contrastive_labeling_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Teste le protocole de labellisation contrastif direct d'interp_embed (Jiang, Sun
# et al. 2025, Appendix C) en alternative au gate odd-one-out actuel -- cf.
# docstring de scripts/contrastive_labeling_test.py. Réutilise les activations et
# fragments DÉJÀ calculés par run_sae_v10_emails.slurm (job 39660) -- aucune
# réextraction Gemma-3, juste un rechargement du modèle comme juge.
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export N_NEG=10
export SEED=42
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/contrastive_labeling_test.py
