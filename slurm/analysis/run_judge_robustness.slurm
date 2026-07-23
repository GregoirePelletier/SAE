#!/bin/bash

#SBATCH --job-name=SAE_judge_robust
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analysis/judge_robustness_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Teste si le taux d'interprétabilité résiduel (~55-59% de features non interprétées,
# cf. RESULTS_TESTS.md §12 / report/04_limites_et_perspectives.md) est en partie dû au
# biais de position du protocole odd-one-out (une seule décision greedy par feature)
# plutôt qu'aux features elles-mêmes. Réutilise les activations/fragments DÉJÀ
# calculés par run_sae_v10_emails.slurm (job 39660) -- aucune réextraction Gemma-3,
# juste un rechargement du modèle comme juge.
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export N_REPEATS=5
export SEED=42
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/judge_robustness_check.py
