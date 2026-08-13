#!/bin/bash

#SBATCH --job-name=SAE_v10_abl_100k
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_v10_abl_tok100k_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation volume (bras BAS) : corpus IDENTIQUE à run_sae_v10_emails.slurm
# (emails+augmentés dominants), seul N_TOKENS_EXTRA_TRAIN change (500000 -> 100000).
# Objectif : isoler l'effet du volume d'entraînement de l'extension SAE de l'effet
# du domaine (déjà corrigé dans les deux bras). SAVE_DIR séparé et extraction
# refaite intégralement (pas de partage de cache avec run_sae_v10_emails.slurm) :
# le cache des résidus bruts (p1_raw_residuals.pt) est dimensionné à
# N_TOKENS_EXTRA_TRAIN à sa création -- le partager entre configs différentes
# réutiliserait silencieusement la mauvaise taille (contamination par du cache
# de tailles/dimensions incompatibles).
# Coût : re-extraction complète (~identique en temps à run_sae_v10_emails.slurm),
# accepté ici pour garantir la validité de la comparaison.
export SEED=42
export SAVE_DIR="./results_v10_ablation_tok100k/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_SIZE=12b
export DTYPE=bf16
export N_FEATURES_TO_LABEL=150
export N_TOKENS_EXTRA_TRAIN=100000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
