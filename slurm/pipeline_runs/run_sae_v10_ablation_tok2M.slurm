#!/bin/bash

#SBATCH --job-name=SAE_v10_abl_2M
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_v10_abl_tok2M_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation volume (bras HAUT) : corpus IDENTIQUE à run_sae_v10_emails.slurm,
# N_TOKENS_EXTRA_TRAIN=2000000 (x4 le défaut). Avec run_sae_v10_emails.slurm
# (500000) et run_sae_v10_ablation_tok100k.slurm (100000), donne 3 points pour
# évaluer si le taux d'interprétabilité (odd-one-out) continue de progresser avec
# le volume UNE FOIS le domaine corrigé, ou plafonne déjà à 500k -- réponse
# directe à la question "volume d'entraînement ou autre chose ?" posée par
# l'utilisateur. Même remarque que le bras 100k sur le coût de re-extraction
# (accepté pour éviter toute contamination de cache entre configs).
export SEED=42
export SAVE_DIR="./results_v10_ablation_tok2M/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_SIZE=12b
export DTYPE=bf16
export N_FEATURES_TO_LABEL=150
export N_TOKENS_EXTRA_TRAIN=2000000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
