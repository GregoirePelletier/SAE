#!/bin/bash
#SBATCH --job-name=SAE_abl_epochs
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=sae_abl_epochs_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "époques SEULES" ────────────────────────────────────────────────────
# Contrepartie de run_ablation_width65k_only.slurm : isole l'effet d'un
# entraînement plus long (EPOCHS_EXTRA 10->40, EPOCHS 30->100), largeur de SAE core
# INCHANGÉE (16k, comme le run principal results_v10_emails_main -- SAE_ID surchargé
# explicitement car le défaut de src/config.py a été relevé à 65k après le run v12).
# N_FEATURES_TO_LABEL laissé à 150 (défaut) pour rester un run à facteur UNIQUE,
# comparable terme à terme à results_v10_emails_main et à
# results_v12_ablation_width65k_only.
export SEED=42
export SAVE_DIR="./results_v12_ablation_epochs_only/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_16k_l0_medium"
export PIPELINES=p1,p2

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=40
export EPOCHS=100

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
