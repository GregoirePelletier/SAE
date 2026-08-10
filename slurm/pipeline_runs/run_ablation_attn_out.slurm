#!/bin/bash
#SBATCH --job-name=SAE_attnout
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_attnout_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Balayage hook-point (RESULTS_TESTS.md §36) : le projet n'a jamais chargé
# autre chose que resid_post. attn_out (entrée de self_attn.o_proj, layer 24,
# hf_hook_point_in vérifié sur le config.json GemmaScope-2 réel) via un forward
# hook (src/sae/saev5.py, HOOK_TYPE != resid_post) -- reste identique au run
# principal (16k, K_EXTRA=32, D_EXTRA=1024, N_FEATURES=150).
export HOOK_TYPE=attn_out
export SEED=42
export SAVE_DIR="./results_v13_ablation_attn_out/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=32

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
