#!/bin/bash
#SBATCH --job-name=SAE_abl_k5
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_abl_k5_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "K_EXTRA faible" (SAE Boost, Koriagin et al., COLM 2025,
# arXiv:2507.12990) ─────────────────────────────────────────────────────────────
# Piste explicitement flaguée comme non testée en RESULTS_TESTS.md §18.2 : l'étude
# de sensibilité du papier (leur Table 12) trouve k=5 optimal pour le compromis
# interprétabilité/EV-domaine d'un SAE résiduel, contre notre K_EXTRA=32 par défaut
# -- jamais testé EN DESSOUS de cette valeur (l'ablation capacité, §17.5, a testé
# K_EXTRA=64, direction opposée). Ce run reproduit EXACTEMENT les conditions du run
# principal (results_v10_emails_main : 16k, N_FEATURES_TO_LABEL=150, EPOCHS_EXTRA=10,
# D_EXTRA=1024, SEED=42, CORPUS_SPLIT_SEED=42 par défaut) mais avec K_EXTRA=5.
# Pas de réutilisation du cache du run principal (cf. discussion dans
# RESULTS_TESTS.md §18.3bis) : même pattern que l'ablation seed-variance
# (run_ablation_seed_variance.slurm) -- un SAVE_DIR neuf, pipeline complet
# rejoué, pour éviter tout risque de collision sur cache/p1_judge_labels_extended.json
# (labels spécifiques aux features de CE K_EXTRA, pas interchangeables avec
# celles de K_EXTRA=32). Coût attendu ~3h (cf. job 41118, même taille de corpus).
export SEED=42
export SAVE_DIR="./results_v13_ablation_k_extra5/"
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
export K_EXTRA=5

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
