#!/bin/bash
#SBATCH --job-name=SAE_p2_stratified_qwen
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pipeline_runs/sae_v10_p2_stratified_qwen_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Rerun Pipeline 2 UNIQUEMENT sous le protocole actuel (sélection stratifiée,
# B.2/§79, jamais portée sur P2 avant ce job -- feature_selection_stratified_
# by_frequency_dense, judge.py -- et juge Qwen par défaut, JUDGE_MODEL_ID,
# jamais utilisé sur P2 avant non plus). L'ancien p2_feature_labels.json
# (17 juillet, magnitude pure, juge Gemma auto-référent, protocole odd-one-out
# d'avant le correctif B.5) est daté ; le seul chiffre d'interprétabilité P2
# du dépôt reposait dessus sans jamais avoir été recalculé. SAVE_DIR
# INCHANGÉ (results_v10_emails_main) : réutilise le cache d'embeddings
# F2LLM-v2-80M déjà présent (train/test/diffcorpus_phrase_emb_dim320_*.pt,
# même backbone/MATRYOSHKA_DIM/corpus -- clé de cache d'extract_f2llm_
# embeddings dérivée de ces paramètres, pas du protocole de jugement) --
# évite de refaire l'extraction F2LLM, seul le PhraseLevelSAE (checkpoint
# supprimé par le nettoyage disque) est réentraîné.
export SEED=42
export SAVE_DIR="./results_v10_emails_main/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="${SAE_ROOT}/models/F2LLM-v2-80M"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p2

export FEATURE_SELECTION_METHOD=stratified
export N_FEATURES_TO_LABEL=150

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py
