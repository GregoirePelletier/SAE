#!/bin/bash
#SBATCH --job-name=SAE_baseline_v2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_v2_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_SAE_DIR="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res"

# NOUVEAU CACHE_DIR (pas de partage avec cache_baseline_full) : baseline_gemmascope.py
# réutilise silencieusement baseline_doc_acts_*.pt si présent (encode_corpus) -- un
# cache partagé rejouerait les activations calculées AVANT le fix du biais Objet:/
# Subject: (load_augmented, cf. commit "Corrige le biais résiduel Objet:/Subject:").
# Objectif de ce run : mesurer l'effet du fix sur les features les plus
# discriminantes par axe (avant : "Subject: followed by email subject lines"
# ressortait comme top feature sur plusieurs axes, cf. RESULTS_TESTS.md §6/§0).
export SAVE_DIR="./results_v11_baseline_objetfix/"
export CACHE_DIR="${SAVE_DIR}cache_baseline_full"

mkdir -p logs "${CACHE_DIR}"

cd /home/h21486/SAE/
.venv/bin/python scripts/baseline_gemmascope.py "${LOCAL_MAILS_PATH}" "${LOCAL_AUGMENTED_MAILS_PATH}"
