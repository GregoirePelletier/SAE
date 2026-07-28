#!/bin/bash
#SBATCH --job-name=SAE_abl_dext2048
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_abl_d_extra2048_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "D_EXTRA seul" (dictionnaire de l'extension plus large, MÊME budget
# de parcimonie K_EXTRA=32) ─────────────────────────────────────────────────────
# L'ablation "capacité" déjà faite (job 40953, RESULTS_TESTS.md §17.5) a doublé
# D_EXTRA ET K_EXTRA ensemble (1024/32 -> 2048/64, ratio K/D=1/32 préservé) --
# jamais testé : plus d'atomes de dictionnaire disponibles SANS changer le budget
# de parcimonie par token (K_EXTRA=32 inchangé). Hypothèse : plus d'atomes parmi
# lesquels distribuer le même nombre de features actives par token pourrait
# produire des features plus spécialisées/monosémantiques (ratio K/D plus faible,
# 32/2048=1,6% contre 32/1024=3,1%), sans le coût d'apprentissage d'un budget de
# parcimonie plus large. Tout identique au run principal sauf D_EXTRA.
export SEED=42
export SAVE_DIR="./results_v13_ablation_d_extra2048_only/"
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
export D_EXTRA=2048
export K_EXTRA=32

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
