#!/bin/bash
#SBATCH --job-name=SAE_abl_c1_frozen
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_c1_frozen_%j.log

# C1 (AUDIT_SAE_2026-08.md, campagne finale) : sanity check Korznikov et al. (décodeur
# W_dec_extra figé à une init ALÉATOIRE, jamais entraîné) rejoué sous méthodologie
# finale (sélection stratifiée + juge Qwen3.8-27B, tous deux DÉFAUTS du dépôt --
# FEATURE_SELECTION_METHOD/JUDGE_MODEL_ID, src/config.py -- aucun flag supplémentaire
# requis ici). L'ancien sanity check (results_v12_sanity_frozen_decoder, 45,3% vs
# 29,3%) datait de la sélection par magnitude + juge auto-référent, les deux biais
# corrigés depuis -- personne ne sait ce que donne un décodeur aléatoire sous le
# protocole retenu comme référence de ce rapport.
#
# Copie exacte de R0 (run_ablation_classic_setup_k5_25m_layer31.slurm,
# results_v27_ablation_classic_setup_k5_25m_layer31), UNE seule variable en plus :
# SANITY_CHECK_FROZEN_DECODER=1. Extraction (LAYER/N_TOKENS_EXTRA_TRAIN/corpus
# identiques à R0) réutilisée depuis le cache partagé déjà écrit par R0 -- ne coûte
# que l'entraînement SAE (décodeur figé, seuls W_enc_extra/b_enc_extra/seuil appris)
# + le jugement, pas une nouvelle extraction.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v37_ablation_c1_sanity_frozen_decoder_stratified_qwen/"
export USE_FROZEN_CORE="1"
export SANITY_CHECK_FROZEN_DECODER="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=25000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
