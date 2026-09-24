#!/bin/bash
#SBATCH --job-name=SAE_confdom_r0
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_confdom_r0_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Rejeu du diagnostic domaine-vs-volume (§12/§46) sous méthodologie R0 --
# RESULTS_TESTS.md §120 (audit de conformité R0 du rapport de stage). §12/§46
# comparaient corpus générique (energy/sports/support) et corpus emails sous
# sélection par magnitude, juge Gemma-3-12b-it auto-référent, négatif non
# corrigé, K_EXTRA=32/layer 24/500k tokens (results_v9_full, purgé) -- un
# protocole non R0. Ce run réutilise CONFIRMATORY_DOMAIN_BASELINE=1 (déjà
# présent dans src/sae/saev5.py, comportement inchangé sinon) avec les
# hyperparamètres R0 exacts (K_EXTRA=5, D_EXTRA=1024, layer 31, 25M tokens,
# n=300, stratifié + Qwen3.8-27B + négatif corrigé -- tous défauts du dépôt
# depuis §117/§119) pour rendre la comparaison au bras emails (R0,
# results_v27_ablation_classic_setup_k5_25m_layer31/, 65,7%) directement
# comparable plutôt que confondue avec un changement d'hyperparamètres.
# Pas de déduplication par mail parent pertinente ici (train_groups=None,
# pas de notion de "mail d'origine" dans le corpus générique) -- écart
# structurel assumé, pas un oubli.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export CONFIRMATORY_DOMAIN_BASELINE=1

export SEED=42
export SAVE_DIR="./results_v40_confirmatory_domain_baseline_r0_setup/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="${SAE_ROOT}/models/F2LLM-v2-80M"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=300
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=25000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py
