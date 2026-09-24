#!/bin/bash
#SBATCH --job-name=SAE_abl_c1b_frozen_cov
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_c1b_frozen_cov_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# C1b (pdf/sanitychecks.pdf, Korznikov et al. 2026) : réplique leur baseline
# Frozen Decoder sous le schéma d'initialisation PRINCIPAL de leur papier --
# "cov" (Gaussien de covariance réelle, puis normalisé), pas "iso" (Gaussien
# isotrope normalisé) utilisé par C1 (job 45953, résultats_v37). Leurs propres
# résultats publiés (Fig. 1, Tables 2-4) utilisent cov pour Frozen Decoder car
# empiriquement plus difficile à battre (Explained Variance 0,570 contre 0,430
# à configuration comparable) -- C1 sous-estimait donc la sévérité réelle de
# leur sanity check. Copie exacte de R0 (results_v27_.../layer31), SEULES
# variables : SANITY_CHECK_FROZEN_DECODER=1 (comme C1) et
# SANITY_CHECK_FROZEN_DECODER_INIT=cov (frozen_core.py::
# FrozenDecoderExtendedSAE::_reinit_decoder_cov). Cache d'extraction partagé
# réutilisé (ni SANITY_CHECK_FROZEN_DECODER ni son schéma d'init n'entrent
# dans la clé de cache) -- coût limité à l'entraînement + jugement.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v42_ablation_c1b_sanity_frozen_decoder_cov_init/"
export USE_FROZEN_CORE="1"
export SANITY_CHECK_FROZEN_DECODER="1"
export SANITY_CHECK_FROZEN_DECODER_INIT="cov"
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
