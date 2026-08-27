#!/bin/bash
#SBATCH --job-name=SAE_probe_50m_l31
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1200G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_probe_50m_layer31_%j.log

# Point de mesure UNIQUE (pas un sweep) : 50M tokens, 12B/layer31/K_EXTRA=5,
# pour obtenir un chiffre RAM réel avant de décider d'un sweep 50M/100M
# complet -- l'extrapolation en loi de puissance faite à partir de deux
# points seulement (5M -> 31,3 Go RSS job 44991 ; 25M -> 381,8 Go RSS job
# 44994) n'est pas fiable pour un facteur x2 hors de la plage mesurée,
# demande explicite de vérifier empiriquement plutôt que de faire confiance
# à l'extrapolation. --mem=1200G : marge large sous les 2,02 To du nœud
# dgx-h100 (sinfo -p h100 -N -o "%N %m"), largement au-dessus de
# l'extrapolation prudente (~760 Go en loi linéaire, plus si la croissance
# reste super-linéaire) -- si ce budget s'avère lui-même insuffisant, le job
# sera tué par cgroup (MaxRSS visible malgré tout via `sacct`, donne un
# signal même en cas d'échec). N_VOLUME_FILLER_TARGET_CHUNKS=1200000 :
# le corpus réel (~14,6M tokens estimés) ne couvre qu'une fraction de la
# cible, le filler comble le reste (~35M tokens) -- pool dimensionné avec
# marge (cf. heuristique déjà utilisée pour la tentative 100M abandonnée,
# run_validation_100M_layer31_h100.slurm, ~21 800 chunks/M tokens cible).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEED=42
export SAVE_DIR="./results_probe_50m_layer31/"
export USE_FROZEN_CORE="1"
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
export N_TOKENS_EXTRA_TRAIN=50000000
export N_VOLUME_FILLER_TARGET_CHUNKS=1200000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
