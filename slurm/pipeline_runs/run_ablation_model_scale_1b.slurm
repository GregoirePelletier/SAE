#!/bin/bash
#SBATCH --job-name=SAE_abl_1b
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_abl_model_scale_1b_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "échelle du modèle" : gemma-3-1b-it à la place de gemma-3-12b-it ──
# Même protocole que run_ablation_model_scale_4b.slurm (cf. son commentaire pour
# le détail du raisonnement) -- ce run teste le point le plus extrême de
# réduction d'échelle disponible avec un GemmaScope officiel (hors 270m, profil
# de validation rapide à capacité de juge limitée). gemma-3-1b-it + gemma-scope-2-1b-it
# (resid_post/layer_13_width_16k_l0_medium, d_model=1152) téléchargés pour ce
# run. PIPELINES=p1 uniquement.
export SEED=42
export SAVE_DIR="./results_v13_ablation_model_scale_1b/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-1b-it"
export MODEL_SIZE=1b
export DTYPE=bf16
export SAE_ID="layer_13_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=32

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
