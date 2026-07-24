#!/bin/bash
#SBATCH --job-name=SAE_abl_seed
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_abl_seed_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "variance de seed" (Unstable Features, Reproducible Subspaces,
# arXiv:2606.12138 ; Toward Identifiable Sparse Autoencoders, arXiv:2605.31245) ──
# Ces travaux montrent que les features individuelles d'un SAE varient
# substantiellement d'un seed d'entraînement à l'autre (seule la sous-structure de
# rang faible qu'elles occupent est reproductible) -- notre projet n'a jamais testé
# cet axe : toutes les ablations précédentes (largeur, époques, capacité, volume)
# utilisent SEED=42 partout. Ce run reproduit EXACTEMENT les conditions du run
# principal (results_v10_emails_main : 16k, N_FEATURES_TO_LABEL=150, EPOCHS_EXTRA=10,
# D_EXTRA=1024/K_EXTRA=32) mais avec SEED=123 (init des poids de l'extension,
# échantillonnage de feature_selection_by_magnitude, etc.). CORPUS_SPLIT_SEED
# n'est PAS modifié (reste à son défaut 42, cf. src/config.py) : le split train/test
# du corpus emails+augmentés reste IDENTIQUE au run principal -- seule la variance
# d'entraînement du SAE est isolée, pas une variance de corpus.
export SEED=123
export SAVE_DIR="./results_v13_ablation_seed123/"
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
