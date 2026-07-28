#!/bin/bash
#SBATCH --job-name=SAE_abl_w262k
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_abl_width262k_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "largeur SEULE" — complète le balayage 16k/65k (run_ablation_
# width65k_only.slurm, job 40952) avec la troisième largeur GemmaScope-2-12b-it
# déjà présente localement (téléchargée pour la mesure de couverture Neuronpedia,
# cf. src/config.py commentaire _PRESETS : 262k -> 13851/262144 = 5,3% de
# couverture, très en dessous de 16k/65k). Question : la couverture Neuronpedia
# plus faible se traduit-elle aussi par un taux d'interprétabilité (juge LLM,
# indépendant de Neuronpedia) plus faible, ou seul le CATALOGUE de labels
# pré-existants est affecté (le juge odd-one-out génère ses propres labels quel
# que soit SAE_ID) ? Tout identique au run principal sauf SAE_ID.
export SEED=42
export SAVE_DIR="./results_v13_ablation_width262k_only/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_262k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export EPOCHS=30

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
