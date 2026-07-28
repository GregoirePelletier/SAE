#!/bin/bash
#SBATCH --job-name=SAE_abl_4b
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_abl_model_scale_4b_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "échelle du modèle" : gemma-3-4b-it à la place de gemma-3-12b-it ──
# Toutes les ablations précédentes (largeur, époques, capacité, volume, seed,
# K_EXTRA) gardent le MODÈLE fixé à gemma-3-12b-it -- jamais testé si le taux
# d'interprétabilité (et la qualité des labels produits) dépend de l'échelle du
# modèle extracteur/juge lui-même. gemma-3-4b-it + son propre GemmaScope
# (gemma-scope-2-4b-it, resid_post/layer_17_width_16k_l0_medium, d_model=2560)
# téléchargés pour ce run (config déjà prévue dans src/config.py::_PRESETS,
# jamais exercée avant). Le juge d'auto-interprétation change AUSSI de modèle en
# même temps que l'extracteur (MODEL_ID partagé par les deux rôles dans
# saev5.py) -- confond donc "qualité des features" et "qualité du juge", comme
# c'est le cas structurellement pour toute paire modèle+SAE GemmaScope adaptée.
# Neuronpedia n'héberge pas de labels pour layer 17/16k de ce modèle (catalogue
# "Core" affichera F{idx} bruts) -- sans impact sur le taux d'interprétabilité
# (features de l'extension, auto-labellisées par le juge, indépendantes de
# Neuronpedia). PIPELINES=p1 uniquement (Pipeline 2/F2LLM est indépendant de la
# taille du modèle Gemma, aucune information supplémentaire à en tirer ici).
export SEED=42
export SAVE_DIR="./results_v13_ablation_model_scale_4b/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-4b-it"
export MODEL_SIZE=4b
export DTYPE=bf16
export SAE_ID="layer_17_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=32

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
