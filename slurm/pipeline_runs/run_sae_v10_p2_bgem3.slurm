#!/bin/bash
#SBATCH --job-name=SAE_p2_bgem3
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_runs/sae_v10_p2_bgem3_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── bge-m3 comme backbone Pipeline 2 (perspective #10, report/04, jamais fait) ──
# bge-m3 est déjà utilisé pour la similarité de labels (§15.2, RESULTS_TESTS.md --
# pooling [CLS], bien supérieur à F2LLM/pooling dernier-token sur des labels
# courts) mais jamais câblé comme backbone d'ENTRAÎNEMENT du PhraseLevelSAE
# lui-même. Nécessitait un vrai fix de code (pas juste un swap EMB_MODEL) :
# extract_f2llm_embeddings (src/sae/phrase_sae.py) faisait un pooling dernier-
# token EN DUR, adapté à un backbone décodeur causal (F2LLM) mais INCORRECT pour
# un backbone encodeur bidirectionnel entraîné pour le pooling [CLS] (bge-m3) --
# utiliser bge-m3 sans ce fix aurait produit des embeddings de phrase corrects
# mais sous-optimaux, sans jamais s'en apercevoir. Nouveau EMB_POOLING="cls"
# (src/config.py, défaut "last_token" préserve tout run F2LLM existant).
# NOUVEAU SAVE_DIR obligatoire (cache d'embeddings non namespacé par modèle,
# cf. run_sae_v10_p2_f2llm330m.slurm).
export SEED=42
export SAVE_DIR="./results_v10_p2_bgem3/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/bge-m3"
export EMB_POOLING="cls"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p2

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
