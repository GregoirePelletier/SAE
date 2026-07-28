#!/bin/bash
#SBATCH --job-name=SAE_p2_160m
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_runs/sae_v10_p2_f2llm160m_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Complète le balayage de backbone Pipeline 2 (§16.5 : F2LLM-v2-80M vs -330M,
# résultat mixte -- 330M légèrement meilleur sur NMSE/acc_SAE mais pas
# uniformément) avec la taille intermédiaire F2LLM-v2-160M, présente localement
# mais jamais exercée dans ce projet. Rerun Pipeline 2 UNIQUEMENT (pas de
# Gemma-3-12B/GemmaScope -- P1 déjà validé dans results_v10_emails_main).
# NOUVEAU SAVE_DIR obligatoire : extract_f2llm_embeddings met en cache par nom de
# fichier dérivé de MATRYOSHKA_DIM/n_phrases uniquement, PAS du modèle -- réutiliser
# un cache existant rechargerait silencieusement les embeddings d'un autre backbone
# (contamination de cache, cf. commentaire de run_sae_v10_p2_f2llm330m.slurm).
export SEED=42
export SAVE_DIR="./results_v10_p2_f2llm160m/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-160M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p2

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
