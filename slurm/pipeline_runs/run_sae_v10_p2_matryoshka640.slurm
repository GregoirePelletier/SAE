#!/bin/bash
#SBATCH --job-name=SAE_p2_mrl640
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_runs/sae_v10_p2_mrl640_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Balayage MATRYOSHKA_DIM (jamais fait) : F2LLM-v2-160M (hidden_size=640) est
# le seul backbone local pour lequel MATRYOSHKA_DIM=320 (défaut) est une VRAIE
# troncature (pour F2LLM-80M, hidden_size=320 == MATRYOSHKA_DIM par défaut : la
# troncature était un no-op pour ce backbone dans toutes les comparaisons
# précédentes, jamais remarqué). Ni la doc F2LLM (README HF, vérifié) ni ce
# projet n'ont jamais confirmé que F2LLM-v2 est entraîné avec un objectif
# Matryoshka (MRL) au sens strict (perte imbriquée garantissant qu'un préfixe
# de dimensions reste un bon embedding) -- ce run teste empiriquement si tronquer
# à 64/640 dégrade en douceur (cohérent avec MRL) ou de façon abrupte (naive
# troncature sans objectif d'entraînement dédié). Point à dim=640 (aucune troncature, embedding complet), comparé aux
# points déjà obtenus à dim=320 (results_v10_p2_f2llm160m) et dim=128/640
# (jobs frères de ce script). SAVE_DIR neuf (cache d'embeddings namespacé par
# MATRYOSHKA_DIM dans le nom de fichier -- pas de risque de contamination, mais
# SAVE_DIR neuf pour ne pas mélanger les checkpoints PhraseLevelSAE non plus).
export SEED=42
export SAVE_DIR="./results_v10_p2_mrl640/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-160M"
export MATRYOSHKA_DIM=640

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p2

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
