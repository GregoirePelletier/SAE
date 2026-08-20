#!/bin/bash
#SBATCH --job-name=SAE_val100k_v4
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val100k_l24_v4_batchsize16384_%j.log
# --signal=B:USR1@600 : cf. run_validation_25m_layer24.slurm (R1, AUDIT_SAE_2026-08.md §2.3/§4.3).

# Ablation BATCH_SIZE_EXTRA (AUDIT_SAE_2026-08.md §2.9, item 7) : réplique
# EXACTE de run_validation_100k_layer24_v3_extractionfillerfix.slurm (job
# 44572, mêmes correctifs filler, même code), SEUL BATCH_SIZE_EXTRA change
# (1024 par défaut -> 16384 ici). Sous BatchTopK (src/sae/batch.py), le
# budget top-k est PARTAGÉ sur le batch aplati : ce n'est pas juste un
# facteur de vitesse, la taille de batch change le régime de sparsité vu à
# l'entraînement -- d'où l'ablation avant tout changement de défaut plutôt
# qu'un correctif de perf appliqué à l'aveugle.
# SAVE_DIR séparé -- pas de partage de cache avec 44560/44571/44572.
# Comparaison prévue une fois ce run et 44572 (v3, BATCH_SIZE_EXTRA=1024)
# terminés tous les deux : rho_sae, fve_pretrained, dead_pct, L0 final,
# loss train/val (results.json + p1_extended_sae_history.json de chaque
# SAVE_DIR), plus le ratio de temps d'entraînement extra (history["step"]
# vs wall-clock du log). Limite connue avant même de lancer : n=1 seed par
# bras, donc un signal directionnel seulement -- pas une conclusion
# statistique au sens de RESULTS_TESTS.md §45 (qui utilise 3 graines pour
# K_EXTRA=5).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v18_validation_100k_layer24_v4_batchsize16384/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p1

export LAYER=24
export SAE_ID="layer_24_width_16k_l0_medium"
export K_EXTRA=32

export N_FEATURES_TO_LABEL=20
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export N_TOKENS_EXTRA_TRAIN=100000
export N_VOLUME_FILLER_TARGET_CHUNKS=3000
export BATCH_SIZE_EXTRA=16384

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
