#!/bin/bash
#SBATCH --job-name=SAE_abl_classic_k5_27b
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=700G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_classic_k5_25m_27b_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Ablation taille du modèle extracteur/juge, palier 27B (setup classique du
# papier, K_EXTRA=5, D_EXTRA=1024) -- étend le sweep 1B/4B/12B/27B. Layer 40
# = ~2/3 profondeur (62 couches, gemma-scope-2-27b-it publie resid_post curé
# à {16,31,40,53} soit ~25/50/65/85%, mêmes proportions que 12/24/31/41 sur
# 48 couches pour 12B -- cf. src/config.py::_PRESETS["27b"]).
# --mem=700G : d_model=5376 (vs 3840 pour 12B, ratio x1,40) -- le 25M run
# 12B/layer31 mesuré à 382 Go RSS réel (job 44994) suggère un ordre de
# grandeur ~535 Go pour 27B à même volume de tokens ; marge conservatrice.
# a100 (40 Go VRAM/carte) EXCLU : les poids seuls (bf16, ~54 Go) ne tiennent
# pas -- H100 (80 Go/carte) uniquement, aucune course de doublon possible ici.
# Le juge d'interprétabilité (odd-one-out) reste Gemma-3-12B-it (MODEL_ID
# séparé dans judge.py, non affecté par ce MODEL_SIZE qui ne pilote que
# l'EXTRACTEUR/SAE core P1) -- indépendance du juge inchangée par cette
# ablation, seule la taille du modèle qui PRODUIT les activations varie.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v31_ablation_classic_setup_k5_25m_model_scale_27b/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="${SAE_ROOT}/models/F2LLM-v2-80M"

export MODEL_SIZE=27b
# Copie partagée déjà présente sur le cluster (/mnt/gvd/modeles_ia/), 52 Go,
# 12/12 shards vérifiés -- pas de copie personnelle redondante (download_sae.py
# en avait déjà téléchargé une avant que cette copie soit trouvée, supprimée).
export MODEL_ID="/mnt/gvd/modeles_ia/gemma-3-27b-it"
export DTYPE=bf16
export LAYER=40
export SAE_ID="layer_40_width_16k_l0_medium"
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
