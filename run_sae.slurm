#!/bin/bash
#SBATCH --job-name=SAE_Dual_Pipeline
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=16:00:00
#SBATCH --output=sae_v8_%j.log

# 1. Isolation réseau totale
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 2. Variables applicatives
export SEED=42
export SAVE_DIR="./results_v8/"
export USE_FROZEN_CORE="1"

# 3. Pipeline 1 — SAE plus large, même bande L0 (isole largeur / calibration seuil)
export MODEL_SIZE=12b
export SAE_ID="layer_24_width_262k_l0_medium"

# 4. Volume corpus ×2.5 — re-encode (poste dominant, linéaire, 2.39 it/s) ~2h10, extraction ~58min
export N_TOTAL_ENERGY=5000
export N_TOTAL_SPORTS=5000
export N_TOTAL_SUPPORT=5000
export N_TOKENS_EXTRA_TRAIN=750000

# 5. Frozen core inchangé (D_EXTRA/K_EXTRA) — isolation des variables

# 6. Pipeline 2 — dead=94.6% stable sur 20 epochs (plafond structurel, pas de convergence) : réduire d_SAE
export D_SAE=2048
export K_SPARSE=16
export EPOCHS=60

# 7. Matière pour la présentation
export N_FEATURES_TO_LABEL=30

# 8. Exécution locale
cd /home/h21486/SAE/
uv run saev5.py