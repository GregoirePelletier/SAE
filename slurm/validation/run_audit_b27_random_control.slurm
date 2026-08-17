#!/bin/bash
#SBATCH --job-name=SAE_b27_ctrl
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/audit_b27_random_control_%j.log

# Audit 2026-08, B.27 : temoin manquant pour
# feature_group_reproducibility_test.py -- regroupement ALEATOIRE (meme
# distribution de tailles que les communautes Louvain reelles) pour isoler
# l'effet geometrique du moyennage de l'effet semantique pretendu.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_b27_random_control.py
