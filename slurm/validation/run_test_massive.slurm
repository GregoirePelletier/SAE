#!/bin/bash
#SBATCH --job-name=SAE_test_acts
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/test_massive_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
# Poids SAE réellement présents ici (local_data/saes/gemma-scope-2-12b-it est vide) :
export LOCAL_SAE_DIR="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res"

mkdir -p logs

cd /home/h21486/SAE/
# uv run tentait de re-résoudre l'env (réseau bloqué sur le cluster) -> échec
# systématique ; .venv/bin/python (déjà provisionné) bypass ce problème, comme
# dans run_sae.slurm.
.venv/bin/python scripts/test_massive_acts.py