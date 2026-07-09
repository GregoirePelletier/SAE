#!/bin/bash
#SBATCH --job-name=SAE_test_acts
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_massive_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

mkdir -p logs

cd /home/h21486/SAE/
uv run python test_massive_acts.py