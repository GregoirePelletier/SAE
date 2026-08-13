#!/bin/bash
#SBATCH --job-name=SAE_pytest_docs
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/validation/pytest_check_docs_%j.log

# Smoke-test CPU-only (tests unitaires + garde-fou documentaire) -- soumis via
# sbatch plutôt qu'exécuté sur le nœud frontal, cf. CLAUDE.md ("Cluster SLURM").

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

mkdir -p logs/validation

cd /home/h21486/SAE/
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/check_docs.py
