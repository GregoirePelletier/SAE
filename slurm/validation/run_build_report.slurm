#!/bin/bash
#SBATCH --job-name=SAE_build_report
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/validation/build_report_%j.log

# Régénération de report/dist/RAPPORT_DE_STAGE.md -- soumis via sbatch plutôt
# qu'exécuté sur le nœud frontal, cf. CLAUDE.md ("Cluster SLURM").

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

mkdir -p logs/validation

cd /home/h21486/SAE/
PYTHONPATH=. .venv/bin/python scripts/build_report.py
