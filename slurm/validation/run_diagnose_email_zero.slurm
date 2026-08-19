#!/bin/bash
#SBATCH --job-name=diag_email_zero
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/validation/diagnose_email_zero_%j.log

export PYTHONUNBUFFERED=1

# Diagnostic CPU-only (aucun GPU demandé) : inspecte le pickle sauvegardé par
# le job 44413 (0/65536 features actives rapporté -- trop extrême pour être
# plausible, cf. scripts/diagnose_email_zero_activations.py).

cd /home/h21486/SAE/
/home/h21486/SAE/external/interp_embed/.venv/bin/python \
    /home/h21486/SAE/scripts/diagnose_email_zero_activations.py
