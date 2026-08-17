#!/bin/bash
#SBATCH --job-name=SAE_b24_inspect
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/validation/audit_b24_inspect_pollution_%j.log

# Audit 2026-08, B.24 (suite) : inspection pure lecture (pandas sur parquets
# deja produits, aucun GPU, aucun recalcul) du resultat "0 flagged" du job
# principal -- verifie si c'est un vrai null ou un artefact de petit
# echantillon (meme classe de probleme que B.27).
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_b24_inspect_pollution.py
