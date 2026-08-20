#!/bin/bash
#SBATCH --job-name=extract_saeboost_pdf
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/validation/extract_saeboost_pdf_%j.log

# CPU-only, aucun GPU requis -- extraction de texte pypdf, quelques secondes.
export PYTHONUNBUFFERED=1
cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_extract_saeboost_pdf_text.py
