#!/bin/bash
#SBATCH --job-name=SAE_audit_p1
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validation/audit_palier1_batch_%j.log

# Audit 2026-08 (docs/AUDIT_2026-08.md), vérifications Palier 1 groupées
# (B.26, B.6, B.1 point 1) : CPU-only, activations/checkpoints déjà en
# cache (results_v10_emails_main/). Soumis en job plutôt qu'exécuté sur le
# nœud de login, cf. règle CLAUDE.md (RAM frontale partagée entre tous les
# utilisateurs) -- même motivation que run_core_vs_extension_ablation.slurm.
#
# CORRECTION post-mortem (job 43216, --cpus-per-task=16/--time=00:45:00) :
# job tué avant complétion, aucune progression stdout visible après 33min --
# même cause exacte que run_core_vs_extension_ablation.slurm (StratifiedKFold/
# GroupKFold lbfgs multinomial sur matrice dense ~41176x17408, 14+ classes,
# compute-bound). Alignement sur la config qui a résolu ce cas précédemment :
# --cpus-per-task=32, --time=02:00:00.
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_palier1_batch.py
