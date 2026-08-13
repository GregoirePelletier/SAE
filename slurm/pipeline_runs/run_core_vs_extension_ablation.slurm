#!/bin/bash
#SBATCH --job-name=SAE_core_vs_ext
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pipeline_runs/core_vs_extension_ablation_%j.log

# ── Phase 1 (diagnostics/reproduction initiative) : core seul vs core+extension
# ("pollution" de l'extension résiduelle sur le signal du core) ─────────────────
# Script CPU-only (aucun GPU requis, réutilise les activations denses déjà en
# cache du run principal) -- soumis en SLURM plutôt qu'exécuté sur le nœud de
# login car ce dernier est partagé et déjà proche de la saturation mémoire
# (observé : ~23/31 Go utilisés, swap quasi plein) : un run silencieusement
# OOM-killé sur le nœud de login n'écrit ni traceback ni artefact, cf. le
# premier essai de ce script qui s'est arrêté sans erreur visible pour cette
# raison. --mem=48G largement suffisant (tenseur en cache = 3 Go, copies
# numpy par condition ≈ 3 Go chacune, MaxRSS mesuré ≈ 11 Go sur le 1er essai)
# sans sur-demander sur un nœud à 1 To.
#
# CORRECTION post-mortem (job 43006, --cpus-per-task=4/--time=00:30:00) :
# TIMEOUT à 30min alors que MaxRSS ne dépassait que 11 Go -- job compute-bound,
# pas memory-bound. Cause : LogisticRegression multinomiale (solveur lbfgs,
# probe 14 classes axes email) sur une matrice dense ~41176×17408 sans
# parallélisme BLAS (4 CPUs, threads BLAS non fixés explicitement). Corrigé :
# --cpus-per-task=32 + threads BLAS explicitement alignés dessus (sinon
# certains backends par défaut sur-souscrivent ou restent mono-thread).
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/core_vs_extension_ablation.py
