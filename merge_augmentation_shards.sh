#!/bin/bash
# Fusionne les 8 shards produits par run_augmentation_full.slurm en un seul
# augmented_mails.jsonl consommable par run_baseline.slurm. À lancer une fois que
# les 8 tâches de l'array sont COMPLETED (squeue -u $USER pour vérifier).
set -euo pipefail
cd /home/h21486/SAE/
SAVE_DIR="./results_v9_test/"
OUT="${SAVE_DIR}augmented_mails.jsonl"

shopt -s nullglob
shards=("${SAVE_DIR}"augmented_mails_shard*of8.jsonl)
if [ "${#shards[@]}" -ne 8 ]; then
    echo "ATTENTION : ${#shards[@]}/8 shards trouvés (attendu 8) : ${shards[*]}" >&2
fi

cat "${shards[@]}" > "$OUT"
echo "Fusion : $(wc -l < "$OUT") lignes -> $OUT"
echo "Acceptées : $(grep -c '"rejected": null' "$OUT")"
