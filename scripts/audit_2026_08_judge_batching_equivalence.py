"""
scripts/audit_2026_08_judge_batching_equivalence.py — Vérifie le correctif
judge batching de AUDIT_SAE_2026-08.md §2.6 (item 1) : src/sae/judge.py::
_batched_generate doit produire, pour un lot de prompts indépendants, EXACTEMENT
les mêmes réponses décodées qu'un appel séquentiel (batch_size=1) prompt par
prompt -- do_sample=False rend chaque génération individuelle déterministe,
donc toute différence entre les deux chemins vient d'un bug de padding/masque
d'attention, pas d'un aléa.

Deux questions, dans cet ordre :
1. ÉQUIVALENCE : réponses décodées identiques, batché (bs>1) vs séquentiel
   (bs=1), sur des prompts de longueurs délibérément hétérogènes (le cas qui
   exercerait le plus un bug de padding_side/attention_mask).
2. GAIN RÉEL : temps total, séquentiel vs batché, sur le MÊME modèle chargé
   une seule fois (le juge de production, Gemma-3-12B-it, bf16).

Indépendant de tout run en cours -- n'écrit rien dans le cache de production.

Usage : sbatch slurm/validation/run_audit_judge_batching_equivalence.slurm
"""
from __future__ import annotations

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_ID, HF_TOKEN
from src.sae.judge import _batched_generate

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16
OUT_PATH = "docs/audit_2026_08_judge_batching_equivalence_results.json"

# Prompts synthétiques de longueurs hétérogènes -- reproduit la variance réelle
# (exemples courts/longs par feature) qui exercerait le plus un bug de padding.
_TOPICS = [
    "la facturation d'électricité", "un changement de compteur Linky",
    "une réclamation sur un délai d'intervention", "la résiliation d'un contrat",
    "une question sur les heures creuses", "un défaut de paiement",
    "la mise en service d'un nouveau logement", "un problème de relève de compteur",
    "une demande de remboursement", "l'ouverture d'un compte client",
    "une panne de courant signalée", "un changement d'adresse",
    "une question sur l'offre verte", "un litige de facturation double",
    "une demande de mensualisation", "un déménagement",
]


def _build_prompts(n: int) -> list[list[dict]]:
    prompts = []
    for i in range(n):
        topic = _TOPICS[i % len(_TOPICS)]
        # Longueur variable : répète le sujet 1 à 4 fois pour faire varier la
        # longueur du prompt d'une ligne à l'autre du lot.
        repeat = 1 + (i % 4)
        text = (
            f"Voici {repeat} exemples de mails clients à propos de {topic}. "
            "Résume en une phrase le thème commun. " * repeat
        )
        prompts.append([{"role": "user", "content": text}])
    return prompts


def main():
    print(f"[judge-batch] Chargement {MODEL_ID} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    n_prompts = 24
    prompts = _build_prompts(n_prompts)
    print(f"[judge-batch] {n_prompts} prompts synthétiques, longueurs hétérogènes.")

    # --- 1. Séquentiel (bs=1), référence -- chemin déjà en production avant ce correctif ---
    print("[judge-batch] Passe 1/2 : séquentiel (bs=1, référence)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sequential_responses = _batched_generate(model, tokenizer, prompts, max_new_tokens=32, batch_size=1)
    torch.cuda.synchronize()
    elapsed_sequential = time.perf_counter() - t0
    print(f"[judge-batch]   {elapsed_sequential:.1f}s")

    # --- 2. Batché (bs=8) ---
    print("[judge-batch] Passe 2/2 : batché (bs=8)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batched_responses = _batched_generate(model, tokenizer, prompts, max_new_tokens=32, batch_size=8)
    torch.cuda.synchronize()
    elapsed_batched = time.perf_counter() - t0
    print(f"[judge-batch]   {elapsed_batched:.1f}s")

    # --- 3. Équivalence stricte, texte par texte ---
    mismatches = [
        (i, seq, bat) for i, (seq, bat) in enumerate(zip(sequential_responses, batched_responses))
        if seq != bat
    ]
    exactly_equal = len(mismatches) == 0
    speedup = elapsed_sequential / elapsed_batched

    print("\n" + "=" * 70)
    print(" RÉSUMÉ — batching du juge (bs=1 vs bs=8)")
    print("=" * 70)
    print(f"  Équivalence stricte (texte par texte) : {exactly_equal} ({len(mismatches)}/{n_prompts} désaccords)")
    print(f"  Temps séquentiel / batché             : {elapsed_sequential:.1f}s / {elapsed_batched:.1f}s")
    print(f"  Speedup                               : {speedup:.2f}x")
    if mismatches:
        for i, seq, bat in mismatches[:5]:
            print(f"  [désaccord #{i}] séquentiel={seq!r} | batché={bat!r}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_prompts": n_prompts, "model_id": MODEL_ID,
            "equivalence": {"exactly_equal": exactly_equal, "n_mismatches": len(mismatches),
                            "mismatches": [{"index": i, "sequential": s, "batched": b} for i, s, b in mismatches]},
            "sequential": {"elapsed_s": elapsed_sequential},
            "batched": {"elapsed_s": elapsed_batched, "batch_size": 8},
            "speedup": speedup,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[judge-batch] Résultats -> {OUT_PATH}")

    if not exactly_equal:
        raise SystemExit(
            "ÉCHEC : le batching change les réponses générées pour au moins un prompt -- "
            "NE PAS faire confiance à _batched_generate tant que ce désaccord n'est pas expliqué."
        )


if __name__ == "__main__":
    main()
