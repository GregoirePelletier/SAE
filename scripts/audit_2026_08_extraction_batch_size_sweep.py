"""
scripts/audit_2026_08_extraction_batch_size_sweep.py — Étape 1 du plan
(sharded-twirling-sunbeam.md) : benchmark de `EXTRACTION_BATCH_SIZE`
(src/config.py, nouvellement configurable -- 4 codé en dur avant cette
session) sur un échantillon réel, GPU chargé UNE SEULE FOIS, pour trouver la
plus grande valeur qui tient en VRAM sans OOM sur H100/H100-bis (80 Go),
avant d'investir des heures d'extraction complète à la valeur par défaut (4).

Mesure le forward-pass Gemma-3-12B lui-même (le coût dominant, GPU-bound),
même signature exacte que la boucle d'extraction de production (saev5.py,
"Extraction P1") : output_hidden_states=True, logits_to_keep=1,
hidden_states[LAYER]. Ne réplique PAS le post-traitement par document
(masquage, encodage SAE core, écriture de fragments CSR) -- coût CPU/IO
largement indépendant de la taille de batch, pas le facteur limitant pour
cette question précise (taille de batch max sans OOM + débit du forward).

N'écrit RIEN dans le cache de production (aucun résidu, aucun fragment) --
purement un benchmark, indépendant du job 44211 en cours (qui, lui, écrit le
cache réel à réutiliser pour de futures comparaisons K_EXTRA, cf. discussion).

Usage : sbatch slurm/validation/run_audit_extraction_batch_sweep.slurm
"""
from __future__ import annotations

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_ID, HF_TOKEN, LAYER, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED
from src.data.preparation import build_email_train_test_corpus

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16
CANDIDATE_BATCH_SIZES = [4, 8, 16, 24, 32, 48, 64]
N_WARMUP_BATCHES = 2
N_TIMED_BATCHES = 8
MAX_LENGTH = 512
OUT_PATH = "docs/audit_2026_08_extraction_batch_size_sweep_results.json"


def main():
    print(f"[batch-sweep] Chargement {MODEL_ID} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()
    print(f"[batch-sweep] LAYER={LAYER} (même config que job 44211, résultats représentatifs de ce run)")

    print("[batch-sweep] Chargement d'un échantillon de textes réels (corpus emails+augmentés)...")
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    max_needed = max(CANDIDATE_BATCH_SIZES) * (N_WARMUP_BATCHES + N_TIMED_BATCHES)
    sample_texts = train_texts[:max_needed]
    print(f"[batch-sweep] {len(sample_texts)} textes disponibles pour le balayage.")

    results = {}
    for bs in CANDIDATE_BATCH_SIZES:
        n_needed = bs * (N_WARMUP_BATCHES + N_TIMED_BATCHES)
        if n_needed > len(sample_texts):
            print(f"[batch-sweep] batch_size={bs} : échantillon insuffisant ({n_needed} requis, "
                  f"{len(sample_texts)} dispo) -- arrêt du balayage ici.")
            break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batches = [sample_texts[i:i + bs] for i in range(0, n_needed, bs)]
            with torch.no_grad():
                # Warmup (cudnn/cublas autotuning, allocateur CUDA) -- exclu du chronométrage.
                for batch in batches[:N_WARMUP_BATCHES]:
                    enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
                    out = llm(**enc, output_hidden_states=True, logits_to_keep=1)
                    _ = out.hidden_states[LAYER].clone()
                    del out
                torch.cuda.synchronize()

                t0 = time.perf_counter()
                n_docs_done = 0
                for batch in batches[N_WARMUP_BATCHES:N_WARMUP_BATCHES + N_TIMED_BATCHES]:
                    enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
                    out = llm(**enc, output_hidden_states=True, logits_to_keep=1)
                    _ = out.hidden_states[LAYER].clone()
                    del out
                    n_docs_done += len(batch)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            docs_per_sec = n_docs_done / elapsed
            batches_per_sec = N_TIMED_BATCHES / elapsed
            results[bs] = {
                "status": "ok", "elapsed_s": elapsed, "n_docs": n_docs_done,
                "docs_per_sec": docs_per_sec, "batches_per_sec": batches_per_sec,
                "peak_vram_gb": peak_vram_gb,
            }
            print(f"[batch-sweep] batch_size={bs:3d} | {docs_per_sec:6.2f} docs/s | "
                  f"{batches_per_sec:5.2f} batches/s | pic VRAM={peak_vram_gb:5.1f} Go")
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            results[bs] = {"status": "oom", "error": str(e)}
            print(f"[batch-sweep] batch_size={bs:3d} | OOM -- arrêt du balayage ici.")
            break

    print("\n" + "=" * 70)
    print(" RÉSUMÉ — balayage EXTRACTION_BATCH_SIZE (LAYER=%d, %s)" % (LAYER, MODEL_ID))
    print("=" * 70)
    for bs, r in results.items():
        if r["status"] == "ok":
            print(f"  batch_size={bs:3d} : {r['docs_per_sec']:6.2f} docs/s, pic VRAM={r['peak_vram_gb']:5.1f} Go")
        else:
            print(f"  batch_size={bs:3d} : OOM")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"layer": LAYER, "model_id": MODEL_ID, "results": {str(k): v for k, v in results.items()}}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
