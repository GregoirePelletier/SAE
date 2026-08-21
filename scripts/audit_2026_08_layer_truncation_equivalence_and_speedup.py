"""
scripts/audit_2026_08_layer_truncation_equivalence_and_speedup.py — Vérifie
le correctif G1 de AUDIT_SAE_2026-08.md §2.2.

VERSION 2 -- la V1 (troncature `layers[:LAYER]` + `output_hidden_states=True` +
`hidden_states[LAYER]`, même mécanisme que le chemin de production actuel)
a ÉCHOUÉ à l'équivalence sur GPU (job 44536, torch.equal=False, écart max
~6e5). Cause confirmée en lisant `transformers/utils/output_capturing.py` :
`output_hidden_states=True` passe par un mécanisme générique
(`_can_record_outputs`) qui remplace INCONDITIONNELLEMENT la DERNIÈRE entrée
de `hidden_states` par `last_hidden_state` (post-RMSNorm final,
`tie_last_hidden_states=True`) -- invisible dans le modèle complet (LAYER
n'est jamais la dernière entrée sur 49), silencieusement faux dans le modèle
tronqué (LAYER DEVIENT la dernière entrée). Cette V2 utilise un
`register_forward_hook` DIRECT sur `layers[LAYER-1]` (même patron déjà
utilisé en production pour HOOK_TYPE=attn_out/mlp_out), qui ne passe jamais
par ce mécanisme.

Deux questions, dans cet ordre (la première conditionne la seconde) :
1. ÉQUIVALENCE : la sortie du hook direct sur layers[LAYER-1] (modèle
   tronqué) doit être STRICTEMENT identique à hidden_states[LAYER] (modèle
   complet, chemin de production actuel) -- un transformer causal/séquentiel
   n'a aucune dépendance arrière, la valeur au bloc LAYER-1 ne peut pas
   dépendre des blocs suivants. Si ce n'est pas `torch.equal` bit-à-bit, le
   correctif ne doit PAS être adopté tel quel, peu importe le gain de vitesse
   mesuré.
2. GAIN RÉEL : débit (docs/s) et pic VRAM, tronqué vs plein, sur le MÊME
   modèle chargé une seule fois, mêmes textes réels, même méthodologie de
   chronométrage (warmup exclu) que
   `benchmarks/extraction_batch_size_sweep.py`.

Indépendant du job 44211 en cours (a100, pas h100 -- aucune contention).
N'écrit rien dans le cache de production.

Usage : sbatch slurm/validation/run_audit_layer_truncation_equivalence.slurm
"""
from __future__ import annotations

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_ID, HF_TOKEN, LAYER, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED,
)
from src.data.preparation import build_email_train_test_corpus

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16
BATCH_SIZE = 8
N_WARMUP_BATCHES = 2
N_TIMED_BATCHES = 8
MAX_LENGTH = 512
OUT_PATH = "docs/audit_2026_08_layer_truncation_equivalence_results_v2_hook.json"


def _run_batches_full(llm, tokenizer, batches, layer_index):
    """Référence : chemin de production ACTUEL (saev5.py, resid_post), modèle
    complet. output_hidden_states=True, logits_to_keep=1, hidden_states[LAYER]."""
    with torch.no_grad():
        for batch in batches[:N_WARMUP_BATCHES]:
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            out = llm(**enc, output_hidden_states=True, logits_to_keep=1)
            _ = out.hidden_states[layer_index].clone()
            del out
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        n_docs_done = 0
        last_hidden = None
        for batch in batches[N_WARMUP_BATCHES:N_WARMUP_BATCHES + N_TIMED_BATCHES]:
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            out = llm(**enc, output_hidden_states=True, logits_to_keep=1)
            last_hidden = out.hidden_states[layer_index].clone()
            del out
            n_docs_done += len(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed, n_docs_done, last_hidden


def _run_batches_truncated_hook(llm, tokenizer, batches, layer_index):
    """Version corrigée : layers[:LAYER] (layer_index blocs, indices
    0..layer_index-1) + hook DIRECT sur layers[layer_index - 1], jamais
    output_hidden_states. Cause du premier échec (confirmé en lisant
    transformers/utils/output_capturing.py) : output_hidden_states=True passe
    par un mécanisme générique (`_can_record_outputs`) qui remplace
    INCONDITIONNELLEMENT la DERNIÈRE entrée de `hidden_states` par
    `last_hidden_state` (post-RMSNorm final, `tie_last_hidden_states=True`,
    "vrai pour tous les modèles de langage") -- correct dans le modèle complet
    (LAYER n'est pas la dernière entrée), silencieusement faux dans le modèle
    tronqué (LAYER Y DEVIENT la dernière entrée). Un hook direct sur le module
    ne passe jamais par ce mécanisme : jamais affecté par la substitution."""
    hook_capture = {}

    def _capture(module, args, output):
        hook_capture["acts"] = output

    handle = llm.model.language_model.layers[layer_index - 1].register_forward_hook(_capture)
    try:
        with torch.no_grad():
            for batch in batches[:N_WARMUP_BATCHES]:
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
                llm(**enc, logits_to_keep=1)
                _ = hook_capture["acts"].clone()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            n_docs_done = 0
            last_hidden = None
            for batch in batches[N_WARMUP_BATCHES:N_WARMUP_BATCHES + N_TIMED_BATCHES]:
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
                llm(**enc, logits_to_keep=1)
                last_hidden = hook_capture["acts"].clone()
                n_docs_done += len(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
    finally:
        handle.remove()
    return elapsed, n_docs_done, last_hidden


def main():
    print(f"[layer-trunc] Chargement {MODEL_ID} (bf16), LAYER={LAYER}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()
    n_layers_full = len(llm.model.language_model.layers)
    print(f"[layer-trunc] {n_layers_full} blocs décodeur au total.")

    print("[layer-trunc] Chargement d'un échantillon de textes réels (corpus emails+augmentés)...")
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_needed = BATCH_SIZE * (N_WARMUP_BATCHES + N_TIMED_BATCHES)
    sample_texts = train_texts[:n_needed]
    batches = [sample_texts[i:i + BATCH_SIZE] for i in range(0, n_needed, BATCH_SIZE)]
    print(f"[layer-trunc] {len(sample_texts)} textes, {len(batches)} batches de {BATCH_SIZE}.")

    # --- 1. Référence : modèle PLEIN (48 blocs), hidden_states[LAYER] (chemin
    #        de production actuel, output_hidden_states=True) ---
    print("[layer-trunc] Passe 1/2 : modèle complet (référence, output_hidden_states=True)...")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    elapsed_full, n_docs_full, hidden_full = _run_batches_full(llm, tokenizer, batches, LAYER)
    peak_vram_full_gb = torch.cuda.max_memory_allocated() / 1e9
    docs_per_sec_full = n_docs_full / elapsed_full
    print(f"[layer-trunc]   {docs_per_sec_full:.2f} docs/s, pic VRAM={peak_vram_full_gb:.1f} Go")

    # --- 2. Modèle TRONQUÉ à layers[:LAYER] + HOOK DIRECT sur layers[LAYER-1]
    #        (jamais output_hidden_states -- cf. docstring de
    #        _run_batches_truncated_hook pour la cause du premier échec) ---
    print(f"[layer-trunc] Troncature layers[:{LAYER}] (sur {n_layers_full} blocs) + hook direct...")
    llm.model.language_model.layers = llm.model.language_model.layers[:LAYER]
    assert len(llm.model.language_model.layers) == LAYER

    print("[layer-trunc] Passe 2/2 : modèle tronqué (hook direct)...")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    elapsed_trunc, n_docs_trunc, hidden_trunc = _run_batches_truncated_hook(llm, tokenizer, batches, LAYER)
    peak_vram_trunc_gb = torch.cuda.max_memory_allocated() / 1e9
    docs_per_sec_trunc = n_docs_trunc / elapsed_trunc
    print(f"[layer-trunc]   {docs_per_sec_trunc:.2f} docs/s, pic VRAM={peak_vram_trunc_gb:.1f} Go")

    # --- 3. Équivalence stricte ---
    exactly_equal = torch.equal(hidden_full, hidden_trunc)
    max_abs_diff = (hidden_full.float() - hidden_trunc.float()).abs().max().item()
    print(f"\n[layer-trunc] torch.equal(hidden_full, hidden_trunc) = {exactly_equal}")
    print(f"[layer-trunc] max |diff| = {max_abs_diff:.3e}")

    speedup = docs_per_sec_trunc / docs_per_sec_full
    vram_saved_gb = peak_vram_full_gb - peak_vram_trunc_gb

    print("\n" + "=" * 70)
    print(" RÉSUMÉ — troncature layers[:LAYER] (LAYER=%d/%d, %s)" % (LAYER, n_layers_full, MODEL_ID))
    print("=" * 70)
    print(f"  Équivalence stricte (torch.equal)     : {exactly_equal}")
    print(f"  Débit plein / tronqué                 : {docs_per_sec_full:.2f} / {docs_per_sec_trunc:.2f} docs/s")
    print(f"  Speedup                               : {speedup:.2f}x")
    print(f"  Pic VRAM plein / tronqué / économisé  : {peak_vram_full_gb:.1f} / {peak_vram_trunc_gb:.1f} / {vram_saved_gb:.1f} Go")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "layer": LAYER, "n_layers_full": n_layers_full, "model_id": MODEL_ID,
            "batch_size": BATCH_SIZE, "n_timed_batches": N_TIMED_BATCHES,
            "equivalence": {"torch_equal": exactly_equal, "max_abs_diff": max_abs_diff},
            "full": {"docs_per_sec": docs_per_sec_full, "peak_vram_gb": peak_vram_full_gb,
                     "elapsed_s": elapsed_full},
            "truncated": {"docs_per_sec": docs_per_sec_trunc, "peak_vram_gb": peak_vram_trunc_gb,
                          "elapsed_s": elapsed_trunc},
            "speedup": speedup, "vram_saved_gb": vram_saved_gb,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[layer-trunc] Résultats -> {OUT_PATH}")

    if not exactly_equal:
        raise SystemExit(
            "ÉCHEC : hidden_states[LAYER] diffère entre le modèle plein et le modèle "
            "tronqué -- NE PAS adopter la troncature layers[:LAYER] dans saev5.py "
            "tant que cette différence n'est pas expliquée."
        )


if __name__ == "__main__":
    main()
