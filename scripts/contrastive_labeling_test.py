"""
scripts/contrastive_labeling_test.py — Teste le protocole de labellisation
CONTRASTIF direct d'interp_embed (Jiang, Sun et al. 2025, Appendix C — cf.
docs/references.md) en alternative au gate odd-one-out actuel
(src/sae/judge.py::odd_one_out_judge).

Contexte : notre protocole ne génère un label QUE si le test odd-one-out (1
négatif mélangé parmi 9 positifs, décision greedy unique) réussit, avec un
prompt de labellisation minimal (9 positifs seulement, pas de négatif). Le
papier de référence ne fait jamais ce gate : il présente toujours 10 positifs +
10 négatifs à un LLM et lui demande de nommer directement la propriété qui les
distingue, avec des instructions bien plus détaillées (regarder le contexte
AVANT les marqueurs << >>, ignorer les tokens spéciaux, chercher UNE propriété
unifiée). Sachant qu'on a déjà mesuré (RESULTS_TESTS.md §13.1) que notre gate
odd-one-out est bruyant (31,3% des décisions changent selon l'ordre de
présentation des mêmes exemples), ce script teste si le protocole contrastif
direct récupère des labels plausibles sur des features que notre gate rejette.

Réutilise les activations et fragments DÉJÀ en cache (résultats_v10_emails_main/)
-- recharge seulement Gemma-3-12B comme juge, aucune réextraction.

Usage :
    SAVE_DIR=./results_v10_emails_main/ MODEL_SIZE=12b \
      PYTHONPATH=. .venv/bin/python scripts/contrastive_labeling_test.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "sae"))

from src.config import MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED
from src.sae.judge import (
    build_feature_examples_with_control, extract_causal_context, _apply_chat_and_extract,
)
from src.storage.fragment_store import fragment_exists, load_fragment, feature_column
from src.data.preparation import build_email_train_test_corpus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
SEED = int(os.environ.get("SEED", "42"))
N_NEG = int(os.environ.get("N_NEG", "10"))

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
TOKEN_FRAGMENTS_DIR = os.path.join(CACHE_DIR, "p1_token_fragments")
OUT_PATH = os.path.join(CACHE_DIR, "p1_contrastive_labels.json")


def build_negative_examples(f_idx, token_fragments_dir, acts, offset=0, n_neg=10, neg_quantile=0.05):
    """Généralise le négatif unique de build_feature_examples_with_control à N négatifs
    (cf. interp_embed Appendix C : 10 positifs + 10 négatifs, symétriques). Les
    marqueurs << >> sont retirés : `extract_causal_context` les pose toujours sur le
    token cible qu'on lui donne, mais pour un négatif ce token est un point arbitraire
    (milieu du document), pas une activation réelle -- le papier de référence est
    explicite ("NEGATIVE samples... no << >> markers") ; les laisser induirait le juge
    à croire qu'un signal existe là où il n'y en a pas."""
    f_acts = acts[:, f_idx].detach().float().numpy()
    threshold_neg = float(np.quantile(f_acts, neg_quantile))
    neg_pool = np.where(f_acts <= threshold_neg)[0].tolist()
    random.shuffle(neg_pool)
    negatives = []
    for d_idx in neg_pool:
        if not fragment_exists(token_fragments_dir, int(d_idx + offset)):
            continue
        doc_data = load_fragment(token_fragments_dir, int(d_idx + offset))
        toks = doc_data["token_strings"]
        mid = len(toks) // 2
        marked = extract_causal_context(toks, mid)
        negatives.append(re.sub(r"<<(.+?)>>", r"\1", marked))
        if len(negatives) >= n_neg:
            break
    return negatives


def contrastive_label_prompt(pos_examples: list[str], neg_examples: list[str]) -> str:
    """Adaptation FR du prompt d'Appendix C (interp_embed) : contexte avant les
    marqueurs pris en compte, tokens spéciaux ignorés, UNE propriété unifiée."""
    pos_txt = "\n".join(f"- {ex}" for ex in pos_examples)
    neg_txt = "\n".join(f"- {ex}" for ex in neg_examples)
    return (
        "Tu es expert en interprétation de features de sparse autoencoders (SAE) pour "
        "modèles de langage.\n\n"
        f"Voici {len(pos_examples)} exemples POSITIFS (la feature s'est activée, mots "
        f"déclencheurs entre << >>) et {len(neg_examples)} exemples NÉGATIFS (la feature "
        "ne s'est PAS activée, aucun marqueur).\n\n"
        "NOTES IMPORTANTES :\n"
        "1. Les marqueurs << >> indiquent où la feature s'est activée, mais ne te limite "
        "PAS à ces seuls tokens — le contexte AVANT les marqueurs donne souvent une "
        "information cruciale sur ce que la feature détecte.\n"
        "2. La feature peut répondre à un motif qui s'étend sur le contexte ET les tokens "
        "marqués.\n"
        "3. Ignore tout token technique/spécial isolé (fin de séquence, remplissage) — ce "
        "n'est jamais une activation significative en soi.\n\n"
        f"EXEMPLES POSITIFS :\n{pos_txt}\n\n"
        f"EXEMPLES NÉGATIFS :\n{neg_txt}\n\n"
        "Ta tâche :\n"
        "- Compare attentivement les exemples positifs et négatifs.\n"
        "- Identifie la propriété la plus spécifique et concise présente dans les "
        "positifs mais absente des négatifs.\n"
        "- Cherche UNE propriété unifiée, pas une liste de propriétés disparates.\n"
        "- Si tu ne trouves vraiment aucune propriété commune cohérente, mets "
        '"confident": false et n\'invente pas de label vague.\n\n'
        "Réponds en JSON strict, avec CES TROIS CLÉS EXACTEMENT (remplace les "
        "valeurs entre <> par ta réponse, ne recopie jamais le texte entre <> "
        "tel quel) :\n"
        '{"label": "<ta réponse : 2 à 4 mots en français, PAS le texte de cet exemple>", '
        '"brief_description": "<ta réponse : une phrase expliquant la propriété détectée>", '
        '"confident": <true ou false, false si aucune propriété cohérente trouvée>}'
    )


def main():
    random.seed(SEED)
    with open(JUDGE_CACHE, encoding="utf-8") as f:
        original_judge_data = json.load(f)
    feature_indices = [int(k) for k in original_judge_data.keys()]
    print(f"[contrastive] {len(feature_indices)} features (mêmes que le run odd-one-out original).")

    print("[contrastive] Reconstruction du split train/test (déterministe, pas de GPU)...")
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train = len(train_texts)

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    train_doc_acts = all_doc_acts[:n_train]

    print(f"[contrastive] Chargement du juge : {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    results = {}
    n_confident_new = 0
    n_orig_noninterp_now_confident = 0
    n_orig_noninterp = 0

    for i, f_idx in enumerate(feature_indices):
        pos_examples, _ = build_feature_examples_with_control(
            f_idx, TOKEN_FRAGMENTS_DIR, train_doc_acts, offset=0, n_pos=10,
        )
        neg_examples = build_negative_examples(f_idx, TOKEN_FRAGMENTS_DIR, train_doc_acts, n_neg=N_NEG)
        if len(pos_examples) < 3 or not neg_examples:
            results[f_idx] = {"label": "dead_feature", "confident": False}
            continue

        prompt = contrastive_label_prompt(pos_examples, neg_examples)
        inputs = _apply_chat_and_extract(
            tokenizer, [{"role": "user", "content": prompt}],
            device=model.device, add_generation_prompt=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=150, do_sample=False)
            resp = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        try:
            label_data = json.loads(re.search(r"\{.*?\}", resp, re.DOTALL).group())
        except Exception:
            label_data = {"label": "PARSE_ERROR", "brief_description": resp[:200], "confident": False}

        original = original_judge_data[str(f_idx)]
        original_interp = original.get("interp_score", 0)
        is_confident = bool(label_data.get("confident", False))

        n_confident_new += int(is_confident)
        if original_interp == 0:
            n_orig_noninterp += 1
            n_orig_noninterp_now_confident += int(is_confident)

        results[f_idx] = {
            **label_data,
            "original_label": original.get("label"),
            "original_interp_score": original_interp,
        }
        if (i + 1) % 25 == 0:
            print(f"[contrastive] {i+1}/{len(feature_indices)} features traitées...")

    n_tested = len(results)
    summary = {
        "n_tested": n_tested,
        "n_confident_new_protocol": n_confident_new,
        "confident_rate_new_protocol": n_confident_new / n_tested,
        "n_original_noninterp": n_orig_noninterp,
        "n_original_noninterp_now_confident": n_orig_noninterp_now_confident,
        "recovery_rate_among_original_noninterp": (
            n_orig_noninterp_now_confident / n_orig_noninterp if n_orig_noninterp else float("nan")
        ),
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — LABELLISATION CONTRASTIVE DIRECTE (interp_embed) vs GATE ODD-ONE-OUT")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
