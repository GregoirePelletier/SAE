"""
judge_patch.py — Remplace les fonctions de labellisation dans saev5.py.

Changements :
  1. extract_causal_context : gestion ▁ SentencePiece + marquage au niveau mot (pas subtoken)
  2. build_feature_examples_with_control : retourne (pos_examples, neg_example)
  3. _apply_chat_and_extract : fix BatchEncoding bug apply_chat_template (4 sites)
  4. odd_one_out_judge : protocole SAEBench feature-detection + ρ_interp (Bills 2023)
  5. feature_selection_by_magnitude : sélection par magnitude token-level (pas fréquence doc-level)

Intégration dans saev5.py :
  - Copier ces fonctions à la place des anciennes (extract_causal_context, build_causal_highlighted_examples,
    local_gemma_judge, le bloc judge Expert+Critique lignes 907-1025)
  - Remplacer les 4 appels apply_chat_template par _apply_chat_and_extract
  - Remplacer freq_core_acts par feature_selection_by_magnitude (ligne 904)
"""

import re
import os
import json
import pickle
import random
import numpy as np
import torch
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# 1. HELPER : BatchEncoding fix (transformers >= 4.43)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_chat_and_extract(tokenizer, messages: list, device, **kwargs) -> torch.Tensor:
    """Retourne toujours un Tensor, quel que soit le type de retour de apply_chat_template."""
    out = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(out, "input_ids"):   # BatchEncoding
        out = out.input_ids
    return out.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# 2. EXTRACTION CONTEXTE — niveau mot, gestion ▁ SentencePiece
# ──────────────────────────────────────────────────────────────────────────────

def _is_word_start(tok: str) -> bool:
    """Token démarre un nouveau mot (SentencePiece ▁ ou GPT-2 Ġ ou position 0)."""
    return tok.startswith("▁") or tok.startswith("Ġ")


def _clean_token(tok: str) -> str:
    return tok.replace("▁", " ").replace("Ġ", " ")


def _word_span(token_strings: list, target_idx: int) -> tuple[int, int]:
    """Retourne (word_start, word_end) inclusifs contenant target_idx."""
    word_start = target_idx
    while word_start > 0 and not _is_word_start(token_strings[word_start]):
        word_start -= 1
    word_end = target_idx
    while word_end + 1 < len(token_strings) and not _is_word_start(token_strings[word_end + 1]):
        word_end += 1
    return word_start, word_end


def extract_causal_context(
    token_strings: list,
    target_idx: int,
    left_window: int = 60,
) -> str:
    """
    Contexte causal (gauche) avec marquage au niveau mot, pas subtoken.
    Le mot complet contenant target_idx est marqué <<mot>>.
    """
    word_start, word_end = _word_span(token_strings, target_idx)
    ctx_start = max(0, word_start - left_window)

    left_part = "".join(_clean_token(t) for t in token_strings[ctx_start:word_start])
    target_word = "".join(_clean_token(t) for t in token_strings[word_start:word_end + 1]).strip()

    ctx = re.sub(r"\s+", " ", left_part).strip()
    return f"{ctx} <<{target_word}>>".strip()


# ──────────────────────────────────────────────────────────────────────────────
# 3. COLLECTE EXEMPLES — positifs + contrôle négatif
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_examples_with_control(
    f_idx: int,
    token_fragments_dir: str,
    acts: torch.Tensor,       # (n_docs, d_sae) max-pool doc acts
    offset: int = 0,
    n_pos: int = 9,
    neg_quantile: float = 0.05,   # docs sous ce quantile d'activation → pool négatif
) -> tuple[list[str], Optional[str]]:
    """
    Retourne (pos_examples, neg_example) pour le protocole odd-one-out.
    neg_example est None si aucun fragment disponible.
    """
    f_acts = acts[:, f_idx].detach().float().numpy()
    threshold_pos = 1e-6
    threshold_neg = float(np.quantile(f_acts, neg_quantile))

    # Positifs : top par magnitude
    # NB : on déduplique sur le mot-cible marqué (<<mot>>), pas sur l'index de document.
    # Sans cela, un même mot-déclencheur très fréquent (ex. "cher" en tête de mail)
    # peut apparaître 2-3 fois comme exemples "positifs" distincts alors qu'il s'agit
    # sémantiquement du même exemple pour le juge LLM.
    sorted_desc = np.argsort(f_acts)[::-1]
    pos_examples = []
    seen_target_words = set()
    for d_idx in sorted_desc:
        if f_acts[d_idx] <= threshold_pos:
            break
        frag_path = os.path.join(token_fragments_dir, f"doc_{int(d_idx + offset):05d}.pkl")
        if not os.path.exists(frag_path):
            continue
        with open(frag_path, "rb") as fh:
            doc_data = pickle.load(fh)
        token_acts = doc_data["token_sae_acts"][:, f_idx].numpy()
        max_act = token_acts.max()
        if max_act <= threshold_pos:
            continue
        target_idx = int(token_acts.argmax())
        ctx = extract_causal_context(doc_data["token_strings"], target_idx)
        m = re.search(r"<<(.+?)>>", ctx)
        target_word = m.group(1).strip().lower() if m else ctx.strip().lower()
        if target_word in seen_target_words:
            continue
        seen_target_words.add(target_word)
        pos_examples.append(ctx)
        if len(pos_examples) >= n_pos:
            break

    # Négatif : doc avec activation nulle ou quasi-nulle
    neg_pool = np.where(f_acts <= threshold_neg)[0].tolist()
    random.shuffle(neg_pool)
    neg_example = None
    for d_idx in neg_pool[:20]:
        frag_path = os.path.join(token_fragments_dir, f"doc_{int(d_idx + offset):05d}.pkl")
        if not os.path.exists(frag_path):
            continue
        with open(frag_path, "rb") as fh:
            doc_data = pickle.load(fh)
        toks = doc_data["token_strings"]
        mid = len(toks) // 2
        neg_example = extract_causal_context(toks, mid)
        break

    return pos_examples, neg_example


# ──────────────────────────────────────────────────────────────────────────────
# 4. SÉLECTION DES FEATURES — magnitude token-level
# ──────────────────────────────────────────────────────────────────────────────

def feature_selection_by_magnitude(
    token_fragments_dir: str,
    doc_indices: list[int],     # indices dans le split d'entraînement
    d_sae: int,
    n_features: int,
    sample_docs: int = 500,
) -> list[int]:
    """
    Sélection des n_features features par mean activation magnitude sur tokens
    (pas par fréquence sur doc max-pool).
    Échantillonne sample_docs documents pour éviter OOM.
    """
    sample_docs = min(sample_docs, len(doc_indices))
    sampled = random.sample(doc_indices, sample_docs)
    acc = np.zeros(d_sae, dtype=np.float64)
    n_tokens = 0
    for d_idx in sampled:
        frag_path = os.path.join(token_fragments_dir, f"doc_{d_idx:05d}.pkl")
        if not os.path.exists(frag_path):
            continue
        with open(frag_path, "rb") as fh:
            frag = pickle.load(fh)
        tok_acts = frag["token_sae_acts"].float().numpy()  # (T, d_sae)
        acc += tok_acts.sum(axis=0)
        n_tokens += tok_acts.shape[0]
    if n_tokens == 0:
        return list(range(n_features))
    mean_mag = acc / n_tokens
    return np.argsort(mean_mag)[::-1][:n_features].tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 5. JUDGE — odd-one-out + ρ_interp (Bills 2023)
# ──────────────────────────────────────────────────────────────────────────────

def odd_one_out_judge(
    model,
    tokenizer,
    feature_indices: list[int],
    token_fragments_dir: str,
    acts: torch.Tensor,
    offset: int = 0,
    n_pos: int = 9,
) -> dict:
    """
    Pour chaque feature :
      1. Présente 9 exemples positifs + 1 négatif (shufflés) au LLM.
      2. Demande lequel est l'intrus → score interp ∈ {0, 1}.
      3. Si interprétable : génère label + description.
      4. Calcule ρ_interp (Spearman) entre prédiction LLM et activations réelles.

    Retourne dict { f_idx: { label, brief_description, interp_score, rho_interp } }.
    """
    from scipy.stats import spearmanr

    model.eval()
    results = {}

    for f_idx in feature_indices:
        pos_examples, neg_example = build_feature_examples_with_control(
            f_idx, token_fragments_dir, acts, offset=offset, n_pos=n_pos,
        )

        if len(pos_examples) < 3:
            results[f_idx] = {
                "label": "dead_feature",
                "brief_description": "Aucune activation.",
                "interp_score": 0,
                "rho_interp": float("nan"),
            }
            continue

        # ── Étape 1 : Odd-one-out ──────────────────────────────────────────
        all_examples = pos_examples + ([neg_example] if neg_example else [])
        neg_position = len(all_examples) - 1 if neg_example else None
        indices = list(range(len(all_examples)))
        random.shuffle(indices)
        shuffled = [all_examples[i] for i in indices]
        correct_answer = indices.index(neg_position) + 1 if neg_example else None  # 1-based

        examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(shuffled))
        prompt_ood = (
            "Voici des exemples de textes où une feature neuronale est fortement activée "
            "(sauf un, qui est un contrôle négatif).\n\n"
            f"{examples_text}\n\n"
            "Quel numéro est l'intrus (celui qui ne partage pas le concept commun des autres) ? "
            "Réponds uniquement avec le numéro."
        )

        inputs = _apply_chat_and_extract(
            tokenizer, [{"role": "user", "content": prompt_ood}],
            device=model.device, add_generation_prompt=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=8, do_sample=False)
            resp_ood = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

        try:
            predicted = int(re.search(r"\d+", resp_ood).group())
        except Exception:
            predicted = -1

        interp_score = int(predicted == correct_answer) if correct_answer is not None else 0

        # ── Étape 2 : Label (si interprétable) ───────────────────────────
        label_data = {"label": f"Feature_{f_idx}", "brief_description": "Non interprétable."}
        if interp_score == 1:
            formatted = "\n".join(f"- {ex}" for ex in pos_examples)
            prompt_label = (
                "Ces exemples textuels activent tous fortement une même feature neuronale "
                "(les mots déclencheurs sont entre << >>).\n\n"
                f"{formatted}\n\n"
                "Génère un objet JSON avec un label court en français (≤3 mots) et une description concise :\n"
                '{"label": "...", "brief_description": "..."}'
            )
            inputs_l = _apply_chat_and_extract(
                tokenizer, [{"role": "user", "content": prompt_label}],
                device=model.device, add_generation_prompt=True, return_tensors="pt",
            )
            with torch.no_grad():
                out_l = model.generate(input_ids=inputs_l, max_new_tokens=128, do_sample=False)
                resp_l = tokenizer.decode(out_l[0][inputs_l.shape[-1]:], skip_special_tokens=True)
            try:
                label_data = json.loads(re.search(r"\{.*?\}", resp_l, re.DOTALL).group())
            except Exception:
                pass

        # ── Étape 3 : ρ_interp (Bills 2023) ──────────────────────────────
        # LLM score chaque exemple (pos + neg) sur [0, 10] ; Spearman vs activation réelle
        rho_interp = float("nan")
        if interp_score == 1 and neg_example:
            label_str = label_data.get("label", "")
            score_prompts = "\n".join(
                f"{i+1}. {ex}" for i, ex in enumerate(shuffled)
            )
            prompt_score = (
                f"Concept : « {label_str} »\n\n"
                "Pour chaque exemple ci-dessous, note de 0 (non lié) à 10 (fortement lié) "
                "l'intensité du lien avec ce concept. "
                "Réponds uniquement avec un JSON : {\"scores\": [s1, s2, ...]}\n\n"
                f"{score_prompts}"
            )
            inputs_s = _apply_chat_and_extract(
                tokenizer, [{"role": "user", "content": prompt_score}],
                device=model.device, add_generation_prompt=True, return_tensors="pt",
            )
            with torch.no_grad():
                out_s = model.generate(input_ids=inputs_s, max_new_tokens=128, do_sample=False)
                resp_s = tokenizer.decode(out_s[0][inputs_s.shape[-1]:], skip_special_tokens=True)
            try:
                scores_llm = json.loads(re.search(r"\{.*?\}", resp_s, re.DOTALL).group())["scores"]
                # Activation réelle pour chaque exemple (positifs connus + 0 pour le négatif)
                f_acts_np = acts[:, f_idx].detach().float().numpy()
                # Reconstituons les activations dans l'ordre shufflé
                # pos_examples[i] correspond à doc trié desc → approx suffisant
                act_ground = []
                for orig_idx in indices:
                    if orig_idx < len(pos_examples):
                        # activation approximative : rang dans le top
                        act_ground.append(float(n_pos - orig_idx))
                    else:
                        act_ground.append(0.0)
                if len(scores_llm) == len(act_ground):
                    rho_interp = float(spearmanr(scores_llm, act_ground).statistic)
            except Exception:
                pass

        results[f_idx] = {
            **label_data,
            "interp_score": interp_score,
            "rho_interp": rho_interp,
            "pos_examples": pos_examples,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 6. JUDGE NIVEAU PHRASE (Pipeline 2 — F2LLM Phrase-Level SAE)
# ──────────────────────────────────────────────────────────────────────────────
# Contrairement à odd_one_out_judge (Pipeline 1), il n'y a pas de fragments de
# tokens sur disque ici : l'unité d'activation EST déjà la phrase entière.
# On applique donc le même protocole odd-one-out + ρ_interp, mais les
# "exemples" sont directement les phrases les plus/moins activantes.

def build_phrase_examples_with_control(
    f_idx: int,
    phrase_texts: list,
    phrase_acts: torch.Tensor,     # (n_phrases, d_sae)
    n_pos: int = 9,
    neg_quantile: float = 0.05,
) -> tuple[list[str], Optional[str]]:
    """Équivalent phrase-level de build_feature_examples_with_control : pas de
    fragments à charger, la phrase elle-même est l'exemple. Déduplication sur
    le texte de la phrase (nettoyé) pour éviter les répétitions."""
    f_acts = phrase_acts[:, f_idx].detach().float().numpy()
    threshold_pos = 1e-6
    threshold_neg = float(np.quantile(f_acts, neg_quantile))

    sorted_desc = np.argsort(f_acts)[::-1]
    pos_examples = []
    seen = set()
    for p_idx in sorted_desc:
        if f_acts[p_idx] <= threshold_pos:
            break
        text = re.sub(r"\s+", " ", phrase_texts[p_idx]).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        pos_examples.append(f"<<{text}>>")
        if len(pos_examples) >= n_pos:
            break

    neg_pool = np.where(f_acts <= threshold_neg)[0].tolist()
    random.shuffle(neg_pool)
    neg_example = None
    for p_idx in neg_pool[:20]:
        text = re.sub(r"\s+", " ", phrase_texts[p_idx]).strip()
        if text:
            neg_example = f"<<{text}>>"
            break

    return pos_examples, neg_example


def local_gemma_judge(
    model,
    tokenizer,
    feature_indices: list[int],
    phrase_texts: list,
    phrase_acts: torch.Tensor,
    phrase_to_doc: Optional[np.ndarray] = None,
    n_pos: int = 9,
) -> dict:
    """
    Labellisation locale (Gemma-3) des features du Phrase-Level SAE (Pipeline 2).
    Même protocole odd-one-out + ρ_interp que odd_one_out_judge (Pipeline 1),
    mais construit directement sur les phrases (pas de fragments tokens à charger).

    `phrase_to_doc` n'est pas requis pour la labellisation elle-même (conservé
    pour compat/signature future si besoin de contexte inter-phrase).
    """
    from scipy.stats import spearmanr

    model.eval()
    results = {}

    for f_idx in feature_indices:
        pos_examples, neg_example = build_phrase_examples_with_control(
            f_idx, phrase_texts, phrase_acts, n_pos=n_pos,
        )

        if len(pos_examples) < 3:
            results[str(f_idx)] = {
                "label": "dead_feature",
                "brief_description": "Aucune activation.",
                "interp_score": 0,
                "rho_interp": float("nan"),
            }
            continue

        all_examples = pos_examples + ([neg_example] if neg_example else [])
        neg_position = len(all_examples) - 1 if neg_example else None
        indices = list(range(len(all_examples)))
        random.shuffle(indices)
        shuffled = [all_examples[i] for i in indices]
        correct_answer = indices.index(neg_position) + 1 if neg_example else None

        examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(shuffled))
        prompt_ood = (
            "Voici des phrases où une feature neuronale est fortement activée "
            "(sauf une, qui est un contrôle négatif). Le mot/groupe déclencheur "
            "est entre << >>.\n\n"
            f"{examples_text}\n\n"
            "Quel numéro est l'intrus (celui qui ne partage pas le concept commun des autres) ? "
            "Réponds uniquement avec le numéro."
        )
        inputs = _apply_chat_and_extract(
            tokenizer, [{"role": "user", "content": prompt_ood}],
            device=model.device, add_generation_prompt=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = model.generate(input_ids=inputs, max_new_tokens=8, do_sample=False)
            resp_ood = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        try:
            predicted = int(re.search(r"\d+", resp_ood).group())
        except Exception:
            predicted = -1
        interp_score = int(predicted == correct_answer) if correct_answer is not None else 0

        label_data = {"label": f"Feature_{f_idx}", "brief_description": "Non interprétable."}
        if interp_score == 1:
            formatted = "\n".join(f"- {ex}" for ex in pos_examples)
            prompt_label = (
                "Ces phrases activent toutes fortement une même feature neuronale "
                "(les mots/groupes déclencheurs sont entre << >>).\n\n"
                f"{formatted}\n\n"
                "Génère un objet JSON avec un label court en français (≤3 mots) et une description concise :\n"
                '{"label": "...", "brief_description": "..."}'
            )
            inputs_l = _apply_chat_and_extract(
                tokenizer, [{"role": "user", "content": prompt_label}],
                device=model.device, add_generation_prompt=True, return_tensors="pt",
            )
            with torch.no_grad():
                out_l = model.generate(input_ids=inputs_l, max_new_tokens=128, do_sample=False)
                resp_l = tokenizer.decode(out_l[0][inputs_l.shape[-1]:], skip_special_tokens=True)
            try:
                label_data = json.loads(re.search(r"\{.*?\}", resp_l, re.DOTALL).group())
            except Exception:
                pass

        rho_interp = float("nan")
        if interp_score == 1 and neg_example:
            label_str = label_data.get("label", "")
            score_prompts = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(shuffled))
            prompt_score = (
                f"Concept : « {label_str} »\n\n"
                "Pour chaque exemple ci-dessous, note de 0 (non lié) à 10 (fortement lié) "
                "l'intensité du lien avec ce concept. "
                "Réponds uniquement avec un JSON : {\"scores\": [s1, s2, ...]}\n\n"
                f"{score_prompts}"
            )
            inputs_s = _apply_chat_and_extract(
                tokenizer, [{"role": "user", "content": prompt_score}],
                device=model.device, add_generation_prompt=True, return_tensors="pt",
            )
            with torch.no_grad():
                out_s = model.generate(input_ids=inputs_s, max_new_tokens=128, do_sample=False)
                resp_s = tokenizer.decode(out_s[0][inputs_s.shape[-1]:], skip_special_tokens=True)
            try:
                scores_llm = json.loads(re.search(r"\{.*?\}", resp_s, re.DOTALL).group())["scores"]
                act_ground = [float(n_pos - i) if i < len(pos_examples) else 0.0 for i in indices]
                if len(scores_llm) == len(act_ground):
                    rho_interp = float(spearmanr(scores_llm, act_ground).statistic)
            except Exception:
                pass

        results[str(f_idx)] = {
            **label_data,
            "interp_score": interp_score,
            "rho_interp": rho_interp,
            "pos_examples": pos_examples,
        }

    return results