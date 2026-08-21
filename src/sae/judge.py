"""
judge.py — Fonctions de labellisation des features SAE par juge LLM, importées
par `saev5.py` (protocole odd-one-out, ρ_interp de Bills 2023) et par les
scripts d'audit de robustesse du juge (`scripts/*judge*.py`,
`scripts/*rejudge*.py`).
"""

import re
import os
import json
import pickle
try:
    from src.storage.fragment_store import (
        load_fragment, fragment_exists, feature_column, sum_columns, doc_maxpool,
    )
except ImportError:
    from fragment_store import (
        load_fragment, fragment_exists, feature_column, sum_columns, doc_maxpool,
    )
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


def _batched_generate(model, tokenizer, list_of_messages: list[list[dict]],
                       max_new_tokens: int, batch_size: int = 16) -> list[str]:
    """Génère une réponse par prompt indépendant, par lots de `batch_size`
    (audit perf §2.6, item 1 : `model.generate` appelé une fois par feature,
    bs=1, borné par la bande passante mémoire -- batcher amortit le transfert
    de poids sur tout le lot, 8-16x mesurés dans l'audit).

    `padding_side="left"` + `attention_mask` explicite : seule façon correcte
    de batcher une génération -- aligne la fin de chaque prompt (donc le début
    de la continuation générée) sur la même colonne pour toutes les lignes du
    lot, et le masque garantit que les tokens de padding n'influencent jamais
    l'attention des tokens réels. `do_sample=False` (inchangé, appelants
    existants) : déterministe, un lot ou un prompt à la fois doit produire la
    MÊME sortie pour un prompt donné -- vérifié par
    tests/test_judge_batched_generation.py (mock, CPU) plutôt que supposé.

    N'utilise PAS `apply_chat_template(..., tokenize=True, return_tensors="pt")`
    sur une LISTE de conversations (support inégal du padding batché selon les
    versions de transformers, cf. `_apply_chat_and_extract` ci-dessus qui
    contourne déjà un piège voisin) -- template appliqué en texte
    (`tokenize=False`) prompt par prompt (CPU, négligeable), puis tokenisation
    batchée avec padding, chemin standard et portable.

    Tri par longueur de prompt avant de découper en lots (audit perf §2.6 :
    sans lui, un lot mélangeant prompts courts et longs paie le padding du
    plus long sur toute la ligne) -- longueur en caractères comme proxy de la
    longueur tokenisée (évite une passe de tokenisation dédiée juste pour
    trier), l'ordre d'origine est restauré à la fin via `order`. N'affecte pas
    la sémantique (mêmes prompts, `do_sample=False`) : seul l'ORDRE de
    traitement et la composition des lots changent, avec le même bruit de
    non-associativité flottante entre lots déjà documenté et accepté pour ce
    juge (item 1, §2.9).
    """
    original_padding_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    n = len(list_of_messages)
    responses: list[str] = [""] * n
    try:
        texts = [
            tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in list_of_messages
        ]
        order = sorted(range(n), key=lambda i: len(texts[i]))
        for start in range(0, n, batch_size):
            batch_order = order[start:start + batch_size]
            chunk_texts = [texts[i] for i in batch_order]
            enc = tokenizer(chunk_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens, do_sample=False,
                )
            gen_only = out[:, enc["input_ids"].shape[-1]:]
            for j, orig_i in enumerate(batch_order):
                responses[orig_i] = tokenizer.decode(gen_only[j], skip_special_tokens=True)
    finally:
        tokenizer.padding_side = original_padding_side
    return responses


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
    return_magnitudes: bool = False,
):
    """
    Retourne (pos_examples, neg_example) pour le protocole odd-one-out.
    neg_example est None si aucun fragment disponible.

    `return_magnitudes=True` (défaut False, RÉTROCOMPATIBLE -- 4 appelants
    existants inchangés) : retourne en plus (pos_magnitudes, neg_magnitude), la
    magnitude d'activation RÉELLE (token-level) de chaque exemple -- permet un
    ρ_interp fidèle à Bills et al. 2023 (magnitude réelle, pas un rang synthétique
    ni un négatif à 0.0 fixe).
    """
    f_acts = acts[:, f_idx].detach().float().numpy()
    threshold_pos = 1e-6
    threshold_neg = float(np.quantile(f_acts, neg_quantile))

    # Positifs : top par magnitude
    # NB : on déduplique sur le mot-cible marqué (<<mot>>), pas sur l'index de document.
    # Sans cela, un même mot-déclencheur très fréquent (ex. "cher" en tête de mail)
    # peut apparaître 2-3 fois comme exemples "positifs" distincts alors qu'il s'agit
    # sémantiquement du même exemple pour le juge LLM.
    # Déduplication par (doc_idx, position du mot argmax) plutôt que par
    # chaîne de mot (B.4, AUDIT_SAE_2026-08.md) : la boucle ci-dessous ne
    # visite déjà chaque d_idx qu'une fois, donc (doc_idx, word_span) est
    # automatiquement unique -- l'ancien filtre sur la chaîne du mot excluait
    # en pratique tout mot-cible répété d'un document à l'autre, empêchant
    # une feature authentiquement lexicale (le même mot dans des contextes
    # variés, cf. Latent Terms ~33% de features purement lexicales) d'être
    # présentée sous sa forme la plus convaincante au juge.
    sorted_desc = np.argsort(f_acts)[::-1]
    pos_examples = []
    pos_magnitudes = []
    for d_idx in sorted_desc:
        if f_acts[d_idx] <= threshold_pos:
            break
        if not fragment_exists(token_fragments_dir, int(d_idx + offset)):
            continue
        doc_data = load_fragment(token_fragments_dir, int(d_idx + offset))
        token_acts = feature_column(doc_data, f_idx)
        max_act = token_acts.max()
        if max_act <= threshold_pos:
            continue
        target_idx = int(token_acts.argmax())
        ctx = extract_causal_context(doc_data["token_strings"], target_idx)
        pos_examples.append(ctx)
        pos_magnitudes.append(float(max_act))
        if len(pos_examples) >= n_pos:
            break

    # Négatif : doc avec activation nulle ou quasi-nulle. Graine locale par feature
    # (pas random.shuffle sur le module global) : un rejeu du même f_idx reconstruit
    # le MÊME négatif, condition nécessaire pour comparer un score rejugé à un score
    # en cache sans confondre "négatif différent" et "juge différent".
    neg_pool = np.where(f_acts <= threshold_neg)[0].tolist()
    random.Random(f_idx).shuffle(neg_pool)
    # B.5 : neg_quantile=0.05 ne garantit pas une activation nulle pour une
    # feature dense -- le 5e percentile peut être strictement positif. Garde
    # le candidat de plus faible magnitude réelle parmi ceux examinés plutôt
    # qu'un seuil dur (candidat > threshold_pos -> rejeté) : sur les features
    # de l'extension (sélectionnées par magnitude, donc denses par
    # construction, cf. B.2), AUCUN candidat du pool ne passe jamais un seuil
    # à threshold_pos=1e-6 -- un seuil dur annule silencieusement neg_example
    # pour la quasi-totalité des features, vérifié sur GPU (job 44831) avant
    # ce correctif. "Meilleur candidat trouvé" reste toujours une amélioration
    # sur l'ancien comportement (premier candidat du pool, sans égard à sa
    # magnitude réelle), sans reproduire l'échec total du seuil dur.
    best_example, best_magnitude = None, None
    for d_idx in neg_pool[:20]:
        if not fragment_exists(token_fragments_dir, int(d_idx + offset)):
            continue
        doc_data = load_fragment(token_fragments_dir, int(d_idx + offset))
        token_acts = feature_column(doc_data, f_idx)
        candidate_magnitude = float(token_acts.max())
        if best_magnitude is not None and candidate_magnitude >= best_magnitude:
            continue
        toks = doc_data["token_strings"]
        # B.3 : argmax de CETTE feature sur ce document non-activant, pas le
        # milieu du document -- même construction que les positifs (contexte
        # autour de l'argmax), pour que la seule différence entre positifs et
        # négatif soit la présence du concept, pas un artefact de position/
        # saillance (explication mécanique plausible de l'instabilité à 31%
        # du protocole odd-one-out, RESULTS_TESTS.md §13.1).
        target_idx = int(token_acts.argmax())
        best_example = extract_causal_context(toks, target_idx)
        best_magnitude = candidate_magnitude
        if best_magnitude <= threshold_pos:
            break  # vrai négatif trouvé, inutile de continuer

    neg_example = best_example
    neg_magnitude = best_magnitude if best_magnitude is not None else 0.0

    if return_magnitudes:
        return pos_examples, neg_example, pos_magnitudes, neg_magnitude
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
    lo: int = 0,                # borne basse (incluse) de la plage d'indices candidats
    hi: int = None,             # borne haute (exclue) — None = d_sae
) -> list[int]:
    """
    Sélection des n_features features par mean activation magnitude sur tokens
    (pas par fréquence sur doc max-pool), RESTREINTE à la plage [lo, hi).
    Découplage frozen-core / extension :
      - core     : lo=0,       hi=d_core
      - extended : lo=d_core,  hi=d_core+D_EXTRA
    Sélectionner séparément dans chaque plage garantit que le top-N d'une partie
    n'est jamais écrasé par les magnitudes de l'autre (les activations JumpReLU
    du core, non bornées, dominent systématiquement celles de l'extension TopK).
    Échantillonne sample_docs documents pour éviter OOM.
    """
    hi = d_sae if hi is None else hi
    sample_docs = min(sample_docs, len(doc_indices))
    sampled = random.sample(doc_indices, sample_docs)
    acc = np.zeros(d_sae, dtype=np.float64)
    n_tokens = 0
    for d_idx in sampled:
        if not fragment_exists(token_fragments_dir, d_idx):
            continue
        frag = load_fragment(token_fragments_dir, d_idx)
        s = sum_columns(frag)
        acc[:len(s)] += s[:d_sae]
        n_tokens += frag["shape"][0]
    if n_tokens == 0:
        return list(range(lo, min(lo + n_features, hi)))
    mean_mag = acc[lo:hi] / n_tokens
    return (np.argsort(mean_mag)[::-1][:n_features] + lo).tolist()


def feature_selection_stratified_by_frequency(
    token_fragments_dir: str,
    doc_indices: list[int],
    d_sae: int,
    n_features: int,
    sample_docs: int = 500,
    lo: int = 0,
    hi: int = None,
    n_bins: int = 10,
    seed: int = 0,
) -> list[int]:
    """
    Sélection stratifiée par bins de fréquence log-espacés (interp-embed, App. J)
    plutôt que par magnitude moyenne. `feature_selection_by_magnitude` sélectionne
    systématiquement les features les plus DENSES (magnitude token-level la plus
    forte), qui sont aussi les plus proches de directions génériques/stop-word --
    le taux d'interprétabilité mesuré sur cet échantillon n'est alors comparable ni
    à un chiffre publié (Bills et al. échantillonnent au hasard, EleutherAI/Paulo
    stratifient) ni entre deux configurations du dépôt dès que la distribution de
    magnitude change (K_EXTRA, largeur, couche, core vs extension) --
    AUDIT_SAE_2026-08.md, item B.2.

    Fréquence = fraction des documents échantillonnés où la feature est active
    (max-pool documentaire > 0, pas magnitude). Les features mortes (fréquence
    nulle) sur l'échantillon sont exclues -- aucun exemple positif n'existerait
    pour elles de toute façon (cf. `build_feature_examples_with_control`).
    Échantillonnage aléatoire (`np.random.default_rng(seed)`) DANS chaque bin,
    pas les n_features/n_bins premières par indice -- éviter un biais positionnel
    au sein d'un bin qui remplacerait le biais de magnitude par un autre biais.
    """
    hi = d_sae if hi is None else hi
    sample_docs = min(sample_docs, len(doc_indices))
    sampled = random.sample(doc_indices, sample_docs)
    freq = np.zeros(hi - lo, dtype=np.float64)
    n_docs_seen = 0
    for d_idx in sampled:
        if not fragment_exists(token_fragments_dir, d_idx):
            continue
        frag = load_fragment(token_fragments_dir, d_idx)
        doc_vec = doc_maxpool(frag).numpy()
        freq += (doc_vec[lo:hi] > 1e-6).astype(np.float64)
        n_docs_seen += 1
    if n_docs_seen == 0:
        return list(range(lo, min(lo + n_features, hi)))
    freq /= n_docs_seen

    alive_idx = np.nonzero(freq > 0)[0]
    if len(alive_idx) == 0:
        return list(range(lo, min(lo + n_features, hi)))
    if len(alive_idx) <= n_features:
        return (alive_idx + lo).tolist()

    log_freq = np.log10(freq[alive_idx])
    lo_edge, hi_edge = log_freq.min(), log_freq.max()
    if lo_edge == hi_edge:  # toutes les features vivantes à la même fréquence -- un seul bin
        bin_ids = np.zeros(len(alive_idx), dtype=int)
        n_bins_eff = 1
    else:
        bin_edges = np.linspace(lo_edge, hi_edge, n_bins + 1)
        bin_ids = np.clip(np.digitize(log_freq, bin_edges[1:-1]), 0, n_bins - 1)
        n_bins_eff = n_bins

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_bin = max(1, n_features // n_bins_eff)
    for b in range(n_bins_eff):
        members = alive_idx[bin_ids == b]
        if len(members) == 0:
            continue
        take = min(per_bin, len(members))
        selected.extend(rng.choice(members, size=take, replace=False).tolist())

    remaining = n_features - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(alive_idx, np.array(selected, dtype=alive_idx.dtype))
        if len(pool) > 0:
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            selected.extend(extra.tolist())

    return (np.array(selected[:n_features], dtype=int) + lo).tolist()


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
    batch_size: int = 16,
) -> dict:
    """
    Pour chaque feature :
      1. Présente 9 exemples positifs + 1 négatif (shufflés) au LLM.
      2. Demande lequel est l'intrus → score interp ∈ {0, 1}.
      3. Si interprétable : génère label + description.
      4. Calcule ρ_interp (Spearman) entre prédiction LLM et activations réelles.

    Retourne dict { f_idx: { label, brief_description, interp_score, rho_interp } }.

    Les 3 étapes sont batchées séparément (audit perf §2.6, item 1 : 3 appels
    `model.generate` en bs=1 par feature -> 33 min pour 500 features, bornées
    par la bande passante mémoire, pas le calcul). Chaque étape ne s'applique
    qu'au SOUS-ENSEMBLE de features qui l'atteint (étape 2/3 : seulement les
    features interprétables), donc 3 passes sur des lots décroissants plutôt
    qu'une seule passe uniforme -- la construction des exemples (étape 0,
    CPU) reste séquentielle et dans le MÊME ordre que l'ancien code pour que
    `random.shuffle` produise les mêmes tirages qu'avant à graine égale.
    """
    from scipy.stats import spearmanr

    model.eval()
    results = {}
    per_feature = {}

    # ── Étape 0 : construction des exemples (CPU, inchangée) ──────────────
    for f_idx in feature_indices:
        pos_examples, neg_example, pos_magnitudes, neg_magnitude = build_feature_examples_with_control(
            f_idx, token_fragments_dir, acts, offset=offset, n_pos=n_pos, return_magnitudes=True,
        )

        if len(pos_examples) < 3:
            results[f_idx] = {
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
        correct_answer = indices.index(neg_position) + 1 if neg_example else None  # 1-based

        per_feature[f_idx] = {
            "pos_examples": pos_examples, "neg_example": neg_example,
            "pos_magnitudes": pos_magnitudes, "neg_magnitude": neg_magnitude,
            "shuffled": shuffled, "indices": indices, "correct_answer": correct_answer,
        }

    live_features = list(per_feature.keys())
    if not live_features:
        return results

    # ── Étape 1 : Odd-one-out, batché sur toutes les features vivantes ────
    ood_messages = []
    for f_idx in live_features:
        examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(per_feature[f_idx]["shuffled"]))
        prompt_ood = (
            "Voici des exemples de textes où une feature neuronale est fortement activée "
            "(sauf un, qui est un contrôle négatif).\n\n"
            f"{examples_text}\n\n"
            "Quel numéro est l'intrus (celui qui ne partage pas le concept commun des autres) ? "
            "Réponds uniquement avec le numéro."
        )
        ood_messages.append([{"role": "user", "content": prompt_ood}])

    ood_responses = _batched_generate(model, tokenizer, ood_messages, max_new_tokens=8, batch_size=batch_size)

    interp_features = []
    for f_idx, resp_ood in zip(live_features, ood_responses):
        try:
            predicted = int(re.search(r"\d+", resp_ood).group())
        except Exception:
            predicted = -1
        correct_answer = per_feature[f_idx]["correct_answer"]
        interp_score = int(predicted == correct_answer) if correct_answer is not None else 0
        per_feature[f_idx]["interp_score"] = interp_score
        per_feature[f_idx]["label_data"] = {"label": f"Feature_{f_idx}", "brief_description": "Non interprétable."}
        if interp_score == 1:
            interp_features.append(f_idx)

    # ── Étape 2 : Label, batché sur les features interprétables uniquement ─
    label_messages = []
    for f_idx in interp_features:
        formatted = "\n".join(f"- {ex}" for ex in per_feature[f_idx]["pos_examples"])
        prompt_label = (
            "Ces exemples textuels activent tous fortement une même feature neuronale "
            "(les mots déclencheurs sont entre << >>).\n\n"
            f"{formatted}\n\n"
            "Génère un objet JSON avec un label court en français (≤3 mots) et une description concise :\n"
            '{"label": "...", "brief_description": "..."}'
        )
        label_messages.append([{"role": "user", "content": prompt_label}])

    label_responses = (
        _batched_generate(model, tokenizer, label_messages, max_new_tokens=128, batch_size=batch_size)
        if label_messages else []
    )
    for f_idx, resp_l in zip(interp_features, label_responses):
        try:
            per_feature[f_idx]["label_data"] = json.loads(re.search(r"\{.*?\}", resp_l, re.DOTALL).group())
        except Exception:
            pass

    # ── Étape 3 : ρ_interp (Bills 2023), batché sur interp + neg_example ───
    # LLM score chaque exemple (pos + neg) sur [0, 10] ; Spearman vs activation réelle
    score_features = [f for f in interp_features if per_feature[f]["neg_example"]]
    score_messages = []
    for f_idx in score_features:
        label_str = per_feature[f_idx]["label_data"].get("label", "")
        score_prompts = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(per_feature[f_idx]["shuffled"]))
        prompt_score = (
            f"Concept : « {label_str} »\n\n"
            "Pour chaque exemple ci-dessous, note de 0 (non lié) à 10 (fortement lié) "
            "l'intensité du lien avec ce concept. "
            "Réponds uniquement avec un JSON : {\"scores\": [s1, s2, ...]}\n\n"
            f"{score_prompts}"
        )
        score_messages.append([{"role": "user", "content": prompt_score}])

    score_responses = (
        _batched_generate(model, tokenizer, score_messages, max_new_tokens=128, batch_size=batch_size)
        if score_messages else []
    )
    for f_idx, resp_s in zip(score_features, score_responses):
        rho_interp = float("nan")
        try:
            scores_llm = json.loads(re.search(r"\{.*?\}", resp_s, re.DOTALL).group())["scores"]
            # Magnitude d'activation RÉELLE (pas un rang synthétique, pas 0.0 fixe
            # pour le négatif) -- fidèle à la définition de Bills et al. 2023 :
            # ρ_interp corrèle le score du juge à l'activation réelle.
            # all_magnitudes est dans le même ordre que all_examples (avant
            # mélange) ; réindexé ici dans l'ordre shufflé effectivement présenté.
            neg_example = per_feature[f_idx]["neg_example"]
            all_magnitudes = per_feature[f_idx]["pos_magnitudes"] + ([per_feature[f_idx]["neg_magnitude"]] if neg_example else [])
            act_ground = [all_magnitudes[orig_idx] for orig_idx in per_feature[f_idx]["indices"]]
            if len(scores_llm) == len(act_ground):
                rho_interp = float(spearmanr(scores_llm, act_ground).statistic)
        except Exception:
            pass
        per_feature[f_idx]["rho_interp"] = rho_interp

    # ── Assemblage ──────────────────────────────────────────────────────────
    for f_idx in live_features:
        pf = per_feature[f_idx]
        results[f_idx] = {
            **pf["label_data"],
            "interp_score": pf["interp_score"],
            "rho_interp": pf.get("rho_interp", float("nan")),
            "pos_examples": pf["pos_examples"],
            "neg_example": pf["neg_example"],  # cf. dashboard (exemples négatifs) -- absent des caches produits avant cet ajout
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
    return_magnitudes: bool = False,
):
    """Équivalent phrase-level de build_feature_examples_with_control : pas de
    fragments à charger, la phrase elle-même est l'exemple. Déduplication sur
    le texte de la phrase (nettoyé) pour éviter les répétitions.

    `return_magnitudes=True` (défaut False, rétrocompatible) : retourne en plus
    (pos_magnitudes, neg_magnitude), l'activation réelle de chaque exemple --
    même correctif que build_feature_examples_with_control (B.5)."""
    f_acts = phrase_acts[:, f_idx].detach().float().numpy()
    threshold_pos = 1e-6
    threshold_neg = float(np.quantile(f_acts, neg_quantile))

    sorted_desc = np.argsort(f_acts)[::-1]
    pos_examples = []
    pos_magnitudes = []
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
        pos_magnitudes.append(float(f_acts[p_idx]))
        if len(pos_examples) >= n_pos:
            break

    # Graine locale par feature (B.28) -- cf. build_feature_examples_with_control.
    neg_pool = np.where(f_acts <= threshold_neg)[0].tolist()
    random.Random(f_idx).shuffle(neg_pool)
    # B.5 : garde le candidat de plus faible activation réelle plutôt qu'un
    # seuil dur -- cf. build_feature_examples_with_control (un seuil dur
    # annule silencieusement neg_example pour les features denses).
    best_example, best_magnitude = None, None
    for p_idx in neg_pool[:20]:
        text = re.sub(r"\s+", " ", phrase_texts[p_idx]).strip()
        if not text:
            continue
        candidate_magnitude = float(f_acts[p_idx])
        if best_magnitude is not None and candidate_magnitude >= best_magnitude:
            continue
        best_example = f"<<{text}>>"
        best_magnitude = candidate_magnitude
        if best_magnitude <= threshold_pos:
            break

    neg_example = best_example
    neg_magnitude = best_magnitude if best_magnitude is not None else 0.0

    if return_magnitudes:
        return pos_examples, neg_example, pos_magnitudes, neg_magnitude
    return pos_examples, neg_example


def local_gemma_judge(
    model,
    tokenizer,
    feature_indices: list[int],
    phrase_texts: list,
    phrase_acts: torch.Tensor,
    phrase_to_doc: Optional[np.ndarray] = None,
    n_pos: int = 9,
    batch_size: int = 16,
) -> dict:
    """
    Labellisation locale (Gemma-3) des features du Phrase-Level SAE (Pipeline 2).
    Même protocole odd-one-out + ρ_interp que odd_one_out_judge (Pipeline 1),
    mais construit directement sur les phrases (pas de fragments tokens à charger).
    Même batching en 3 passes par sous-ensemble décroissant, cf. docstring
    d'odd_one_out_judge (audit perf §2.6, item 1).

    `phrase_to_doc` n'est pas requis pour la labellisation elle-même (conservé
    pour compat/signature future si besoin de contexte inter-phrase).
    """
    from scipy.stats import spearmanr

    model.eval()
    results = {}
    per_feature = {}

    # ── Étape 0 : construction des exemples (CPU, inchangée) ──────────────
    for f_idx in feature_indices:
        pos_examples, neg_example, pos_magnitudes, neg_magnitude = build_phrase_examples_with_control(
            f_idx, phrase_texts, phrase_acts, n_pos=n_pos, return_magnitudes=True,
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

        per_feature[f_idx] = {
            "pos_examples": pos_examples, "neg_example": neg_example,
            "pos_magnitudes": pos_magnitudes, "neg_magnitude": neg_magnitude,
            "shuffled": shuffled, "indices": indices, "correct_answer": correct_answer,
        }

    live_features = list(per_feature.keys())
    if not live_features:
        return results

    # ── Étape 1 : Odd-one-out, batché sur toutes les features vivantes ────
    ood_messages = []
    for f_idx in live_features:
        examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(per_feature[f_idx]["shuffled"]))
        prompt_ood = (
            "Voici des phrases où une feature neuronale est fortement activée "
            "(sauf une, qui est un contrôle négatif). Le mot/groupe déclencheur "
            "est entre << >>.\n\n"
            f"{examples_text}\n\n"
            "Quel numéro est l'intrus (celui qui ne partage pas le concept commun des autres) ? "
            "Réponds uniquement avec le numéro."
        )
        ood_messages.append([{"role": "user", "content": prompt_ood}])

    ood_responses = _batched_generate(model, tokenizer, ood_messages, max_new_tokens=8, batch_size=batch_size)

    interp_features = []
    for f_idx, resp_ood in zip(live_features, ood_responses):
        try:
            predicted = int(re.search(r"\d+", resp_ood).group())
        except Exception:
            predicted = -1
        correct_answer = per_feature[f_idx]["correct_answer"]
        interp_score = int(predicted == correct_answer) if correct_answer is not None else 0
        per_feature[f_idx]["interp_score"] = interp_score
        per_feature[f_idx]["label_data"] = {"label": f"Feature_{f_idx}", "brief_description": "Non interprétable."}
        if interp_score == 1:
            interp_features.append(f_idx)

    # ── Étape 2 : Label, batché sur les features interprétables uniquement ─
    label_messages = []
    for f_idx in interp_features:
        formatted = "\n".join(f"- {ex}" for ex in per_feature[f_idx]["pos_examples"])
        prompt_label = (
            "Ces phrases activent toutes fortement une même feature neuronale "
            "(les mots/groupes déclencheurs sont entre << >>).\n\n"
            f"{formatted}\n\n"
            "Génère un objet JSON avec un label court en français (≤3 mots) et une description concise :\n"
            '{"label": "...", "brief_description": "..."}'
        )
        label_messages.append([{"role": "user", "content": prompt_label}])

    label_responses = (
        _batched_generate(model, tokenizer, label_messages, max_new_tokens=128, batch_size=batch_size)
        if label_messages else []
    )
    for f_idx, resp_l in zip(interp_features, label_responses):
        try:
            per_feature[f_idx]["label_data"] = json.loads(re.search(r"\{.*?\}", resp_l, re.DOTALL).group())
        except Exception:
            pass

    # ── Étape 3 : ρ_interp, batché sur interp + neg_example ────────────────
    score_features = [f for f in interp_features if per_feature[f]["neg_example"]]
    score_messages = []
    for f_idx in score_features:
        label_str = per_feature[f_idx]["label_data"].get("label", "")
        score_prompts = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(per_feature[f_idx]["shuffled"]))
        prompt_score = (
            f"Concept : « {label_str} »\n\n"
            "Pour chaque exemple ci-dessous, note de 0 (non lié) à 10 (fortement lié) "
            "l'intensité du lien avec ce concept. "
            "Réponds uniquement avec un JSON : {\"scores\": [s1, s2, ...]}\n\n"
            f"{score_prompts}"
        )
        score_messages.append([{"role": "user", "content": prompt_score}])

    score_responses = (
        _batched_generate(model, tokenizer, score_messages, max_new_tokens=128, batch_size=batch_size)
        if score_messages else []
    )
    for f_idx, resp_s in zip(score_features, score_responses):
        rho_interp = float("nan")
        try:
            scores_llm = json.loads(re.search(r"\{.*?\}", resp_s, re.DOTALL).group())["scores"]
            # Magnitude réelle, pas un rang synthétique (B.5) -- cf. odd_one_out_judge.
            neg_example = per_feature[f_idx]["neg_example"]
            all_magnitudes = per_feature[f_idx]["pos_magnitudes"] + ([per_feature[f_idx]["neg_magnitude"]] if neg_example else [])
            act_ground = [all_magnitudes[orig_idx] for orig_idx in per_feature[f_idx]["indices"]]
            if len(scores_llm) == len(act_ground):
                rho_interp = float(spearmanr(scores_llm, act_ground).statistic)
        except Exception:
            pass
        per_feature[f_idx]["rho_interp"] = rho_interp

    # ── Assemblage ──────────────────────────────────────────────────────────
    for f_idx in live_features:
        pf = per_feature[f_idx]
        results[str(f_idx)] = {
            **pf["label_data"],
            "interp_score": pf["interp_score"],
            "rho_interp": pf.get("rho_interp", float("nan")),
            "pos_examples": pf["pos_examples"],
            "neg_example": pf["neg_example"],  # cf. dashboard (exemples négatifs) -- absent des caches produits avant cet ajout
        }

    return results