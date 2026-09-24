"""
retrieval_rerank.py — Reranking LLM du top-k (App. G, "Combining results and
second stage retrieval", arXiv:2512.10092v2, docs/archive/references/PDF_APPENDICES_EXTRACT.md
lignes 640-647) : "we also add in LLM reranking of the top 50" -- le papier
ne publie pas le prompt de reranking utilisé (contrairement au juge
odd-one-out/vérification, App. C/K), seulement le protocole ("top 50") et
son effet sur MAP/MP@10 (Table 20).

Écart de protocole documenté (R6) : à défaut du prompt original, reranking
POINTWISE (note 0-10 de pertinence par document, même convention que
`odd_one_out_judge`'s étape ρ_interp, `src/sae/judge.py`) plutôt que
listwise (une requête demandant au LLM de réordonner N documents en un seul
appel est plus sujette à erreur de parsing/troncature pour N=50, et le
papier ne précise pas non plus la méthode -- pointwise reste comparable en
esprit : "LLM reranking", pas nécessairement le mécanisme exact).
"""
from __future__ import annotations

import json
import re

from src.sae.judge import _batched_generate

RERANK_PROMPT = """QUERY: {query}

DOCUMENT:
{document}

TASK: Rate how relevant this document is to the query, from 0 (not relevant at all)
to 10 (perfectly relevant).

Respond with a JSON object: {{"relevance": <integer 0-10>}}"""


def _parse_relevance_score(response: str) -> float:
    """0.0 en repli si la réponse n'est pas parsable (JSON absent/malformé) --
    place le document en fin de classement plutôt que de faire planter le
    reranking sur une réponse isolée mal formée."""
    try:
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        score = float(json.loads(match.group())["relevance"])
        return max(0.0, min(10.0, score))
    except Exception:
        return 0.0


def llm_rerank(
    model,
    tokenizer,
    query: str,
    ranked_doc_ids: list,
    doc_texts: dict,
    top_k: int = 50,
    batch_size: int = 16,
) -> list:
    """Rerank les `top_k` premiers documents de `ranked_doc_ids` (déjà classés
    par une méthode de base -- RRF, Latent Terms, TF-IDF...) par score de
    pertinence LLM pointwise, laisse le reste de la liste inchangé après
    (App. G : reranking du "top 50" seulement, pas de tout le corpus classé).

    `doc_texts` : dict doc_id -> texte (évite de retransmettre le corpus
    entier, seuls les `top_k` textes nécessaires sont formatés en prompt).

    Retourne la liste complète des doc_ids, top_k en tête reclassés par score
    LLM décroissant (égalités : ordre du classement de base préservé, tri
    stable), puis le reste de `ranked_doc_ids` inchangé."""
    head = ranked_doc_ids[:top_k]
    tail = ranked_doc_ids[top_k:]
    if not head:
        return list(tail)

    messages = [
        [{"role": "user", "content": RERANK_PROMPT.format(query=query, document=doc_texts[d])}]
        for d in head
    ]
    responses = _batched_generate(model, tokenizer, messages, max_new_tokens=32, batch_size=batch_size)
    scores = [_parse_relevance_score(r) for r in responses]

    reranked_head = [d for _, d in sorted(zip(scores, head), key=lambda sd: -sd[0])]
    return reranked_head + list(tail)
