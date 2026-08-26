"""
hypothesis_verifier.py — Vérification d'hypothèses de diffing par juge LLM
(App K.1, arXiv:2512.10092v2 ; définitions de `verification_rate`/`coverage`
en Figures 11/12, cf. docs/PDF_APPENDICES_EXTRACT.md lignes 340 et 740-775).

Adaptateur de `external/interp_embed/paper/diffing/hypothesis_verifier.py::
HypothesisVerifier` vers le juge local du projet (`JUDGE_MODEL_ID`,
`src.sae.judge.load_judge_model`/`_batched_generate`) : leur implémentation
appelle `interp_embed.llm.utils.call_async_llm` (API OpenAI/OpenRouter async),
incompatible avec ce dépôt (aucune clé API configurée, juge local uniquement
-- même contrainte que `odd_one_out_judge`/`local_gemma_judge`). Le protocole
(matrice hypothèse × document, seuil de différence de fréquence vérifiée
>1%) est repris identique ; seul le mécanisme d'appel LLM change (génération
batchée locale au lieu d'appels async sur API).

Écart de prompt documenté (R6) : le prompt ci-dessous est repris VERBATIM du
PDF (§K.1 — le PDF fait foi, convention de mission) plutôt que de
`hypothesis_verifier.py::verify_hypothesis_response`, dont le prompt diffère
légèrement du PDF (6 instructions au lieu de 7 -- l'instruction 7, sur les
hypothèses formulées comme des phrases et les mentions "assistant"/"user",
est absente du code des auteurs -- et "document" vs "response text" dans la
consigne de tâche ; micro-écart déjà repéré dans
`docs/INTERP_EMBED_COVERAGE.md`).
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.sae.judge import _batched_generate

VERIFICATION_PROMPT = """You are an expert at analyzing whether text exhibits specific properties or characteristics.

HYPOTHESIS: {hypothesis_description}

RESPONSE TEXT TO ANALYZE:
{response}

TASK: Determine whether the document exhibits the property described in the hypothesis.

INSTRUCTIONS:
1. Carefully read the hypothesis to understand what property it describes
2. Analyze the document to see if it clearly embodies that property.
3. Consider both explicit and implicit manifestations of the property
4. Be consistent and objective in your evaluation
5. If you are unsure, answer "NO"
6. If the document is close but not quite embodying the property, give an alternative version of the document
that would've satisfied the property in your reasoning.
7. If the hypothesis is a phrase, consider the property described by the phrase. Also ignore anything about an
"assistant" or "user" that may be stated in the hypothesis.

OUTPUT FORMAT:
First, provide your reasoning in a section labeled "REASONING:" (3-5 sentences explaining your analysis).
Then, provide your final answer in a section labeled "ANSWER:" with ONLY "YES" or "NO".

Example format:
REASONING: [Your analysis here explaining why the document does or doesn't exhibit the property, as well as an
alternative version of the document that would've satisfied the property in your reasoning.]
ANSWER: YES/NO

Your response:"""


def _parse_verification_answer(full_response: str) -> bool:
    """Même logique de repli que hypothesis_verifier.py::verify_hypothesis_response :
    section ANSWER: si présente, sinon le texte entier ; toute réponse qui ne
    commence pas par YES est traitée comme NO (repli sûr, cf. consigne 5 du prompt)."""
    tail = full_response.split("ANSWER:")[-1] if "ANSWER:" in full_response else full_response
    return tail.strip().upper().startswith("YES")


def verify_hypotheses(
    model,
    tokenizer,
    hypotheses: list,
    documents: list,
    batch_size: int = 16,
) -> np.ndarray:
    """Matrice (n_hypotheses, n_documents) de vérification booléenne (App K.1).

    Un appel `_batched_generate` par hypothèse (tous les documents partagent
    le même texte d'hypothèse, seul le document varie dans le prompt) --
    même mécanique de batching que `odd_one_out_judge`/`local_gemma_judge`
    (`src/sae/judge.py`), pas de logique de batching dupliquée ici.
    """
    n_h, n_d = len(hypotheses), len(documents)
    matrix = np.zeros((n_h, n_d), dtype=int)
    for h_idx, hyp in enumerate(hypotheses):
        messages = [
            [{"role": "user", "content": VERIFICATION_PROMPT.format(hypothesis_description=hyp, response=doc)}]
            for doc in documents
        ]
        responses = _batched_generate(model, tokenizer, messages, max_new_tokens=400, batch_size=batch_size)
        for d_idx, resp in enumerate(responses):
            matrix[h_idx, d_idx] = int(_parse_verification_answer(resp))
    return matrix


def compute_verification_metrics(
    matrix: np.ndarray,
    group_mask: np.ndarray,
    hypothesis_labels: Optional[list] = None,
    threshold: float = 0.01,
) -> tuple:
    """Métriques App K.1 (docs/PDF_APPENDICES_EXTRACT.md lignes 340, 742).

    - `verification_rate` : fraction des hypothèses dont la différence de
      fréquence vérifiée |rate_in_group - rate_out_group| dépasse `threshold`
      (>1% dans le papier, Figure 11) -- métrique de fidélité comparable au
      papier pour le diffing (aucune autre ne l'est dans ce dépôt à ce jour,
      cf. AUDIT_SAE_2026-08.md §7).
    - `coverage` : fraction des documents du groupe cible (`group_mask`)
      couverts par AU MOINS une hypothèse valide dans le sens
      target > reste, vérifiée vraie sur ce document précis (Figure 12 :
      "% de réponses où au moins une hypothèse s'applique uniquement à la
      target").

    `group_mask` : booléen, True = groupe cible (le sens dans lequel la
    couverture est mesurée) ; même colonnes que `matrix`. `threshold` en
    fraction (0.01 = 1 point de pourcentage), pas en pourcentage.

    Retourne (per_hypothesis: DataFrame, summary: dict).
    """
    n_h, n_d = matrix.shape
    group_mask = np.asarray(group_mask, dtype=bool)
    if hypothesis_labels is None:
        hypothesis_labels = [f"H{i}" for i in range(n_h)]

    rate_in = matrix[:, group_mask].mean(axis=1) if group_mask.any() else np.full(n_h, np.nan)
    rate_out = matrix[:, ~group_mask].mean(axis=1) if (~group_mask).any() else np.full(n_h, np.nan)
    diff = rate_in - rate_out
    valid = np.abs(diff) > threshold
    target_direction_valid = valid & (diff > 0)

    per_hypothesis = pd.DataFrame({
        "hypothesis": hypothesis_labels,
        "rate_in_group": rate_in,
        "rate_out_group": rate_out,
        "verified_diff": diff,
        "valid": valid,
    })

    verification_rate = float(valid.mean()) if n_h > 0 else float("nan")

    valid_idx = np.where(target_direction_valid)[0]
    if len(valid_idx) > 0 and group_mask.any():
        covered = matrix[np.ix_(valid_idx, group_mask)].any(axis=0)
        coverage = float(covered.mean())
    else:
        coverage = 0.0

    summary = {
        "n_hypotheses": n_h,
        "n_documents_in_group": int(group_mask.sum()),
        "n_documents_out_group": int((~group_mask).sum()),
        "threshold": threshold,
        "verification_rate": verification_rate,
        "coverage": coverage,
    }
    return per_hypothesis, summary
