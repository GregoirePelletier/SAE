"""
diff_hypothesis_generator.py — Génération d'hypothèses structurées (App. D.2,
arXiv:2512.10092v2, prompt verbatim reproduit dans
docs/archive/references/PDF_APPENDICES_EXTRACT.md lignes 216-263) : convertit une liste de
features SAE discriminantes entre deux corpus en un tableau JSON d'hypothèses
concises, chacune citant les features qui la supportent, un sens
("target"/"other"), une force de différence, une confiance.

Remplace `generate_llm_diff_hypothesis` (saev5.py) pour tout usage qui a
besoin d'hypothèses STRUCTURÉES vérifiables (`hypothesis_verifier.py`) plutôt
que d'un paragraphe libre — les deux coexistent, `generate_llm_diff_hypothesis`
reste le résumé "une phrase" affiché dans `results.json`.

Écart de protocole documenté (R6) : le nombre de features réellement passées
au prompt dépend de `select_top_diff_features_by_frequency`
(src/analysis/cooccurrence.py) et du corpus de diffing du dépôt (souvent bien
plus restreint que celui du papier) — peut être très inférieur à 200.
"""
from __future__ import annotations

import json
import re

from src.sae.judge import _batched_generate

# Prompt VERBATIM (App. D.2, PDF fait foi -- convention de mission) : ne pas
# reformuler, seuls {query}/{num_hypotheses}/{features_block} sont substitués.
DIFF_HYPOTHESIS_PROMPT = """You are analyzing differences between two datasets. Below are the most significant features that are
differences between a "target" and "other" dataset:
IMPORTANT NOTES:
1. The << >> markers in examples indicate WHERE features activated, but you should NOT restrict your
understanding to just those marked tokens. The context BEFORE the marked tokens often provides crucial
information about what the feature is detecting.
2. Features often respond to patterns that span both the preceding context AND the marked tokens together.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature
activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end
of text, not a meaningful activation.
4. Note that some features are not accurate. If the feature description does not accurately describe the
tokens marked with << >>, you should disregard the feature. Only use features that you are certain are
valid.
5. Please ensure that all hypothesis descriptions are clearly distinct from each other. You do not need to
generate the exact amount of hypotheses to meet the quota.
6. Each feature will have a "difference strength", which is the percentage difference between the target and
other dataset. If it is positive, the target dataset has more of the feature than the other dataset. If
it is negative, the other dataset has more of the feature than the target dataset.
7. Please try to make each hypothesis specific, focused, and distinct from each other.

FEATURES:
{features_block}

USER QUERY: {query}

Generate at most {num_hypotheses} hypotheses that answer the user's query for the "target" dataset. I'm
looking for differences of the format Dataset A is more X than Dataset B, where X is the difference.
Each hypothesis should be formatted as a JSON object with these exact fields:
- "dataset": "target" or "other" (the dataset that has more of this property)
- "description": Describe a response that would validly have property X. Start with "This response .." Use 1-2
sentences to clearly and specifically describe the property, such that using this description could be
used to identify the property on its own. Do not mention the model names. Be specific so that responses
that don't have this property could not be misclassified as having this property based on this
description.
- "feature_ids": List of feature ID(s) that support this hypothesis. It could be a list of a single feature ID
, or a list of multiple feature IDs.
- "examples": List of examples. Provide at most 3 examples. Be concise. For each example, cite the feature ID
and feature description and explain how the positive / negative example pairs from the dataset
illustrate the hypothesis, considering both the marked tokens AND their preceding context). You should
just highlight the portion of the example pairs that are relevant for the feature; do not print out the
entire positive / negative example pairs unless it is necessary to understand the feature.
- "percentage_difference": 0.XX (the percentage difference, between -1 and 1). Use the maximum difference
strength among the features used. Positive percentage if target has more of this property, negative
otherwise.
- "confidence": 0.XX (confidence in this hypothesis, between 0 and 1)
Remember that <eot_id> tokens should be ignored as they are just EOS markers, not meaningful feature
activations.
Return the response as a JSON array of at most {num_hypotheses} hypothesis objects. Make sure the JSON is
valid and can be parsed directly."""

REQUIRED_FIELDS = ("dataset", "description", "feature_ids", "examples", "percentage_difference", "confidence")


def _format_feature_block(feature: dict) -> str:
    """Une entrée `FEATURES:` -- id, description, force de différence, un
    exemple positif et un exemple négatif si disponibles (marqueurs << >>
    dans les exemples, convention `build_feature_examples_with_control`,
    `src/sae/judge.py`). `pos_example`/`neg_example` optionnels (R6) : quand
    seul le label déjà produit par le juge est disponible (pas de
    ré-extraction de fragments token-level pour le corpus de diffing), le
    prompt reste valide sans exemple illustré -- moins riche que le protocole
    complet du papier, mais le champ `description` du label lui-même porte
    déjà l'information utile."""
    lines = [
        f"Feature ID: {feature['feature_id']}",
        f"Description: {feature['label']}",
        f"Difference strength: {feature['percentage_difference']:+.2f}",
    ]
    if feature.get("pos_example"):
        lines.append(f"Positive example: {feature['pos_example']}")
    if feature.get("neg_example"):
        lines.append(f"Negative example: {feature['neg_example']}")
    return "\n".join(lines)


def _parse_hypotheses_json(response: str) -> list[dict]:
    """Extrait le tableau JSON de la réponse -- repli sur liste vide si le
    JSON n'est pas parsable (réponse tronquée/malformée), jamais une
    exception qui ferait perdre les hypothèses des autres appels d'un même
    lot (cf. appelant, une génération par lot logique, pas par hypothèse)."""
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            return []
        return [h for h in parsed if isinstance(h, dict) and all(k in h for k in REQUIRED_FIELDS)]
    except Exception:
        return []


def generate_structured_diff_hypotheses(
    model,
    tokenizer,
    features: list[dict],
    query: str,
    num_hypotheses: int = 10,
    max_new_tokens: int = 2048,
) -> list[dict]:
    """Génère au plus `num_hypotheses` hypothèses structurées (App. D.2) à
    partir de `features` (chaque entrée : `feature_id`, `label`,
    `percentage_difference`, `pos_example`, `neg_example` -- typiquement
    `select_top_diff_features_by_frequency` + `build_feature_examples_with_
    control` par feature). UN SEUL appel LLM (pas un par feature, contrairement
    à `hypothesis_verifier.verify_hypotheses`) : le prompt App. D.2 résume
    l'ensemble des features en une seule passe, c'est son rôle (les
    descriptions de features individuelles peuvent se recouper)."""
    features_block = "\n\n".join(_format_feature_block(f) for f in features)
    prompt = DIFF_HYPOTHESIS_PROMPT.format(
        features_block=features_block, query=query, num_hypotheses=num_hypotheses,
    )
    responses = _batched_generate(model, tokenizer, [[{"role": "user", "content": prompt}]],
                                   max_new_tokens=max_new_tokens, batch_size=1)
    return _parse_hypotheses_json(responses[0])[:num_hypotheses]
