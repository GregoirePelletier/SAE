"""
clustering_llm.py — Génération de mots-clés LLM + union top-k de latents
(Appendix F.1), accuracy par cluster par réassignation LLM (§4.3/Appendix
K.3), z-score de conductance en espace d'embedding dense (§4.3, page 7 du
papier -- formule absente des Appendices, retrouvée dans le corps principal :
"we compute the z-score of each cluster's conductance in dense embedding
space relative to a random sample (lower = tighter)"). Complète
`src/sae/saev5.py::targeted_clustering_by_axis`, qui ne fait aujourd'hui que
la sélection par UN SEUL `axis_query` (App. F.1 prévoit une union de latents
sur PLUSIEURS mots-clés, éventuellement générés par LLM) et n'a ni étiquetage
de cluster, ni accuracy, ni z-score de conductance (docs/archive/audits/AUDIT_SAE_2026-08.md §1/§7).

Adaptateur du juge local du projet (`src.sae.judge._batched_generate`), même
raison que les autres modules `*_verified`/`hypothesis_verifier` : le code des
auteurs suppose une API OpenAI/OpenRouter absente de ce dépôt. Prompts repris
VERBATIM du PDF (Appendix F.1, K.3, et le paragraphe "Real world evaluation
metrics" p.7 pour la définition de l'accuracy).
"""

import re
from typing import Callable, Optional

import networkx as nx
import numpy as np
from sklearn.neighbors import kneighbors_graph

from src.sae.judge import _batched_generate

KEYWORD_GENERATION_PROMPT = """You are an NLP feature-brainstorming assistant.
Task: Given a user query, suggest 2 to 5+ **distinctive and semantically specific** keywords or phrases that capture the key concepts relevant to that query.
- If the goal refers to a **binary or low-dimensional** axis (e.g. sentiment, tense, polarity), return only the **most salient few items (2-4)**.
- If the axis is **broad or multi-class** (e.g. topic, genre, domain), return more **diverse sub-categories** (up to 10).
- Each item should be a **single coherent concept** that could plausibly describe the activation of a sparse autoencoder feature.
- Include contrasting pairs or subtypes when applicable (e.g. "positive", "negative").
- Avoid generic catch-alls like "style", "content", or "other".
- Return each item on its own line, without bullets or numbering.

User query: {query}"""

CLUSTER_LABELING_PROMPT = """You are an assistant for labeling clusters of natural language text.
You will be given multiple clusters at once. For each cluster, you have the top {n_relabel} distinctive features and top {n_relabel} examples.
Your task is to create DISTINCTIVE, human-like labels that capture what unites each cluster.
IMPORTANT:
- Each cluster label must be DIFFERENT from all others
- Focus on what makes each cluster UNIQUE, not just common themes
- Create natural, descriptive labels that a human would understand immediately
- Labels can be longer and more detailed if needed to capture the essence
- Look for patterns in content, tone, style, intent, or context
- Only quote specific phrases if they're extremely clear and defining
- If a cluster is truly unclear, label it "UNCLEAR"
Return your response in this exact format:
Cluster 0: [label]
Cluster 1: [label]
Cluster 2: [label]
...and so on

{clusters_block}"""

CLUSTER_ASSIGNMENT_SYSTEM_PROMPT = """You are a text-classification assistant. You are given a text, and descriptions of clusters.
Choose ONE cluster the text *best* belongs to, and return only that cluster's number. Do not simply choose the most generic cluster."""


def generate_keywords(model, tokenizer, query: str) -> list:
    """Appendix F.1, génération de mots-clés. Une ligne = un mot-clé."""
    prompt = KEYWORD_GENERATION_PROMPT.format(query=query)
    resp = _batched_generate(model, tokenizer, [[{"role": "user", "content": prompt}]], max_new_tokens=200)[0]
    return [line.strip("-* \t") for line in resp.strip().splitlines() if line.strip("-* \t")]


def select_latents_union(
    keywords: list,
    feature_labels: dict,
    select_fn: Callable[[str, dict, int], list],
    top_k: int = 100,
) -> list:
    """Union des top-k latents par mot-clé (Appendix F.1 : "the union of all
    these latents taken"). `select_fn` : fonction de similarité d'embedding
    dense déjà présente dans le dépôt (`saev5.py::select_latents_by_similarity`,
    injectée par l'appelant -- ce module reste agnostique du modèle
    d'embedding, même convention que `cooccurrence.find_interesting_pairs`)."""
    union = set()
    for kw in keywords:
        union.update(select_fn(kw, feature_labels, top_k))
    return sorted(union)


def generate_cluster_labels(
    model, tokenizer,
    clusters: list,
    n_relabel: int = 5,
) -> list:
    """Appendix F.1, "Generating cluster labels". `clusters` : liste de
    {"features": [...labels...], "examples": [...textes...]}, un élément par
    cluster, dans l'ordre. Retourne une liste de labels dans le même ordre."""
    blocks = []
    for cid, c in enumerate(clusters):
        feats = "\n".join(f"  - {f}" for f in c.get("features", [])[:n_relabel])
        exs = "\n".join(f"  - {e}" for e in c.get("examples", [])[:n_relabel])
        blocks.append(f"Cluster {cid}:\nFeatures:\n{feats}\nExamples:\n{exs}")
    clusters_block = "\n\n".join(blocks)
    prompt = CLUSTER_LABELING_PROMPT.format(n_relabel=n_relabel, clusters_block=clusters_block)
    resp = _batched_generate(model, tokenizer, [[{"role": "user", "content": prompt}]], max_new_tokens=400)[0]

    labels = [None] * len(clusters)
    for cid_str, label in re.findall(r"Cluster\s+(\d+):\s*\[?([^\n\[\]]+)\]?", resp):
        cid = int(cid_str)
        if 0 <= cid < len(clusters):
            labels[cid] = label.strip()
    return [lbl if lbl else "UNCLEAR" for lbl in labels]


def compute_cluster_accuracy(
    model, tokenizer,
    texts: list,
    cluster_ids: list,
    cluster_descriptions: dict,
    batch_size: int = 16,
) -> dict:
    """§4.3, "Real world evaluation metrics" (page 7) : "given a clustering
    and its cluster descriptions, we ask an LLM to assign each text to one
    cluster using only these descriptions, then compute the fraction of texts
    from the original cluster that remain." Prompt système K.3 verbatim.

    `cluster_ids` : assignation originale (même longueur que `texts`).
    `cluster_descriptions` : {cluster_id: description}. Retourne
    {cluster_id: accuracy} (fraction des textes du cluster original
    réassignés au MÊME cluster par le juge)."""
    options = "\n".join(f"{cid}: {desc}" for cid, desc in sorted(cluster_descriptions.items()))
    messages = [
        [
            {"role": "system", "content": CLUSTER_ASSIGNMENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"TEXT:\n{text}\n\nCLUSTERS:\n{options}\n\nCluster number:"},
        ]
        for text in texts
    ]
    responses = _batched_generate(model, tokenizer, messages, max_new_tokens=16, batch_size=batch_size)

    correct = {cid: 0 for cid in cluster_descriptions}
    total = {cid: 0 for cid in cluster_descriptions}
    for true_cid, resp in zip(cluster_ids, responses):
        total[true_cid] = total.get(true_cid, 0) + 1
        match = re.search(r"-?\d+", resp)
        predicted = int(match.group()) if match else None
        if predicted == true_cid:
            correct[true_cid] = correct.get(true_cid, 0) + 1
    return {cid: (correct[cid] / total[cid] if total.get(cid, 0) > 0 else float("nan")) for cid in cluster_descriptions}


def _build_knn_graph(embeddings: np.ndarray, k_neighbors: int = 10) -> nx.Graph:
    """Graphe k-NN non dirigé, symétrisé par union (arête si i est parmi les
    k plus proches de j OU l'inverse) -- convention standard pour un graphe de
    conductance sur un nuage de points, pas une pondération par similarité
    (conductance = coupe/volume EN NOMBRE D'ARÊTES, cf. `conductance`)."""
    n = embeddings.shape[0]
    k = min(k_neighbors, n - 1)
    if k < 1:
        return nx.empty_graph(n)
    adj = kneighbors_graph(embeddings, n_neighbors=k, mode="connectivity", include_self=False)
    adj = adj.maximum(adj.T)  # symétrisation par union
    return nx.from_scipy_sparse_array(adj)


def conductance(graph: nx.Graph, cluster_mask: np.ndarray) -> float:
    """Conductance standard : cut(S, V\\S) / min(vol(S), vol(V\\S)). `graph`
    doit couvrir tous les indices 0..n-1 (construit par `_build_knn_graph`).
    Cas dégénérés (S vide/plein, ou S/complément de volume nul) -> 0.0
    (cluster trivialement "parfait" ou non calculable, pas une erreur)."""
    nodes = np.asarray(cluster_mask).nonzero()[0].tolist()
    if not nodes or len(nodes) == graph.number_of_nodes():
        return 0.0
    S, complement = set(nodes), set(graph.nodes()) - set(nodes)
    vol_s = sum(dict(graph.degree(S)).values())
    vol_comp = sum(dict(graph.degree(complement)).values())
    if vol_s == 0 or vol_comp == 0:
        return 0.0
    cut = sum(1 for u, v in graph.edges(S) if v not in S)
    return float(cut) / float(min(vol_s, vol_comp))


def conductance_zscore(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    k_neighbors: int = 10,
    n_random: int = 100,
    seed: int = 0,
) -> dict:
    """§4.3 (page 7) : "the z-score of each cluster's conductance in dense
    embedding space relative to a random sample (lower = tighter)". Pour
    chaque cluster réel, compare sa conductance à celle de `n_random`
    sous-ensembles ALÉATOIRES de MÊME TAILLE (même graphe k-NN) --
    z = (conductance_observée - moyenne_nulle) / écart_type_nul.

    Retourne {cluster_id: z_score}. Un cluster de taille dégénérée (0, 1 ou
    tout le corpus) reçoit `nan` (conductance non significative à cette
    taille)."""
    rng = np.random.default_rng(seed)
    graph = _build_knn_graph(embeddings, k_neighbors=k_neighbors)
    n = embeddings.shape[0]
    all_indices = np.arange(n)

    results = {}
    for cid in sorted(set(cluster_labels.tolist()) - {-1}):  # -1 = bruit HDBSCAN, exclu
        mask = cluster_labels == cid
        size = int(mask.sum())
        if size == 0 or size >= n:
            results[cid] = float("nan")
            continue
        obs = conductance(graph, mask)

        null_vals = np.empty(n_random)
        for r in range(n_random):
            rand_idx = rng.choice(all_indices, size=size, replace=False)
            rand_mask = np.zeros(n, dtype=bool)
            rand_mask[rand_idx] = True
            null_vals[r] = conductance(graph, rand_mask)

        std = null_vals.std()
        results[cid] = float((obs - null_vals.mean()) / std) if std > 0 else float("nan")
    return results
