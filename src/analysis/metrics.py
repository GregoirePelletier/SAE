"""
metrics.py — Métriques d'évaluation scientifique des auto-encodeurs (FVE, NMSE, L0, Spearman rank).
Alignement strict sur les formules mathématiques de SAELens et interp_embed.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple


def dead_pct_core_extension(doc_acts: torch.Tensor, d_core: int) -> Tuple[float, float]:
    """`dead_pct` par plage (core GemmaScope figé vs extension entraînée),
    plutôt qu'un seul chiffre mélangeant les deux -- un SAE core généraliste
    a une proportion normale de features jamais activées sur un corpus
    spécifique et restreint (emails EDF) que son propre corpus
    d'entraînement, donc le chiffre blended sur-estime structurellement le
    taux de mort côté extension, la seule dont le taux de mort est un signal
    de qualité d'ENTRAÎNEMENT pertinent (déjà noté une fois à la main,
    RESULTS_TESTS.md §17.4, jamais calculé systématiquement par le pipeline
    depuis). `doc_acts` : `[n_docs, d_core + d_extra]`, activations
    documentaires max-poolées. Retourne (dead_pct_core, dead_pct_extension),
    chacun dans [0, 100]. `d_core >= doc_acts.shape[1]` (pas d'extension,
    ex. Latent Terms token-level) -> `(dead_pct_global, nan)`, la distinction
    n'a pas de sens sans coeur figé."""
    if d_core >= doc_acts.shape[1]:
        dead_global = (doc_acts.sum(dim=0) == 0).float().mean().item() * 100
        return dead_global, float("nan")
    dead_core = (doc_acts[:, :d_core].sum(dim=0) == 0).float().mean().item() * 100
    dead_extension = (doc_acts[:, d_core:].sum(dim=0) == 0).float().mean().item() * 100
    return dead_core, dead_extension


def compute_metrics(
    model: torch.nn.Module,
    acts: torch.Tensor,
    is_saelens: bool = False,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    FVE = 1 - NMSE, où NMSE = MSE(x, x̂) / Var(x).
    Note : variance normalisée sur l'ensemble du batch (mean sur tokens ET dimensions),
    ce qui est cohérent avec la définition scalaire de FVE utilisée dans SAEBench.
    """
    model.eval()
    acts_bf16 = acts.to(device).to(torch.bfloat16)

    with torch.no_grad():
        if is_saelens:
            recon = model.decode(model.encode(acts_bf16))
        else:
            out = model(acts_bf16)
            recon = out["sae_out"]

    acts_f = acts.to(device).float()   # référence non quantifiée
    recon_f = recon.float()

    mse = torch.mean((acts_f - recon_f).pow(2))
    variance = torch.mean((acts_f - acts_f.mean(dim=0, keepdim=True)).pow(2)) + 1e-8

    nmse = mse / variance
    fve = 1.0 - nmse

    # FVE/NMSE ci-dessus portent sur `sae_out` =
    # core+extra (scope complet), mais "L0" seul (ci-dessous, INCHANGÉ pour rester
    # rétrocompatible avec les chiffres déjà publiés) ne comptait que l'extra --
    # numérateur et dénominateur de portée différente si lus comme un même point
    # Pareto FVE/L0. L0_core et L0_total ajoutés pour un appariement correct.
    l0_core = float("nan")
    if is_saelens:
        with torch.no_grad():
            codes = model.encode(acts_bf16)
        l0 = (codes.abs() > 1e-6).float().sum(dim=-1).mean().item()
        l0_total = l0
    else:
        l0 = out.get("l0_extra", torch.tensor(0.0)).item()
        if "core_acts" in out:
            l0_core = (out["core_acts"].abs() > 1e-6).float().sum(dim=-1).mean().item()
            l0_total = l0_core + l0
        else:
            l0_total = l0

    return {
        "FVE": float(fve.item()),
        "NMSE": float(nmse.item()),
        "L0": float(l0),
        "L0_core": float(l0_core),
        "L0_total": float(l0_total),
    }


def compute_rho_sae(
    model: torch.nn.Module,
    acts: torch.Tensor,
    n_sample: int = 500,
    is_saelens: bool = False,
    device: str = "cuda",
) -> float:
    """
    ρ_SAE = Spearman(cos_sim_originaux, cos_sim_reconstruits) sur n_sample² / 2 paires.
    Mesure la conservation de la topologie locale dans l'espace de représentation.
    Complexité : O(n²) en mémoire — n_sample = 500 → 125K paires, tractable.
    """
    from scipy.stats import spearmanr
    import torch.nn.functional as F

    n = min(n_sample, acts.shape[0])
    idx = torch.randperm(acts.shape[0])[:n]
    sub = acts[idx].to(device).to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        if is_saelens:
            recon = model.decode(model.encode(sub)).float()
        else:
            out = model(sub)
            recon = out["sae_out"].float()

    sub = sub.float()
    orig_norm = F.normalize(sub, dim=1)
    recon_norm = F.normalize(recon, dim=1)

    cos_orig = (orig_norm @ orig_norm.T).cpu().numpy()
    cos_recon = (recon_norm @ recon_norm.T).cpu().numpy()

    triu = np.triu_indices(n, k=1)
    rho, _ = spearmanr(cos_orig[triu], cos_recon[triu])
    return float(rho)


def downstream_classification(
    acts_by_label: Dict[str, torch.Tensor],
    raw_emb_by_label: Dict[str, torch.Tensor] = None,
    groups_by_label: Dict[str, np.ndarray] = None,
) -> Dict[str, float]:
    """
    Sonde logistique à 5 plis pour évaluer la séparabilité linéaire des activations
    latentes vs embeddings bruts.

    `groups_by_label` (optionnel, `RESULTS_TESTS.md` §57) : parent_id (mail
    d'origine) par échantillon, même clés que `acts_by_label`. Si fourni, la CV
    utilise `StratifiedGroupKFold` (groupe = mail d'origine, un mail original et
    toutes ses variantes augmentées restent du même côté d'un pli) au lieu de
    `StratifiedKFold` -- sans ça, une variante augmentée d'un mail présent dans le
    pli de train peut fuiter dans le pli de test (quasi-duplicata sémantique),
    gonflant artificiellement l'accuracy mesurée. Rétrocompatible : `None` (défaut)
    garde le comportement `StratifiedKFold` existant pour tout appelant qui n'a pas
    d'info de groupe (ex. probe energy/sports/support, mails indépendants).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.metrics import accuracy_score
    from scipy import sparse as sp

    # X_sae passé en CSR sparse à LogisticRegression, jamais dense : les
    # activations SAE sont ~99,9% de zéros (BatchTopK), et sklearn recopie tout
    # tableau dense fp32 en fp64 en interne -- dense upcasté fp64 sur [n, d_sae]
    # (d_sae=17408+ en Pipeline 1) est ce qui rendait `downstream_classification`
    # compute-bound au point de tourner indéfiniment sans sortie (CLAUDE.md,
    # rule "cpus-per-task=32"). CSR réduit le travail de deux à trois ordres de
    # grandeur (`liblinear`/`lbfgs` supportent tous deux un X sparse nativement)
    # -- 32 coeurs n'est un correctif que si le profil montre un coût réellement
    # dense après ce changement, pas un point de départ.
    X_sae_list, X_raw_list, y_list, groups_list = [], [], [], []
    for label_id, (label_name, sae_acts) in enumerate(acts_by_label.items()):
        sae_np = sae_acts.float().detach().cpu().numpy()
        X_sae_list.append(sp.csr_matrix(sae_np))
        y_list.append(np.full(sae_np.shape[0], label_id))
        if raw_emb_by_label and label_name in raw_emb_by_label:
            X_raw_list.append(raw_emb_by_label[label_name].float().detach().cpu().numpy())
        if groups_by_label is not None:
            groups_list.append(np.asarray(groups_by_label[label_name]))

    X_sae = sp.vstack(X_sae_list, format="csr")
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0) if groups_by_label is not None else None

    # liblinear ne supporte que la classification binaire (>=3 classes lève une
    # ValueError depuis les versions récentes de sklearn, cf. probe multi-classe
    # sur les axes email introduit ultérieurement) -- lbfgs (multinomial nativement
    # supporté) au-delà de 2 classes, liblinear conservé pour le cas binaire déjà
    # validé (energy/sports) afin de ne rien changer à un résultat existant.
    solver = "liblinear" if len(acts_by_label) <= 2 else "lbfgs"

    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        split_args = (X_sae, y, groups)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_args = (X_sae, y)
    accs_sae = []

    for train_idx, test_idx in skf.split(*split_args):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver=solver, random_state=42)
        clf.fit(X_sae[train_idx], y[train_idx])
        preds = clf.predict(X_sae[test_idx])
        accs_sae.append(accuracy_score(y[test_idx], preds))

    results = {"acc_sae": float(np.mean(accs_sae))}

    if X_raw_list:
        X_raw = np.concatenate(X_raw_list, axis=0)
        raw_split_args = (X_raw, y, groups) if groups is not None else (X_raw, y)
        accs_raw = []
        for train_idx, test_idx in skf.split(*raw_split_args):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver=solver, random_state=42)
            clf.fit(X_raw[train_idx], y[train_idx])
            preds = clf.predict(X_raw[test_idx])
            accs_raw.append(accuracy_score(y[test_idx], preds))
        results["acc_raw"] = float(np.mean(accs_raw))
        results["delta_acc"] = results["acc_sae"] - results["acc_raw"]

    return results


def normalize_by_p90_and_score(matched_acts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Score documentaire pondéré à température pour le retrieval par propriété
    (`property_based_retrieval`, `saev5.py`), activations normalisées par le 90e
    percentile non nul de chaque latent (interp_embed, Fig. 10 étape 1) avant la
    somme pondérée par rang -- sans cette normalisation, les magnitudes JumpReLU
    non bornées du core (outliers ~1e5) écrasent le poids de rang, le score est
    dominé par l'échelle des latents plutôt que par leur pertinence
    (AUDIT_SAE_2026-08.md). `matched_acts` : [n_docs, k_latents]."""
    k = matched_acts.shape[1]
    p90 = torch.ones(k, dtype=matched_acts.dtype)
    for j in range(k):
        nonzero = matched_acts[:, j][matched_acts[:, j] > 0]
        if nonzero.numel() > 0:
            p90[j] = torch.quantile(nonzero, 0.9).clamp(min=1e-8)
    normalized_acts = matched_acts / p90
    return (normalized_acts * weights).sum(dim=-1)


def average_precision(relevance: list[bool]) -> float:
    """AP (interp_embed, App. G) : AP = (1/|R|) * Σ_k (précision@k) * 1{d_k pertinent},
    `relevance` déjà classé par rang décroissant de score. |R| = nombre total de
    documents pertinents dans `relevance` (suppose que tous les documents pertinents
    du corpus apparaissent dans la liste classée -- vrai si `relevance` couvre le
    corpus entier, sinon |R| sous-estimé)."""
    n_relevant = sum(relevance)
    if n_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for k, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            precision_sum += hits / k
    return precision_sum / n_relevant


def precision_at_k(relevance: list[bool], k: int) -> float:
    """P@K (interp_embed, App. G) : fraction de documents pertinents dans les k premiers rangs."""
    if k == 0:
        return 0.0
    return sum(relevance[:k]) / k


def mean_average_precision(relevance_lists: list[list[bool]]) -> float:
    """MAP = moyenne de `average_precision` sur les requêtes (App. G mentionne MAP comme
    métrique agrégée mais n'écrit pas cette équation explicitement -- moyenne standard,
    seule lecture cohérente avec AP/P@K telles que définies dans le texte)."""
    if not relevance_lists:
        return 0.0
    return float(np.mean([average_precision(r) for r in relevance_lists]))


def mean_precision_at_k(relevance_lists: list[list[bool]], k: int) -> float:
    """MP@K = moyenne de `precision_at_k` sur les requêtes (même remarque que
    `mean_average_precision` -- agrégation standard, non explicitée séparément dans le
    texte de l'App. G)."""
    if not relevance_lists:
        return 0.0
    return float(np.mean([precision_at_k(r, k) for r in relevance_lists]))


def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> list[tuple]:
    """RRF (Cormack, Clarke & Buettcher 2009 [83], cité mais pas reproduit dans le texte
    de l'App. G) : score(d) = Σ_rankings 1/(k + rang(d)). `k=60` = constante conventionnelle
    de l'article original -- l'App. G ne précise pas sa valeur pour OpenAI+LLM/Combined,
    hypothèse à documenter (R6) si les chiffres sont comparés à ceux du papier. Rang 1-indexé ;
    un document absent d'un ranking ne contribue aucun terme pour ce ranking. Retourne
    [(doc_id, score), ...] trié par score décroissant."""
    scores: dict = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rank_biased_overlap(ranking_a: list, ranking_b: list, p: float = 0.98, depth: Optional[int] = None) -> float:
    """RBO de base, non extrapolé (Webber, Moffat & Zobel 2010 [84], cité mais pas
    reproduit dans le texte de l'App. G, seul p=0.98 y est donné) :
    RBO = (1-p) * Σ_{d=1}^{depth} p^(d-1) * |A_∩d ∩ B_∩d| / d. `depth` par défaut =
    min(len(ranking_a), len(ranking_b)). L'extrapolation à profondeur infinie du papier
    original (RBO_EXT, pour compenser la troncature) n'est pas implémentée (R6) : cette
    formule de base ne vaut jamais exactement 1 pour deux classements identiques de
    longueur finie (converge vers 1 seulement quand depth -> infini -- à depth=50,
    p=0.98 comme dans l'App. G, deux classements identiques donnent 1-0.98^50≈0,64, pas
    1), mais reste strictement croissante avec l'accord réel entre les deux classements
    (identique > partiellement recouvrant > disjoint = 0), donc utilisable pour comparer
    des méthodes de retrieval entre elles."""
    depth = depth if depth is not None else min(len(ranking_a), len(ranking_b))
    if depth == 0:
        return 0.0
    set_a, set_b = set(), set()
    total = 0.0
    for d in range(1, depth + 1):
        set_a.add(ranking_a[d - 1])
        set_b.add(ranking_b[d - 1])
        overlap = len(set_a & set_b)
        total += (p ** (d - 1)) * (overlap / d)
    return (1 - p) * total