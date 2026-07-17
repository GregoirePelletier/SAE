"""
src/sae/retrieval/latent_terms.py — Latent Terms (Clavié et al. 2026, arXiv:2605.29384)
appliqué aux mails de retour client EDF.

Principe (§3 du papier) : SAE gelé entraîné par pure reconstruction sur les
représentations du retriever → w̃(x) = Σ_t z_t (SUM-pooling, pas max : §3.2),
ϕ(u)=√u, puis BM25 sur le vocabulaire latent V_SAE avec f(j,D) = w_j(D).

Pas de dépôt officiel publié par Mixedbread à ce jour ; seule réimplémentation
communautaire : github.com/x-tabdeveloping/latent_terms (JAX, from scratch).
Ce script réutilise donc au maximum NOTRE dépôt (Pipeline 2) :
  - src.data.dataset.load_mails_tsv          (ingestion Mails.tsv)
  - src.data.preparation.split_into_phrases  (unités d'indexation)
  - src.sae.phrase_sae.extract_f2llm_embeddings / load_or_train_sae
    (F2LLM-v2-80M + PhraseLevelSAE BatchTopK+AuxK, déjà conforme au setup SAE
    du papier : reconstruction pure, top-k, aucun signal retrieval)
et n'ajoute que le scoring BM25 sparse (scipy CSR, ~40 lignes).

Usage (cluster, offline) :
  uv run python src/sae/retrieval/latent_terms.py \
      --mails /home/h21486/SAE/local_data/emails/Mails.tsv \
      --query "contestation facture Linky trop-perçu"
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from scipy import sparse

try:
    from src.data.dataset import load_mails_tsv
    from src.data.preparation import split_into_phrases
    from src.sae.phrase_sae import extract_f2llm_embeddings, load_or_train_sae
    from src.config import D_SAE, K_SPARSE, SAVE_DIR, CACHE_DIR
except ImportError:  # exécution à plat
    from dataset import load_mails_tsv
    from preparation import split_into_phrases
    from phrase_sae import extract_f2llm_embeddings, load_or_train_sae
    from config import D_SAE, K_SPARSE, SAVE_DIR, CACHE_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── w(x) : encode phrase-level → sum-pool doc → ϕ=√ (Eq. 8–9 du papier) ───

@torch.no_grad()
def latent_doc_weights(sae, phrase_emb: torch.Tensor, phrase_to_doc: np.ndarray,
                       n_docs: int, batch: int = 1024) -> sparse.csr_matrix:
    sae.eval()
    rows, cols, vals = [], [], []
    for s in range(0, phrase_emb.shape[0], batch):
        z = sae.encode(phrase_emb[s:s + batch].to(DEVICE)).cpu()      # [b, d_sae]
        b_rows, b_cols = (z > 1e-6).nonzero(as_tuple=True)
        rows.append(torch.from_numpy(phrase_to_doc[s + b_rows.numpy()]))
        cols.append(b_cols)
        vals.append(z[b_rows, b_cols])
    M = sparse.coo_matrix(
        (torch.cat(vals).float().numpy(),
         (torch.cat(rows).numpy(), torch.cat(cols).numpy())),
        shape=(n_docs, sae.d_sae),
    ).tocsr()                       # duplicates sommés par COO→CSR = SUM-pooling
    M.data = np.sqrt(M.data)        # ϕ(u) = √u
    return M


# ─── BM25 sur V_SAE (Eq. 3–4 + §3.3 : w_j(q) en poids explicite) ───

class LatentTermsIndex:
    def __init__(self, W_docs: sparse.csr_matrix, k1: float = 8.0, b: float = 0.7):
        # Défauts non tunés du papier (App. D) : k1=8, b=0.7, ϕ=√ des deux côtés.
        self.W, self.k1, self.b = W_docs, k1, b
        N = W_docs.shape[0]
        df = np.asarray((W_docs > 0).sum(axis=0)).ravel()             # n(t)
        self.idf = np.log((N - df + 0.5) / (df + 0.5)).clip(min=0.0)  # Eq. 4
        dl = np.asarray(W_docs.sum(axis=1)).ravel()                   # |D| = ||w(d)||₁
        self.K = 1.0 - b + b * dl / (dl.mean() + 1e-9)                # [N]

    def search(self, w_q: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        q_idx = np.nonzero(w_q > 0)[0]
        scores = np.zeros(self.W.shape[0], dtype=np.float64)
        for j in q_idx:
            col = self.W.getcol(j)                                    # f(j, D) = w_j(D)
            d_idx, f = col.indices, col.data.astype(np.float64)
            contrib = self.idf[j] * f * (self.k1 + 1) / (f + self.k1 * self.K[d_idx])
            scores[d_idx] += float(w_q[j]) * contrib                  # poids requête explicite
        top = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mails", default=os.environ.get("LOCAL_MAILS_PATH", "local_data/emails/Mails.tsv"))
    ap.add_argument("--query", default="réclamation facturation compteur Linky")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    df = load_mails_tsv(args.mails)
    texts = df["text"].tolist()
    phrases, p2d = split_into_phrases(texts, max_phrases_per_doc=20)
    p2d = np.array(p2d)
    print(f"{len(texts)} mails → {len(phrases)} phrases")

    emb, d_in = extract_f2llm_embeddings(
        phrases, max_length=128,
        cache_path=os.path.join(CACHE_DIR, f"lt_mails_phrase_emb_n{len(phrases)}"),
    )
    sae, _ = load_or_train_sae(
        d_in=d_in, d_sae=D_SAE, k=K_SPARSE, embeddings=emb,
        save_path=os.path.join(SAVE_DIR, f"lt_sae_d{D_SAE}_k{K_SPARSE}.pt"),
    )
    sae = sae.to(DEVICE)

    W_docs = latent_doc_weights(sae, emb, p2d, n_docs=len(texts))
    index = LatentTermsIndex(W_docs)

    # Requête : même chemin d'encodage (une "phrase" unique, sum-pool trivial).
    q_emb, _ = extract_f2llm_embeddings([args.query], max_length=128, cache_path=None)
    w_q = np.asarray(
        latent_doc_weights(sae, q_emb, np.zeros(1, dtype=int), n_docs=1).todense()
    ).ravel()

    print(f"\nRequête : {args.query!r}")
    for rank, (i, s) in enumerate(index.search(w_q, top_k=args.top_k), 1):
        print(f"  #{rank}  BM25={s:8.3f}  | {texts[i][:110]}...")


if __name__ == "__main__":
    main()
