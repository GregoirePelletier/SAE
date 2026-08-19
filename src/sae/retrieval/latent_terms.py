"""
src/sae/retrieval/latent_terms.py — Latent Terms (Clavié et al. 2026,
arXiv:2605.29384) sur F2LLM-v2, réimplémentation fidèle token-level.

Remplace une première version qui opérait au niveau phrase (réutilisation de
PhraseLevelSAE/extract_f2llm_embeddings : pooling last-token par phrase avant
le SAE) — un montage pratique, pas la méthode du papier (§3.1-3.2 : le SAE
encode CHAQUE activation token du retriever, les codes sont sum-poolés sur
les tokens d'un document/requête, Eq. 7-9). Résultats de l'ancienne version :
RESULTS_TESTS.md §26/§68/§69, supersédés par §<N-À-COMPLÉTER> — méthode
différente, pas un simple rerun.

Aucun dépôt officiel Mixedbread à ce jour (vérifié : blog + org GitHub sans
repo dédié) ; la seule réimplémentation tierce connue
(x-tabdeveloping/latent_terms, JAX, non maintenue, aucun usage vérifiable)
n'est pas vendorisée — ce module réimplémente directement depuis le papier
(pdf/LatentTerms.pdf), pas depuis un tiers.

Écarts assumés à l'échelle du papier (Table 4), le budget de calcul de ce
projet ne permettant pas 30B tokens / dictionnaire 32768 / 5 graines :
  - D_SAE/K_SPARSE (8192/16, src/config.py) : défauts déjà établis dans ce
    dépôt pour Pipeline 2, même raisonnement d'ordre de grandeur que le
    papier ("same order of magnitude as common monolingual tokenizers") ;
    celui-ci rapporte par ailleurs que le retrieval en aval est robuste à
    ces deux choix.
  - Corpus SAE = TRAIN_TOKENS (défaut 33M) tokens uniques, FineWeb2-fr
    générique HORS domaine (§3.1 : "the SAE never sees data which is
    directly in-domain for retrieval tasks" — pas Mails.tsv), x TRAIN_EPOCHS
    (3, Table 4) = ~100M vues. Volume choisi au-delà du régime que ce dépôt a
    lui-même mesuré comme "affamé" pour un dictionnaire comparable
    (RESULTS_TESTS.md §18.3/§23 : effet net seulement au-delà de ~10-25M
    tokens pour D_EXTRA=1024, 8x plus petit que D_SAE ici) — sans viser
    l'échelle littérale du papier. Une seule graine (le papier en utilise 5
    pour réduire la variance) — réplication non engagée ici.
  - Activations SAE = état caché final BRUT de F2LLM (pas de troncature
    Matryoshka/renormalisation L2, contrairement à extract_f2llm_embeddings
    de phrase_sae.py) : le papier lit "the final hidden states of the
    backbone model" sans transformation, la troncature Matryoshka est une
    convention Pipeline 2 spécifique au pooling de phrase, sans rapport ici.
  - Requête encodée par la MÊME fonction que les documents (symétrique,
    comme dans le papier — pas de prompt d'instruction "search_query:"
    spécifique à F2LLM, la méthode du papier n'en utilise pas non plus).

Usage (cluster, offline) :
  uv run python src/sae/retrieval/latent_terms.py \
      --mails /home/h21486/SAE/local_data/emails/Mails.tsv \
      --query "contestation facture Linky trop-perçu"
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from transformers import AutoModel, AutoTokenizer

try:
    from src.data.dataset import load_mails_tsv
    from src.data.preparation import sample_fineweb2_chunks
    from src.config import D_SAE, K_SPARSE, SAVE_DIR, CACHE_DIR, EMB_MODEL, LOCAL_DATASET_PATH
except ImportError:  # exécution à plat
    from dataset import load_mails_tsv
    from preparation import sample_fineweb2_chunks
    from config import D_SAE, K_SPARSE, SAVE_DIR, CACHE_DIR, EMB_MODEL, LOCAL_DATASET_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUX_ALPHA = 1.0 / 32.0  # Gao et al. 2024, coefficient recommandé pour l'AuxK
                        # (déjà validé dans ce dépôt, src/sae/phrase_sae.py)
TRAIN_TOKENS = int(os.environ.get("LT_TRAIN_TOKENS", 33_000_000))
TRAIN_EPOCHS = int(os.environ.get("LT_TRAIN_EPOCHS", 3))        # Table 4
TRAIN_MAX_LENGTH = int(os.environ.get("LT_TRAIN_MAX_LENGTH", 256))
SAE_BATCH = int(os.environ.get("LT_SAE_BATCH", 4096))           # Table 4
PEAK_LR = float(os.environ.get("LT_PEAK_LR", 1e-3))             # Table 4
WARMUP_FRAC = 0.05                                               # Table 4


class LatentTermsSAE(nn.Module):
    """Top-K SAE (Gao et al. 2024) tel que spécifié Table 4 du papier Latent
    Terms : k identique et EXACT en train/eval, per-échantillon (pas de
    budget partagé de batch ni de seuil JumpReLU global —
    src/sae/batch.py::BatchTopKEncoder, correct pour Pipeline 2, diffère
    volontairement ici). AuxK conservé (α=1/32) : le papier dit suivre
    l'architecture Top-K SAE de Gao et al. 2024, qui l'inclut ; Table 4 ne le
    liste pas explicitement — hypothèse assumée, pas silencieuse."""

    def __init__(self, d_in: int, d_sae: int, k: int, dead_steps_threshold: int = 200):
        super().__init__()
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        self.k_aux = min(2 * k, d_sae // 2)
        self.dead_steps_threshold = dead_steps_threshold

        W_dec = torch.empty(d_sae, d_in)
        nn.init.kaiming_uniform_(W_dec, a=math.sqrt(5))  # Table 4 : "Decoder init: Kaiming"
        self.W_dec = nn.Parameter(F.normalize(W_dec, dim=1))
        self.W_enc = nn.Parameter(self.W_dec.data.T.clone())  # Table 4 : "Encoder init: transposed decoder"
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("steps_since_active", torch.zeros(d_sae))

    @torch.no_grad()
    def init_from_data(self, token_pool: torch.Tensor):
        n = min(100_000, len(token_pool))
        self.b_dec.data.copy_(token_pool[:n].float().mean(dim=0))

    def _pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def _topk(self, pre: torch.Tensor) -> torch.Tensor:
        k = min(self.k, pre.shape[-1])
        vals, idx = pre.topk(k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals.clamp(min=0.0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._topk(self._pre_acts(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def _aux_loss(self, pre: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """AuxK : reconstruit e = x − x̂ avec les k_aux features mortes les plus pré-activées."""
        dead = self.steps_since_active > self.dead_steps_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return torch.zeros((), device=pre.device, dtype=pre.dtype)
        k_aux = min(self.k_aux, n_dead)
        pre_dead = pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
        vals, idx = pre_dead.topk(k_aux, dim=-1)
        f_aux = torch.zeros_like(pre).scatter_(-1, idx, vals.clamp(min=0.0))
        e_hat = f_aux @ self.W_dec
        return F.mse_loss(e_hat, residual) / (residual.pow(2).mean() + 1e-8)

    def forward(self, x: torch.Tensor) -> dict:
        pre = self._pre_acts(x)
        f = self._topk(pre)
        x_recon = self.decode(f)

        mse = F.mse_loss(x_recon, x)
        normalized_mse = mse / (torch.var(x) + 1e-8)

        aux = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.training:
            with torch.no_grad():
                active = (f > 1e-6).any(dim=0)
                self.steps_since_active[active] = 0
                self.steps_since_active[~active] += 1
            aux = self._aux_loss(pre, (x - x_recon).detach())

        l0 = (f > 1e-6).float().sum(dim=-1).mean()
        dead_frac = (self.steps_since_active > self.dead_steps_threshold).float().mean()
        return {
            "sae_out": x_recon,
            "loss": normalized_mse + AUX_ALPHA * aux,
            "normalized_mse": normalized_mse,
            "aux_loss": aux,
            "l0": l0,
            "dead_frac": dead_frac,
            "feature_acts": f,
        }

    @torch.no_grad()
    def normalize_decoder(self):
        """Projection norme-unité + projection du gradient parallèle (Towards Monosemanticity)."""
        if self.W_dec.grad is not None:
            parallel = (self.W_dec.grad * self.W_dec.data).sum(-1, keepdim=True) * self.W_dec.data
            self.W_dec.grad -= parallel
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


def load_f2llm():
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(EMB_MODEL, local_files_only=True).to(DEVICE).eval()
    return tokenizer, model


def _batch_token_activations(texts, tokenizer, model, max_length, batch_size=64):
    """Génère, par document, les activations token-level NON-PAD de la
    dernière couche cachée -- brutes, sans pooling ni troncature (§3.1
    Eq. 7 : chaque token est une entrée indépendante pour le SAE).

    `torch.no_grad()` explicite AUTOUR DU CORPS de la boucle, jamais en
    décorateur sur cette fonction génératrice : un décorateur ne protégerait
    que l'appel créant le générateur, pas les itérations faites ensuite par
    l'appelant -- l'autograd resterait actif sur chaque forward F2LLM
    (piège déjà rencontré sur ce dépôt, cf. CLAUDE.md)."""
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.bool()
            per_doc = [hidden[row][mask[row]].to(torch.bfloat16).cpu() for row in range(hidden.shape[0])]
        for t in per_doc:
            yield t


def build_token_training_pool(target_tokens: int, tokenizer, model, cache_path: str = None,
                               chunk_length: int = 1024, max_length: int = TRAIN_MAX_LENGTH,
                               ) -> torch.Tensor:
    """Corpus d'entraînement du SAE : texte générique FineWeb2-fr HORS
    domaine (§3.1), jamais Mails.tsv. Sur-demande de chunks (le ratio
    tokens/chunk réel dépend du tokenizer) puis arrêt dès que la cible de
    tokens est atteinte -- pas de traitement au-delà du nécessaire."""
    if cache_path and os.path.exists(cache_path + ".pt"):
        print(f"  [LatentTerms] Restauration du pool d'entraînement : {cache_path}.pt")
        return torch.load(cache_path + ".pt", map_location="cpu")

    n_target_chunks = max(target_tokens // 150, 10_000)  # sur-demande volontaire, cf. docstring
    chunks = sample_fineweb2_chunks(n_target_chunks, chunk_length=chunk_length,
                                     local_dataset_path=LOCAL_DATASET_PATH)
    if not chunks:
        raise RuntimeError(f"Corpus générique FineWeb2-fr introuvable ({LOCAL_DATASET_PATH}) "
                            "-- requis pour entraîner le SAE hors-domaine (§3.1 du papier).")

    pool, n_tok = [], 0
    for tok in _batch_token_activations(chunks, tokenizer, model, max_length):
        if tok.shape[0] == 0:
            continue
        pool.append(tok)
        n_tok += tok.shape[0]
        if n_tok >= target_tokens:
            break
    embeddings = torch.cat(pool, dim=0)[:target_tokens]
    if embeddings.shape[0] < 0.9 * target_tokens:
        print(f"  [LatentTerms] ATTENTION : pool d'entraînement sous la cible "
              f"({embeddings.shape[0]} / {target_tokens} tokens) -- corpus FineWeb2-fr local "
              f"insuffisant à cette taille de chunk.")
    print(f"  [LatentTerms] Pool d'entraînement : {embeddings.shape[0]} tokens "
          f"({len(chunks)} chunks génériques sollicités).")
    if cache_path:
        torch.save(embeddings, cache_path + ".pt")
    return embeddings


def load_or_train_latent_terms_sae(d_in: int, d_sae: int, k: int, token_pool: torch.Tensor,
                                    save_path: str) -> tuple[LatentTermsSAE, dict]:
    sae = LatentTermsSAE(d_in, d_sae, k).to(DEVICE)
    sae.init_from_data(token_pool)

    if os.path.exists(save_path):
        print(f"  [LatentTerms] Restauration du SAE token-level : {save_path}")
        ckpt = torch.load(save_path, map_location=DEVICE)
        sae.load_state_dict(ckpt["state_dict"])
        return sae, ckpt.get("history", {})

    n = token_pool.shape[0]
    steps_per_epoch = max(1, n // SAE_BATCH)
    total_steps = steps_per_epoch * TRAIN_EPOCHS
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    optimizer = torch.optim.AdamW(sae.parameters(), lr=PEAK_LR)  # Table 4

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  # Table 4 : cosine decay

    print(f"  [LatentTerms] Entraînement SAE token-level : {n} tokens, {TRAIN_EPOCHS} époques, "
          f"batch={SAE_BATCH} ({total_steps} pas).")
    history = {"epoch": [], "loss": [], "l0": [], "dead_frac": [], "aux_loss": [], "step": [], "lr": []}
    step = 0
    out = None
    for epoch in range(TRAIN_EPOCHS):
        sae.train()
        perm = torch.randperm(n)
        for i in range(0, n - SAE_BATCH + 1, SAE_BATCH):
            batch = token_pool[perm[i:i + SAE_BATCH]].to(DEVICE).float()
            out = sae(batch)
            optimizer.zero_grad()
            out["loss"].backward()
            sae.normalize_decoder()          # projection gradient AVANT step
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)

            history["loss"].append(out["loss"].item())
            history["l0"].append(out["l0"].item())
            history["dead_frac"].append(out["dead_frac"].item())
            history["aux_loss"].append(float(out["aux_loss"]))
            history["epoch"].append(epoch)
            history["step"].append(step)
            history["lr"].append(scheduler.get_last_lr()[0])
            step += 1
        print(f"  Epoch {epoch+1:02d}/{TRAIN_EPOCHS} | NMSE={out['normalized_mse'].item():.4f} | "
              f"L0={out['l0'].item():.1f} | dead={out['dead_frac'].item():.3f} | "
              f"aux={float(out['aux_loss']):.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    ckpt = {
        "state_dict": {k_: v.cpu() for k_, v in sae.state_dict().items()},
        "config": {"d_in": d_in, "d_sae": d_sae, "k": k, "epochs": TRAIN_EPOCHS,
                   "train_tokens": n, "batch": SAE_BATCH, "peak_lr": PEAK_LR},
        "history": history,
    }
    torch.save(ckpt, save_path)
    with open(save_path.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return sae, history


@torch.no_grad()
def latent_doc_weights(sae: LatentTermsSAE, texts: list[str], tokenizer, model,
                       max_length: int = 512, batch_size: int = 32) -> sparse.csr_matrix:
    """w(x) : encode chaque token -> SAE -> SUM-pooling sur les tokens du
    document (§3.2 Eq. 9), puis ϕ(u) = √u."""
    sae.eval()
    rows, cols, vals = [], [], []
    for doc_idx, tok in enumerate(_batch_token_activations(texts, tokenizer, model, max_length, batch_size)):
        if tok.shape[0] == 0:
            continue
        z = sae.encode(tok.to(DEVICE).float()).cpu()      # [n_tok_doc, d_sae]
        b_rows, b_cols = (z > 1e-6).nonzero(as_tuple=True)
        rows.append(torch.full_like(b_rows, doc_idx))
        cols.append(b_cols)
        vals.append(z[b_rows, b_cols])
    M = sparse.coo_matrix(
        (torch.cat(vals).numpy(), (torch.cat(rows).numpy(), torch.cat(cols).numpy())),
        shape=(len(texts), sae.d_sae),
    ).tocsr()                       # duplicats sommés par COO->CSR = SUM-pooling sur les tokens
    M.data = np.sqrt(M.data)        # ϕ(u) = √u
    return M


class LatentTermsIndex:
    """BM25 sur V_SAE (Eq. 3-4 + §3.3 : w_j(q) en poids explicite). Inchangé
    par rapport à la première version -- déjà fidèle au papier."""

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
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    tokenizer, model = load_f2llm()
    d_in = model.config.hidden_size

    df = load_mails_tsv(args.mails)
    texts = df["text"].tolist()
    print(f"{len(texts)} mails.")

    model_tag = os.path.basename(EMB_MODEL.rstrip("/"))
    token_pool = build_token_training_pool(
        TRAIN_TOKENS, tokenizer, model,
        cache_path=os.path.join(CACHE_DIR, f"lt_generic_token_pool_n{TRAIN_TOKENS}_{model_tag}"))
    sae, _ = load_or_train_latent_terms_sae(
        d_in=d_in, d_sae=D_SAE, k=K_SPARSE, token_pool=token_pool,
        save_path=os.path.join(SAVE_DIR, f"lt_sae_token_d{D_SAE}_k{K_SPARSE}_tok{TRAIN_TOKENS}_{model_tag}.pt"))

    W_docs = latent_doc_weights(sae, texts, tokenizer, model, max_length=args.max_length)
    index = LatentTermsIndex(W_docs)

    W_q = latent_doc_weights(sae, [args.query], tokenizer, model, max_length=args.max_length)
    w_q = np.asarray(W_q.todense()).ravel()

    print(f"\nRequête : {args.query!r}")
    for rank, (i, s) in enumerate(index.search(w_q, top_k=args.top_k), 1):
        print(f"  #{rank}  BM25={s:8.3f}  | {texts[i][:110]}...")


if __name__ == "__main__":
    main()
