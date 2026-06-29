import gc
import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_pool(model_output, attention_mask):
    token_emb = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class PhraseLevelSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        W_dec = F.normalize(torch.randn(d_sae, d_in), dim=1)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    @torch.no_grad()
    def init_from_data(self, embeddings: torch.Tensor):
        n = min(10000, len(embeddings))
        sample = embeddings[:n].float()
        self.b_dec.data.copy_(sample.mean(dim=0).to(self.b_dec.dtype))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Élimination du F.relu erroné avant le BatchTopK, en accord avec la littérature et frozen_core
        from src.sae.batch import batch_topk_encode
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return batch_topk_encode(pre, self.k, self.training)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        f = self.encode(x)
        x_recon = self.decode(f)
        mse = F.mse_loss(x_recon, x)
        variance = torch.var(x) + 1e-8
        normalized_mse = mse / variance
        l0 = (f > 1e-6).float().sum(dim=-1).mean()
        # Track dead features
        dead_frac = (f.sum(dim=0) == 0).float().mean()
        return {
            "sae_out": x_recon,
            "loss": normalized_mse,
            "normalized_mse": normalized_mse,
            "l0": l0,
            "dead_frac": dead_frac,
            "feature_acts": f
        }


def extract_f2llm_embeddings(texts: list[str], max_length: int = 128, cache_path: str = None) -> tuple[torch.Tensor, int]:
    if cache_path and os.path.exists(cache_path + ".pt"):
        print(f"  [Phrase] Restauration cache d'embeddings : {cache_path}.pt")
        emb = torch.load(cache_path + ".pt", map_location="cpu")
        return emb, emb.shape[1]

    print(f"  [Phrase] Extraction embeddings avec F2LLM-v2-80M ({len(texts)} phrases)...")
    from saev5 import EMB_MODEL
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(EMB_MODEL, local_files_only=True).to(DEFAULT_DEVICE).eval()

    all_embs = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEFAULT_DEVICE)
            outputs = model(**inputs)
            pooled = _mean_pool(outputs, inputs["attention_mask"])
            
            # Utilisation de la dimension Matryoshka tronquée
            from saev5 import MATRYOSHKA_DIM
            pooled_m = F.normalize(pooled[:, :MATRYOSHKA_DIM], p=2, dim=-1)
            all_embs.append(pooled_m.cpu())

    embeddings = torch.cat(all_embs, dim=0)
    if cache_path:
        torch.save(embeddings, cache_path + ".pt")
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return embeddings, embeddings.shape[1]


def encode_documents_with_phrase_sae(
    n_docs: int,
    sae: PhraseLevelSAE,
    phrase_embeddings: torch.Tensor,
    phrase_to_doc: np.ndarray,
) -> torch.Tensor:
    sae.eval()
    device = DEFAULT_DEVICE
    sae = sae.to(device)
    all_phrase_acts = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, phrase_embeddings.shape[0], batch_size):
            b = phrase_embeddings[i: i + batch_size].to(device)
            f = sae.encode(b)
            all_phrase_acts.append(f.cpu())
    phrase_acts = torch.cat(all_phrase_acts, dim=0)
    
    # Vectorisation optimisée du Max-Pooling par document
    phrase_to_doc_t = torch.from_numpy(phrase_to_doc).long()
    doc_acts = torch.full((n_docs, sae.d_sae), float("-inf"), dtype=phrase_acts.dtype)
    doc_acts.scatter_reduce_(0, phrase_to_doc_t.unsqueeze(-1).expand(-1, sae.d_sae), phrase_acts, reduce="amax", include_self=False)
    doc_acts = torch.where(doc_acts == float("-inf"), torch.zeros_like(doc_acts), doc_acts)
    return doc_acts


def load_or_train_sae(d_in: int, d_sae: int, k: int, embeddings: torch.Tensor, save_path: str, epochs: int = 20, lr: float = 1e-3) -> tuple[PhraseLevelSAE, dict]:
    sae = PhraseLevelSAE(d_in, d_sae, k).to(DEFAULT_DEVICE)
    sae.init_from_data(embeddings)

    if os.path.exists(save_path):
        print(f"  [Phrase] Restauration du Phrase-Level SAE : {save_path}")
        ckpt = torch.load(save_path, map_location=DEFAULT_DEVICE)
        sae.load_state_dict(ckpt["state_dict"])
        return sae, ckpt.get("history", {})

    print(f"  [Phrase] Entraînement du Phrase-Level SAE sur {embeddings.shape[0]} phrases...")
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    batch_size = 256
    history = {"epoch": [], "loss": [], "l0": [], "dead_frac": [], "step": []}
    step = 0

    for epoch in range(epochs):
        sae.train()
        permutation = torch.randperm(embeddings.shape[0])
        for i in range(0, embeddings.shape[0], batch_size):
            indices = permutation[i: i + batch_size]
            b_emb = embeddings[indices].to(DEFAULT_DEVICE)
            
            out = sae(b_emb)
            loss = out["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            history["loss"].append(loss.item())
            history["l0"].append(out["l0"].item())
            history["dead_frac"].append(out["dead_frac"].item())
            history["epoch"].append(epoch)
            history["step"].append(step)
            step += 1
        print(
            f"  Epoch {epoch+1:02d}/{epochs} | NMSE={out['normalized_mse'].item():.4f} | "
            f"L0={out['l0'].item():.1f} | dead={out['dead_frac'].item():.3f}"
        )

    ckpt = {
        "state_dict": {k: v.cpu() for k, v in sae.state_dict().items()},
        "config": {"d_in": d_in, "d_sae": d_sae, "k": k, "epochs": epochs, "lr": lr},
        "history": history,
    }
    torch.save(ckpt, save_path)
    with open(save_path.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return sae, history


def compute_sae_metrics(sae: PhraseLevelSAE, embeddings: torch.Tensor, batch_size: int = 1024) -> dict:
    sae.eval()
    nmse_acc, l0_acc, n_tok = 0.0, 0.0, 0
    active_counts = torch.zeros(sae.d_sae)
    device = DEFAULT_DEVICE
    with torch.no_grad():
        for i in range(0, embeddings.shape[0], batch_size):
            b = embeddings[i: i + batch_size].to(device).to(torch.bfloat16)
            out = sae(b)
            n_b = b.shape[0]
            nmse_acc += out["normalized_mse"].item() * n_b
            l0_acc += out["l0"].item() * n_b
            n_tok += n_b
            
            # Feature frequency tracks
            active_counts += (out["feature_acts"] > 1e-6).float().sum(dim=0).cpu()
            
    dead_pct = (active_counts == 0).float().mean().item() * 100
    return {
        "NMSE": nmse_acc / n_tok,
        "L0": l0_acc / n_tok,
        "dead_pct": dead_pct
    }