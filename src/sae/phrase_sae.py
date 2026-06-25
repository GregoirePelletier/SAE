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
        data_mean = sample.mean(dim=0)
        self.b_dec.data.copy_(data_mean.to(self.b_dec.dtype))
        try:
            _, _, Vt = torch.linalg.svd(sample - data_mean, full_matrices=False)
            n_pca = min(self.d_sae, Vt.shape[0])
            pad = F.normalize(torch.randn(self.d_sae - n_pca, self.d_in), dim=1)
            W_init = F.normalize(torch.cat([Vt[:n_pca], pad], dim=0).float(), dim=1)
            self.W_dec.data.copy_(W_init.to(self.W_dec.dtype))
            self.W_enc.data.copy_(W_init.T.to(self.W_enc.dtype))
            print(f"  W_dec ← SVD ({n_pca} directions, {self.d_sae - n_pca} padding)")
        except Exception as e:
            print(f"  Échec SVD ({e}), init aléatoire.")
        print(f"  b_dec ← mean (norme={data_mean.norm():.4f})")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        from src.sae.batch import batch_topk_encode
        return batch_topk_encode(pre, self.k, self.training)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        acts = self.encode(x)
        x_hat = self.decode(acts)
        mse = F.mse_loss(x_hat, x)
        var_x = (x - x.mean(dim=0)).pow(2).mean()
        return {
            "sae_out": x_hat,
            "feature_acts": acts,
            "normalized_mse": mse / (var_x + 1e-8),
            "l0": (acts.abs() > 1e-6).float().sum(dim=-1).mean(),
            "dead_frac": ((acts.abs() > 1e-6).float().sum(dim=0) == 0).float().mean(),
        }

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


def extract_f2llm_embeddings(
    phrases: list,
    emb_model: str,
    matryoshka_dim: int,
    max_length: int = 128,
    batch_size: int = 64,
    cache_path: str = None,
):
    if cache_path and os.path.exists(cache_path + ".pt"):
        emb = torch.load(cache_path + ".pt", weights_only=True)
        print(f"  Cache embeddings : {emb.shape}")
        return emb, emb.shape[1]

    tokenizer = AutoTokenizer.from_pretrained(
        emb_model, token=os.environ.get("HF_TOKEN"), trust_remote_code=True, local_files_only=True
    )
    device = DEFAULT_DEVICE
    model = AutoModel.from_pretrained(
        emb_model, token=os.environ.get("HF_TOKEN"), trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    full_dim = model.config.hidden_size
    matryoshka_dim = min(matryoshka_dim, full_dim)

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(phrases), batch_size), desc="F2LLM embeddings"):
            inputs = tokenizer(
                phrases[i: i + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            out = model(**inputs)
            emb = _mean_pool(out, inputs["attention_mask"])
            emb = F.normalize(emb[:, :matryoshka_dim], p=2, dim=1)
            all_emb.append(emb.cpu())

    embeddings = torch.cat(all_emb)
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    if cache_path:
        torch.save(embeddings, cache_path + ".pt")
    return embeddings, matryoshka_dim


def encode_documents_with_phrase_sae(
    n_docs: int,
    sae: PhraseLevelSAE,
    phrase_embeddings: torch.Tensor,
    phrase_to_doc: np.ndarray,
) -> torch.Tensor:
    device = DEFAULT_DEVICE
    sae.eval()
    all_acts = []
    with torch.no_grad():
        for i in tqdm(range(0, len(phrase_embeddings), 512), desc="Encodage Document SAE"):
            b = phrase_embeddings[i: i + 512].to(device).to(torch.bfloat16)
            all_acts.append(sae.encode(b).cpu())
    all_acts = torch.cat(all_acts)

    doc_acts = torch.full((n_docs, sae.d_sae), -1e9, dtype=all_acts.dtype)
    idx_exp = torch.from_numpy(phrase_to_doc).long().unsqueeze(1).expand(-1, sae.d_sae)
    doc_acts.scatter_reduce_(0, idx_exp, all_acts, reduce="amax", include_self=True)
    doc_acts[doc_acts == -1e9] = 0.0
    print(f"  Matrice doc SAE (max-pool phrases) : {doc_acts.shape}")
    return doc_acts


def load_or_train_sae(
    d_in: int, d_sae: int, k: int,
    embeddings: torch.Tensor,
    save_path: str,
    epochs: int = 30,
    lr: float = 5e-4,
):
    device = DEFAULT_DEVICE
    if os.path.exists(save_path):
        print(f"  Chargement PhraseSAE : {save_path}")
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        sae = PhraseLevelSAE(d_in, d_sae, k).to(device).to(torch.bfloat16)
        sae.load_state_dict(ckpt["state_dict"])
        return sae, ckpt.get("history", {})

    sae = PhraseLevelSAE(d_in, d_sae, k).to(device).to(torch.bfloat16)
    sae.init_from_data(embeddings)

    N = embeddings.shape[0]
    batch_train = int(os.environ.get("BATCH_TRAIN", "256"))
    steps_per_ep = max(1, N // batch_train)
    total_steps = steps_per_ep * epochs
    warmup_steps = min(500, total_steps // 10)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / max(warmup_steps, 1)
        if step < warmup_steps
        else 0.5 * (1.0 + math.cos(math.pi * ((step - warmup_steps) / max(total_steps - warmup_steps, 1))))
    )
    history = {"nmse": [], "l0": [], "dead_frac": [], "step": []}
    step = 0
    sae.train()

    for epoch in range(epochs):
        perm = torch.randperm(N)
        for start in range(0, N - batch_train + 1, batch_train):
            b = embeddings[perm[start: start + batch_train]].to(device).to(torch.bfloat16)
            out = sae(b)
            optimizer.zero_grad(set_to_none=True)
            out["normalized_mse"].backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            sae.normalize_decoder()
            if step % 50 == 0:
                history["nmse"].append(out["normalized_mse"].item())
                history["l0"].append(out["l0"].item())
                history["dead_frac"].append(out["dead_frac"].item())
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
            nmse_acc += out["normalized_mse"].item() * b.shape[0]
            l0_acc += out["l0"].item() * b.shape[0]
            active_counts += (out["feature_acts"].abs() > 1e-6).float().sum(dim=0).cpu()
            n_tok += b.shape[0]
    return {
        "NMSE": nmse_acc / n_tok,
        "L0": l0_acc / n_tok,
        "dead_pct": (active_counts == 0).float().mean().item() * 100,
        "active_features": int((active_counts > 0).sum()),
    }
