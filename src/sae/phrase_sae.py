"""
phrase_sae.py — PhraseLevelSAE avec BatchTopKEncoder (seuil θ appris,
embarqué dans le state_dict) + AuxK (Gao et al. 2024) : loss auxiliaire de
reconstruction du résidu e = x − x̂ par les k_aux features mortes de plus
forte pré-activation.

    L = NMSE + α · NMSE_aux,  α = 1/32,  k_aux = 2k (borné à d_sae/2)

Une feature est "morte" si inactive depuis dead_steps_threshold steps
(buffer steps_since_active, persistant).
"""
import gc
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

try:
    from src.sae.batch import BatchTopKEncoder
except ImportError:
    from batch import BatchTopKEncoder

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUX_ALPHA = 1.0 / 32.0

# Le backbone F2LLM tourne en bf16 (embeddings L2-normalisés, bornés à 1.0 --
# aucun risque d'overflow, contrairement au residual stream Gemma-3 où bf16
# est requis pour d'autres raisons). PyTorch désactive TF32 par défaut pour
# les matmuls fp32 depuis 1.12 ; sans ces deux lignes, `AutoModel.from_pretrained`
# charge en fp32 plein (torch_dtype non précisé) et les rares matmuls fp32
# restants (ex. sur CPU) n'utilisent pas TF32 non plus (audit perf §2.7, item 5).
torch.set_float32_matmul_precision("high")


def _mean_pool(model_output, attention_mask):
    token_emb = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class PhraseLevelSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int, dead_steps_threshold: int = 200):
        super().__init__()
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        self.k_aux = min(2 * k, d_sae // 2)
        self.dead_steps_threshold = dead_steps_threshold

        W_dec = F.normalize(torch.randn(d_sae, d_in), dim=1)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.topk = BatchTopKEncoder(k)
        self.register_buffer("steps_since_active", torch.zeros(d_sae))

    @torch.no_grad()
    def init_from_data(self, embeddings: torch.Tensor):
        n = min(10000, len(embeddings))
        self.b_dec.data.copy_(embeddings[:n].float().mean(dim=0).to(self.b_dec.dtype))

    def _pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.topk(self._pre_acts(x))

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
        e_hat = f_aux @ self.W_dec                     # sans b_dec : cible = résidu centré
        return F.mse_loss(e_hat, residual) / (residual.pow(2).mean() + 1e-8)

    def forward(self, x: torch.Tensor) -> dict:
        pre = self._pre_acts(x)
        f = self.topk(pre)
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


def extract_f2llm_embeddings(texts: list[str], max_length: int = 128, cache_path: str = None) -> tuple[torch.Tensor, int]:
    if cache_path and os.path.exists(cache_path + ".pt"):
        print(f"  [Phrase] Restauration cache d'embeddings : {cache_path}.pt")
        emb = torch.load(cache_path + ".pt", map_location="cpu")
        return emb, emb.shape[1]

    try:
        from src.config import EMB_MODEL, MATRYOSHKA_DIM, EMB_POOLING
    except ImportError:
        from config import EMB_MODEL, MATRYOSHKA_DIM, EMB_POOLING
    # EMB_MODEL affiché (pas "F2LLM-v2-80M" figé) : le message était trompeur pour
    # tout run avec un backbone différent (ex. F2LLM-v2-330M, cf. RESULTS_TESTS.md).
    print(f"  [Phrase] Extraction embeddings avec {EMB_MODEL} (pooling={EMB_POOLING}, "
          f"{len(texts)} phrases)...")
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(
        EMB_MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(DEFAULT_DEVICE).eval()

    all_embs, batch_size = [], 128
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(DEFAULT_DEVICE)
            attention_mask = enc["attention_mask"].to(DEFAULT_DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # EMB_POOLING="last_token" (défaut, préserve tout run existant F2LLM) :
            # dernier token non-padding, adapté à un backbone décodeur causal comme
            # F2LLM. EMB_POOLING="cls" : premier token, requis pour un backbone
            # encodeur bidirectionnel entraîné avec cet objectif (bge-m3, cf.
            # RESULTS_TESTS.md §15.2 pour la similarité de labels).
            if EMB_POOLING == "cls":
                pooled = outputs.last_hidden_state[:, 0]
            else:
                last_idx = attention_mask.sum(dim=1) - 1
                pooled = outputs.last_hidden_state[
                    torch.arange(outputs.last_hidden_state.shape[0]), last_idx]
            pooled_m = F.normalize(pooled[:, :MATRYOSHKA_DIM], p=2, dim=-1)
            # PhraseLevelSAE est entraîné from-scratch en fp32 : caster ici plutôt que de
            # laisser passer le dtype natif du checkpoint F2LLM (bf16 avec les versions
            # récentes de transformers) qui casse le backward (paramètres fp32 vs grad bf16).
            all_embs.append(pooled_m.float().cpu())

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
    with torch.no_grad():
        for i in range(0, phrase_embeddings.shape[0], 1024):
            f = sae.encode(phrase_embeddings[i:i + 1024].to(device))
            all_phrase_acts.append(f.cpu())
    phrase_acts = torch.cat(all_phrase_acts, dim=0)

    try:
        from src.analysis.activations import scatter_maxpool
    except ImportError:
        from activations import scatter_maxpool
    return scatter_maxpool(phrase_acts, torch.from_numpy(phrase_to_doc), n_docs, sae.d_sae)


def load_or_train_sae(d_in: int, d_sae: int, k: int, embeddings: torch.Tensor,
                      save_path: str, epochs: int = 20, lr: float = 1e-3) -> tuple[PhraseLevelSAE, dict]:
    sae = PhraseLevelSAE(d_in, d_sae, k).to(DEFAULT_DEVICE)

    if os.path.exists(save_path):
        print(f"  [Phrase] Restauration du Phrase-Level SAE : {save_path}")
        ckpt = torch.load(save_path, map_location=DEFAULT_DEVICE)
        missing, _ = sae.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  [Phrase] Checkpoint sans certains buffers (θ/AuxK) : fallback "
                  f"TopK per-sample en eval. Clés manquantes : {missing}")
        return sae, ckpt.get("history", {})

    sae.init_from_data(embeddings)
    print(f"  [Phrase] Entraînement du Phrase-Level SAE sur {embeddings.shape[0]} phrases...")
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    # BATCH_TRAIN (config.py), pas une constante locale -- même valeur par défaut
    # (256), mais respecte désormais une éventuelle surcharge par variable
    # d'environnement plutôt que de l'ignorer silencieusement (audit 2026-08
    # round 3, §6.8).
    try:
        from src.config import BATCH_TRAIN as batch_size
    except ImportError:
        from config import BATCH_TRAIN as batch_size
    history = {"epoch": [], "loss": [], "l0": [], "dead_frac": [], "aux_loss": [], "step": []}
    step = 0

    for epoch in range(epochs):
        sae.train()
        permutation = torch.randperm(embeddings.shape[0])
        for i in range(0, embeddings.shape[0], batch_size):
            b_emb = embeddings[permutation[i:i + batch_size]].to(DEFAULT_DEVICE)
            out = sae(b_emb)
            optimizer.zero_grad()
            out["loss"].backward()
            sae.normalize_decoder()          # projection gradient AVANT step
            optimizer.step()
            with torch.no_grad():
                sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)

            history["loss"].append(out["loss"].item())
            history["l0"].append(out["l0"].item())
            history["dead_frac"].append(out["dead_frac"].item())
            history["aux_loss"].append(float(out["aux_loss"]))
            history["epoch"].append(epoch)
            history["step"].append(step)
            step += 1
        print(f"  Epoch {epoch+1:02d}/{epochs} | NMSE={out['normalized_mse'].item():.4f} | "
              f"L0={out['l0'].item():.1f} | dead={out['dead_frac'].item():.3f} | "
              f"aux={float(out['aux_loss']):.4f} | θ={float(sae.topk.threshold):.4f}")

    ckpt = {
        "state_dict": {k_: v.cpu() for k_, v in sae.state_dict().items()},
        "config": {"d_in": d_in, "d_sae": d_sae, "k": k, "epochs": epochs, "lr": lr,
                   "threshold": float(sae.topk.threshold)},
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
    with torch.no_grad():
        for i in range(0, embeddings.shape[0], batch_size):
            # PhraseLevelSAE est fp32 (embeddings F2LLM déjà L2-normalisés, bornés à
            # 1.0 -- aucun risque d'overflow, bf16 n'y apporte rien, cf. CLAUDE.md
            # règle bf16). Caster l'entrée en bf16 ici la dégraderait avant même
            # d'atteindre un SAE fp32 : les métriques publiées (NMSE/L0/dead_pct)
            # seraient calculées sur une entrée moins précise que celle utilisée à
            # l'entraînement (audit perf §2.7).
            b = embeddings[i:i + batch_size].to(DEFAULT_DEVICE).float()
            out = sae(b)
            n_b = b.shape[0]
            nmse_acc += out["normalized_mse"].item() * n_b
            l0_acc += out["l0"].item() * n_b
            n_tok += n_b
            active_counts += (out["feature_acts"] > 1e-6).float().sum(dim=0).cpu()
    return {
        "NMSE": nmse_acc / n_tok,
        "L0": l0_acc / n_tok,
        "dead_pct": (active_counts == 0).float().mean().item() * 100,
        "threshold": float(sae.topk.threshold),
    }