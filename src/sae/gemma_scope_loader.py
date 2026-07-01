"""
src/sae/gemma_scope_loader.py
==============================
Chargement hors ligne d'un SAE Gemma Scope 2 (sae-lens 6.39.0) depuis disque.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

from sae_lens import SAE
from sae_lens.registry import SAE_CLASS_REGISTRY

# Gemma Scope écrit "jump_relu" ; sae-lens enregistre la classe sous "jumprelu".
# Alias exécuté une seule fois à l'import du package.
if "jump_relu" not in SAE_CLASS_REGISTRY:
    SAE_CLASS_REGISTRY["jump_relu"] = SAE_CLASS_REGISTRY["jumprelu"]


def gemma_scope_converter(path, device: str = "cpu", cfg_overrides: Optional[dict] = None):
    """
    Converter pour SAE.load_from_disk(). Lit cfg.json + params.safetensors
    sans aucune écriture disque. Mappe w_enc/w_dec -> W_enc/W_dec.
    """
    path = Path(path)

    with open(path / "cfg.json", "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    raw_state = load_file(str(path / "params.safetensors"), device=device)
    d_in, d_sae = raw_state["w_enc"].shape

    state_dict = {
        "W_enc": raw_state["w_enc"],
        "W_dec": raw_state["w_dec"],
        "b_enc": raw_state["b_enc"],
        "b_dec": raw_state["b_dec"],
        "threshold": raw_state["threshold"],
    }

    cfg_dict = {
        "architecture": "jump_relu",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "bfloat16",
        "device": device,
        "model_name": raw_cfg.get("model_name", "google/gemma-3-12b-it"),
        "hook_name": raw_cfg.get("hf_hook_point_in", "blocks.24.hook_resid_post"),
        "hook_layer": 24,
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
    }

    if cfg_overrides:
        cfg_dict.update(cfg_overrides)

    return cfg_dict, state_dict


def load_gemma_scope_sae(
    sae_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    release_id: str = "gemma-scope-2-12b-it-res",
    sae_id: str = "resid_post/layer_24_width_16k_l0_medium",
) -> SAE:
    """
    Charge un SAE Gemma Scope local si sae_dir existe, sinon fallback Hub.

    sae_dir : chemin direct vers le sous-dossier contenant cfg.json
              (ex: .../snapshots/<rev>/resid_post/layer_24_width_16k_l0_medium)
    """
    sae_path = Path(sae_dir)

    if sae_path.is_dir():
        return SAE.load_from_disk(str(sae_path), device=device, converter=gemma_scope_converter)

    sae, _cfg, _sparsity = SAE.from_pretrained(release=release_id, sae_id=sae_id, device=device)
    return sae