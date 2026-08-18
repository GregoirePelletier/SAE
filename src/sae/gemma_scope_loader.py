"""
src/sae/gemma_scope_loader.py
==============================
Chargement hors ligne d'un SAE Gemma Scope 2 (sae-lens 6.39.0) depuis disque.
"""

import json
import re
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
    Converter pour SAE.load_from_disk(). Lit config.json (ou cfg.json — nom historique
    des premières releases GemmaScope 2 avant renommage upstream) + params.safetensors
    sans aucune écriture disque. Mappe w_enc/w_dec -> W_enc/W_dec.
    """
    path = Path(path)

    cfg_path = path / "config.json"
    if not cfg_path.exists():
        cfg_path = path / "cfg.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
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

    hook_name = raw_cfg.get("hf_hook_point_in", "blocks.24.hook_resid_post")
    # hook_layer était figé à 24 (biais 12b/layer-24) : on le dérive du hook_name résolu,
    # avec repli sur raw_cfg["hook_layer"] si présent, sinon 24 (comportement historique).
    # Deux conventions observées : "blocks.N.hook_resid_post" (style TransformerLens,
    # anciennes releases) et "model.layers.N.output" (style HF, gemma-scope-2-270m-it).
    _layer_match = re.search(r"(?:blocks|model\.layers)\.(\d+)\.", hook_name)
    default_hook_layer = (
        int(_layer_match.group(1)) if _layer_match
        else raw_cfg.get("hook_layer", 24)
    )

    # A.5 (docs/AUDIT_2026-08.md) : les deux replis ci-dessus sont silencieux -- si
    # config.json/cfg.json est absent de "hf_hook_point_in" ET "hook_layer" (fichier
    # de métadonnées corrompu/incomplet, ou dossier mal peuplé), la couche résolue
    # tombe sur 24 sans jamais être comparée à celle réellement attendue. Le nom du
    # dossier (`.../layer_N_width_.../`) porte la couche VOULUE par l'appelant --
    # comparaison explicite ici, échec bruyant plutôt qu'un run entier sur la
    # mauvaise couche découvert (ou pas) après coup.
    _dir_layer_match = re.search(r"layer_(\d+)_", path.name)
    if _dir_layer_match:
        expected_layer = int(_dir_layer_match.group(1))
        if expected_layer != default_hook_layer:
            raise ValueError(
                f"Couche résolue depuis les métadonnées SAE ({default_hook_layer}) "
                f"!= couche attendue d'après le nom du dossier ({expected_layer}, "
                f"{path.name!r}) -- config.json/cfg.json probablement corrompu ou "
                f"dossier mal peuplé. Ne pas continuer silencieusement (A.5)."
            )
    print(f"  [gemma_scope_loader] Couche résolue : {default_hook_layer} "
          f"(hook_name={hook_name!r}, dossier={path.name!r})")

    cfg_dict = {
        "architecture": "jump_relu",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "bfloat16",
        "device": device,
        "model_name": raw_cfg.get("model_name", "google/gemma-3-12b-it"),
        "hook_name": hook_name,
        "hook_layer": default_hook_layer,
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
    }

    if cfg_overrides:
        cfg_dict.update(cfg_overrides)

    return cfg_dict, state_dict


def load_gemma_scope_sae(
    sae_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    release_id: Optional[str] = None,
    sae_id: Optional[str] = None,
) -> SAE:
    """
    Charge un SAE Gemma Scope local si sae_dir existe, sinon fallback Hub.

    sae_dir : chemin direct vers le sous-dossier contenant cfg.json
              (ex: .../snapshots/<rev>/resid_post/layer_24_width_16k_l0_medium)
    """
    sae_path = Path(sae_dir)

    if sae_path.is_dir():
        return SAE.load_from_disk(str(sae_path), device=device, converter=gemma_scope_converter)

    if release_id is None or sae_id is None:
        from src.config import RELEASE_ID as _default_release, HOOK_TYPE as _hook, SAE_ID as _sae_id
        release_id = release_id or _default_release
        sae_id = sae_id or f"{_hook}/{_sae_id}"

    sae, _cfg, _sparsity = SAE.from_pretrained(release=release_id, sae_id=sae_id, device=device)
    return sae