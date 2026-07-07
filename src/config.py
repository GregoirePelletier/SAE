"""
src/config.py — Source unique des constantes de la pipeline.
Casse l'import circulaire phrase_sae → saev5 et centralise l'env.
Tout est surchargé par variables d'environnement (compat sbatch existants).
"""
import os

SEED = int(os.environ.get("SEED", "42"))

# ─── Pipeline 2 (F2LLM) ───
EMB_MODEL      = os.environ.get("EMB_MODEL", "/home/h21486/SAE/models/F2LLM-v2-80M")
MATRYOSHKA_DIM = int(os.environ.get("MATRYOSHKA_DIM", "320"))
D_SAE          = int(os.environ.get("D_SAE", "8192"))
K_SPARSE       = int(os.environ.get("K_SPARSE", "16"))
EPOCHS         = int(os.environ.get("EPOCHS", "30"))
LR             = float(os.environ.get("LR", "5e-4"))
BATCH_TRAIN    = int(os.environ.get("BATCH_TRAIN", "256"))
MAX_PHRASES_DOC = int(os.environ.get("MAX_PHRASES_DOC", "20"))

# ─── Pipeline 1 (FrozenCore) ───
D_EXTRA      = int(os.environ.get("D_EXTRA", "1024"))
K_EXTRA      = int(os.environ.get("K_EXTRA", "32"))
EPOCHS_EXTRA = int(os.environ.get("EPOCHS_EXTRA", "10"))
LR_EXTRA     = float(os.environ.get("LR_EXTRA", "3e-4"))
USE_FROZEN_CORE = os.environ.get("USE_FROZEN_CORE", "1").strip() in ("1", "true", "True")
N_TOKENS_EXTRA_TRAIN = int(os.environ.get("N_TOKENS_EXTRA_TRAIN", "500000"))
N_FEATURES_TO_LABEL  = int(os.environ.get("N_FEATURES_TO_LABEL", "10"))

# ─── Modèle Gemma-3 / GemmaScope ───
MODEL_SIZE = os.environ.get("MODEL_SIZE", "12b")
_PRESETS = {
    "12b":  ("/home/h21486/SAE/models/gemma-3-12b-it", "gemma-scope-2-12b-it-res", "layer_24_width_16k_l0_medium", 24),
    "4b":   ("/home/h21486/SAE/models/gemma-3-4b-it",  "gemma-scope-2-4b-it-res",  "layer_17_width_16k_l0_medium", 17),
    "1b":   ("/home/h21486/SAE/models/gemma-3-1b-it",  "gemma-scope-2-1b-it-res",  "layer_13_width_16k_l0_medium", 13),
    "270m": ("/home/h21486/SAE/models/gemma-3-270m",   "gemma-scope-2-270m-pt-res", "layer_12_width_16k_l0_medium", 12),
}
_m, RELEASE_ID, _sae_default, LAYER = _PRESETS.get(MODEL_SIZE, _PRESETS["12b"])
MODEL_ID  = os.environ.get("MODEL_ID", _m)
SAE_ID    = os.environ.get("SAE_ID", _sae_default) if MODEL_SIZE == "12b" else _sae_default
HOOK_TYPE = os.environ.get("HOOK_TYPE", "resid_post")
LOCAL_SAE_ROOT = os.environ.get("LOCAL_SAE_DIR", f"/home/h21486/SAE/saes/{RELEASE_ID}")
SAE_SNAPSHOT   = os.environ.get("SAE_SNAPSHOT", "0" * 40)

# ─── Chemins ───
HF_TOKEN = os.environ.get("HF_TOKEN")
SAVE_DIR = os.environ.get("SAVE_DIR", "./results/")
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
LOCAL_DATASET_PATH = os.environ.get(
    "LOCAL_DATASET_PATH",
    "/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet")
LOCAL_MAILS_PATH = os.environ.get("LOCAL_MAILS_PATH", "/home/h21486/SAE/Mails.tsv")