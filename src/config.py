"""
src/config.py — Source unique des constantes de la pipeline.
Casse l'import circulaire phrase_sae → saev5 et centralise l'env.
Tout est surchargé par variables d'environnement (compat sbatch existants).
"""
import os
import re

SEED = int(os.environ.get("SEED", "42"))

# ─── Pipeline 2 (F2LLM) ───
# Repo HF réel confirmé : codefuse-ai/F2LLM-v2-80M (l'org "Alibaba-NLP" utilisée
# ailleurs dans le dépôt historique n'existe pas -> 404). Résolu depuis le cache HF
# après un premier téléchargement (huggingface_hub.snapshot_download), comme MODEL_ID.
EMB_MODEL      = os.environ.get("EMB_MODEL", "codefuse-ai/F2LLM-v2-80M")
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
# Défaut : 12b (expérimentation transférée sur une machine avec plus de VRAM — la
# variante 270m locale 6 Go a servi à valider le pipeline de bout en bout, cf.
# Context.md, mais sa capacité d'auto-interprétation (labels juge) est limitée).
MODEL_SIZE = os.environ.get("MODEL_SIZE", "12b")

# (model_path, release_id, sae_id_default, layer, d_model)
# d_model sert de repli pour mocked_get_safetensors_tensor_shapes (saev5.py) quand
# aucune config locale n'est trouvée en cache.
# MODEL_ID pointe directement le repo HF (pas un chemin disque) : après download_sae.py,
# il est résolu depuis le cache HF par from_pretrained(local_files_only=True) — portable
# entre machines, contrairement aux anciens chemins /home/h21486/SAE/... codés en dur.
# RELEASE_ID vérifié empiriquement (API HF authentifiée) : le repo réel est
# "google/gemma-scope-2-{taille}-it" SANS suffixe "-res" (l'ancien suffixe, hérité de
# GemmaScope v1, pointait vers un repo inexistant — jamais vérifié avant ce passage).
_PRESETS = {
    # Largeur 16k choisie plutôt que 262k (utilisée par run_sae.slurm) : la couverture
    # des labels Neuronpedia sur 24-gemmascope-2-res-262k est faible (~10k features
    # labellisées sur 262144, observé manuellement) ; 16k est nettement plus dense en
    # proportion (comparable à la couverture ~98% obtenue empiriquement sur 270m/65k)
    # et bien moins coûteux à traiter pour un premier run complet.
    "12b":  ("google/gemma-3-12b-it", "gemma-scope-2-12b-it", "layer_24_width_16k_l0_medium", 24, 4096),
    "4b":   ("google/gemma-3-4b-it",  "gemma-scope-2-4b-it",  "layer_17_width_16k_l0_medium", 17, 2560),
    "1b":   ("google/gemma-3-1b-it",  "gemma-scope-2-1b-it",  "layer_13_width_16k_l0_medium", 13, 1152),
    # google/gemma-3-270m-it (LM) + google/gemma-scope-2-270m-it (SAE, resid_post,
    # layer 12, largeur 65k confirmée via Neuronpedia : gemma-3-270m-it/12-gemmascope-2-res-65k).
    # d_model=640 confirmé empiriquement (w_enc.shape du SAE téléchargé), pas 1024 comme
    # supposé initialement d'après des specs publiques génériques du modèle. Utilisé pour
    # la validation locale (6 Go VRAM) — cf. Context.md pour les résultats et limites.
    "270m": ("google/gemma-3-270m-it", "gemma-scope-2-270m-it", "layer_12_width_65k_l0_medium", 12, 640),
}
_m, RELEASE_ID, _sae_default, LAYER, D_MODEL = _PRESETS.get(MODEL_SIZE, _PRESETS["270m"])
MODEL_ID  = os.environ.get("MODEL_ID", _m)
SAE_ID    = os.environ.get("SAE_ID", _sae_default)
HOOK_TYPE = os.environ.get("HOOK_TYPE", "resid_post")
LOCAL_SAE_ROOT = os.environ.get("LOCAL_SAE_DIR", f"./local_data/saes/{RELEASE_ID}")
SAE_SNAPSHOT   = os.environ.get("SAE_SNAPSHOT", "0" * 40)

# ─── Précision ───
# bf16 par défaut, y compris en local. Testé empiriquement : Gemma-3 a des activations
# "massives" documentées dans le residual stream (outliers ~1e5) qui dépassent le max
# représentable en fp16 (~65504) -> overflow silencieux vers inf/nan, qui contamine tout
# l'entraînement de ExtendedSAE (Loss=nan dès l'epoch 1, confirmé sur run local 270m).
# bf16 a le même exposant 8 bits que fp32 (plage jusqu'à ~3e38) donc pas d'overflow, au
# prix d'un calcul plus lent sur Turing (pas de tensor cores bf16 natifs, upcast logiciel)
# — acceptable ici vu la taille de 270M. fp16 reste possible via env si un futur modèle
# n'a pas ce problème d'activations massives, mais ce n'est plus le défaut.
DTYPE = os.environ.get("DTYPE", "bf16").strip().lower()

# ─── Mode réseau ───
# Sur le cluster (pas d'accès internet direct), saev5.py force HF_HUB_OFFLINE=1 et
# désactive la vérification SSL. En local ces patchs empêcheraient tout téléchargement
# initial du modèle/SAE : désactivés par défaut, activables via env pour reproduire
# l'environnement cluster.
CLUSTER_OFFLINE_MODE = os.environ.get("CLUSTER_OFFLINE_MODE", "0").strip() in ("1", "true", "True")

# ─── Chemins ───
HF_TOKEN = os.environ.get("HF_TOKEN")
SAVE_DIR = os.environ.get("SAVE_DIR", "./results/")
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
LOCAL_DATASET_PATH = os.environ.get(
    "LOCAL_DATASET_PATH",
    "./local_data/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet")
# Corpus EDF réel + variantes augmentées (générées par scripts/run_augmentation.py,
# cf. RESULTS_TESTS.md §0) : emplacement canonique unique, utilisé par saev5.py,
# scripts/run_augmentation.py et scripts/baseline_gemmascope.py. Anciennement
# Mails.tsv à la racine et augmented_mails.jsonl sous un results_*/ de run
# particulier (accidentel : un jeu de données d'entrée ne doit pas vivre sous un
# dossier de résultats d'expérience).
LOCAL_MAILS_PATH = os.environ.get("LOCAL_MAILS_PATH", "./local_data/emails/Mails.tsv")
LOCAL_AUGMENTED_MAILS_PATH = os.environ.get(
    "LOCAL_AUGMENTED_MAILS_PATH", "./local_data/emails/augmented_mails.jsonl")

# Labels Neuronpedia (cf. src/sae/neuronpedia_labels.py) : cache partagé entre TOUS
# les runs (indépendant de SAVE_DIR/CACHE_DIR), régénérable hors-cluster via
# fetch_neuronpedia_labels() mais réutilisé tel quel une fois présent -- jamais
# re-téléchargé, jamais dupliqué par run. Override par env si besoin d'un autre jeu.
# Largeur dérivée de SAE_ID (ex. "layer_24_width_16k_l0_medium" -> "16k") au lieu
# d'être figée en dur : varie selon MODEL_SIZE (65k pour 270m, 16k pour 12b/4b/1b).
_WIDTH_MATCH = re.search(r"width_(\w+?)_l0", SAE_ID)
_SAE_WIDTH = _WIDTH_MATCH.group(1) if _WIDTH_MATCH else "16k"
NEURONPEDIA_LABELS_PATH = os.environ.get(
    "NEURONPEDIA_LABELS_PATH",
    f"./local_data/neuronpedia_labels/neuronpedia_labels_{LAYER}-gemmascope-2-res-{_SAE_WIDTH}.json")
