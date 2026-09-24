"""
src/config.py — Source unique des constantes de la pipeline.
Casse l'import circulaire phrase_sae → saev5 et centralise l'env.
Tout est surchargé par variables d'environnement (compat sbatch existants).
"""
import os
import re

SEED = int(os.environ.get("SEED", "42"))
# Seed DÉCOUPLÉE de SEED pour le split train/test du corpus emails
# (build_email_train_test_corpus) : permet de faire varier SEED (init des poids SAE,
# échantillonnage feature_selection_by_magnitude, etc.) pour une ablation de variance
# d'entraînement SANS changer le split train/test lui-même — sinon la comparaison
# entre deux SEED mélangerait variance d'entraînement et variance de corpus.
CORPUS_SPLIT_SEED = int(os.environ.get("CORPUS_SPLIT_SEED", "42"))

# ─── Pipeline 2 (F2LLM) ───
# F2LLM-v2-330M : la variante 80M n'était plus qu'un défaut historique, jamais
# révisé après la décision "backbone assez grand" (RESULTS_TESTS.md ~L815,
# results_v10_p2_f2llm330m/) -- restait un défaut divergent entre la doc et le
# code (désynchro EMB_MODEL, corrigée). Chemin LOCAL, pas un ID Hub
# ("codefuse-ai/F2LLM-v2-330M") : vérifié cette session, aucune entrée F2LLM dans
# ~/.cache/huggingface/hub -- un ID Hub échouerait sur un nœud de calcul offline
# (HF_HUB_OFFLINE=1) malgré les poids déjà présents localement. Toujours surchargeable.
EMB_MODEL      = os.environ.get("EMB_MODEL", "./models/F2LLM-v2-330M")
# "last_token" (défaut) : backbone décodeur causal (F2LLM). "cls" : backbone
# encodeur bidirectionnel entraîné pour ce pooling (bge-m3) -- cf.
# src/sae/phrase_sae.py::extract_f2llm_embeddings.
EMB_POOLING    = os.environ.get("EMB_POOLING", "last_token")
MATRYOSHKA_DIM = int(os.environ.get("MATRYOSHKA_DIM", "320"))

# Modèle d'embedding pour select_latents_by_similarity (src/sae/saev5.py) --
# recherche de latents par similarité de leur label à une requête (clustering
# ciblé + retrieval par propriétés). bge-m3 (pooling CLS, multilingue, conçu
# pour la similarité sémantique/retrieval) retenu après comparaison empirique
# à F2LLM (pooling dernier-token, adapté à la génération plutôt qu'à la
# similarité de courts labels).
LATENT_LABEL_EMB_MODEL = os.environ.get("LATENT_LABEL_EMB_MODEL", "./models/bge-m3")
D_SAE          = int(os.environ.get("D_SAE", "8192"))
K_SPARSE       = int(os.environ.get("K_SPARSE", "16"))
EPOCHS         = int(os.environ.get("EPOCHS", "30"))
LR             = float(os.environ.get("LR", "5e-4"))
BATCH_TRAIN    = int(os.environ.get("BATCH_TRAIN", "256"))
MAX_PHRASES_DOC = int(os.environ.get("MAX_PHRASES_DOC", "20"))

# ─── Pipeline 1 (FrozenCore) ───
D_EXTRA      = int(os.environ.get("D_EXTRA", "1024"))
# K=5 (SAE Boost, Koriagin et al.) : jamais adopté en défaut malgré un signal
# directionnel confirmé sur 3 graines dans ce dépôt (RESULTS_TESTS.md §45,
# z poolé=1,70, p=0,089 -- borderline mais cohérent).
K_EXTRA      = int(os.environ.get("K_EXTRA", "5"))
EPOCHS_EXTRA = int(os.environ.get("EPOCHS_EXTRA", "10"))
LR_EXTRA     = float(os.environ.get("LR_EXTRA", "3e-4"))
# Taille de batch d'entraînement de SAEBoostResidualSAE/FrozenCoreResidualSAE
# (sae_shared.py::load_or_train_extended_sae) -- 1024 codé en dur jusqu'ici.
# Sous BatchTopK (src/sae/batch.py), le budget top-k est PARTAGÉ sur le batch
# (top k·B pré-activations du batch aplati) : changer BATCH_SIZE_EXTRA change
# le régime de sparsité vu à l'entraînement (variance de L0 par échantillon),
# pas seulement la vitesse -- ne pas remonter le défaut sans ablation
# comparant la fidélité de reconstruction avant/après (docs/archive/audits/AUDIT_SAE_2026-08.md
# §2.9, item 7).
BATCH_SIZE_EXTRA = int(os.environ.get("BATCH_SIZE_EXTRA", "1024"))
USE_FROZEN_CORE = os.environ.get("USE_FROZEN_CORE", "1").strip() in ("1", "true", "True")
N_TOKENS_EXTRA_TRAIN = int(os.environ.get("N_TOKENS_EXTRA_TRAIN", "500000"))
# Longueur de troncature du tokenizer à l'extraction (saev5.py, boucle
# "Extraction P1") -- 512 était codé en dur, appliqué aux mails ENTIERS (pas
# chunkés, contrairement au filler) et jamais mesuré à ce seuil précis (N7,
# docs/archive/audits/AUDIT_SAE_2026-08.md §8 ; §74/RESULTS_TESTS.md mesure une troncature à 2048,
# sur le corpus d'AUGMENTATION, seuil et corpus différents). À rapprocher de
# ρ(longueur, n_features)=0,906 (§59) : une troncature agressive pourrait
# elle-même être une source de ce signal de longueur, pas juste le corréler.
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
# Seuil de clip des outliers de norme (src/analysis/activations.py::norm_outlier_mask,
# appelé depuis saev5.py) -- 4.0 codé en dur jusqu'ici (N8, docs/archive/audits/AUDIT_SAE_2026-08.md §8).
SIGMA_CLIP = float(os.environ.get("SIGMA_CLIP", "4.0"))
# Exclut le premier token de contenu (après les tokens spéciaux BOS/rôle) du
# masquage -- True codé en dur jusqu'ici (N8, docs/archive/audits/AUDIT_SAE_2026-08.md §8).
SKIP_FIRST_CONTENT_TOKEN = os.environ.get("SKIP_FIRST_CONTENT_TOKEN", "1").strip() in ("1", "true", "True")
# Taille de batch pour l'extraction Gemma-3 (saev5.py, boucle "Extraction P1") --
# 4 était codé en dur, jamais mesuré contre une valeur plus grande sur A100/H100
# (12B en simple passe avant, marge VRAM probable). Configurable pour permettre
# un balayage empirique avant le run de référence à grande échelle.
EXTRACTION_BATCH_SIZE = int(os.environ.get("EXTRACTION_BATCH_SIZE", "4"))
# Taille de batch pour le ré-encodage SAEBoostResidualSAE (saev5.py, audit perf
# §2.9 item 8) : les fragments d'un même lot sont concaténés le long de la
# dimension token avant UN SEUL appel à _encode_extra_acts, au lieu d'un appel
# par document. Équivalence exacte garantie par construction (BatchTopKEncoder
# en mode eval -- seuil global élément-par-élément, pas de budget partagé par
# batch comme en entraînement, cf. src/sae/batch.py) : la seule source d'écart
# possible est la non-associativité flottante du GEMM batché sur GPU, pas un
# changement de sémantique. 128 par défaut, prudent (VRAM du batch dépend de la
# longueur token totale du lot, variable d'un document à l'autre).
REENCODE_BATCH_SIZE = int(os.environ.get("REENCODE_BATCH_SIZE", "128"))
# Reprise après coupure (R1, docs/archive/audits/AUDIT_SAE_2026-08.md §2.3/§4.3) : nombre de documents
# entre deux checkpoints de progression de l'extraction P1 (compteurs du
# réservoir de Vitter + indice du prochain document à traiter). Borne le
# travail reperdu en cas de crash/SIGKILL à ce nombre de documents, pas à
# l'extraction entière -- ne pas descendre trop bas (checkpoint = écriture
# disque, même légère) ni trop haut (perte de travail proportionnelle).
EXTRACTION_CHECKPOINT_INTERVAL = int(os.environ.get("EXTRACTION_CHECKPOINT_INTERVAL", "2000"))
# Sanity-check (Korznikov et al. 2026, "Sanity Checks for Sparse Autoencoders : Do SAEs
# Beat Random Baselines?") : construit un FrozenDecoderExtendedSAE (décodeur figé,
# initialisation aléatoire jamais entraînée) à la place de SAEBoostResidualSAE, pour tester si nos
# métriques (juge odd-one-out, sondes de classification) distinguent un apprentissage de
# features significatif d'un simple ajustement de l'encodeur à des directions arbitraires.
SANITY_CHECK_FROZEN_DECODER = os.environ.get("SANITY_CHECK_FROZEN_DECODER", "0").strip() in ("1", "true", "True")
# "iso" (défaut, Gaussien isotrope normalisé -- uniforme sur la sphère) ou "cov"
# (Gaussien de covariance réelle, puis normalisé -- schéma PRINCIPAL de Korznikov
# et al., utilisé pour tous leurs résultats Frozen Decoder publiés car plus
# difficile à battre que iso, cf. frozen_core.py::FrozenDecoderExtendedSAE).
SANITY_CHECK_FROZEN_DECODER_INIT = os.environ.get("SANITY_CHECK_FROZEN_DECODER_INIT", "iso").strip().lower()
# Initialisation des directions du décodeur EXTRA de SAEBoostResidualSAE : "pca" (défaut,
# top-`D_EXTRA` directions PCA du résidu, déterministe sur le réservoir partagé -- SEED ne
# fait alors varier QUE l'ordre des mini-lots) ou "random" (gaussien isotrope normalisé
# sous SEED, échelles/biais calibrés à l'identique). "random" sert de bras témoin
# d'indépendance à l'initialisation (E05) et exige un SAVE_DIR distinct de tout run "pca" :
# `load_or_train_extended_sae` ne valide pas la config d'un checkpoint existant.
EXTRA_DECODER_INIT = os.environ.get("EXTRA_DECODER_INIT", "pca").strip().lower()
N_FEATURES_TO_LABEL  = int(os.environ.get("N_FEATURES_TO_LABEL", "10"))

# ─── Modèle Gemma-3 / GemmaScope ───
MODEL_SIZE = os.environ.get("MODEL_SIZE", "12b")

# (model_path, release_id, sae_id_default, layer, d_model)
# d_model sert de repli pour mocked_get_safetensors_tensor_shapes (saev5.py) quand
# aucune config locale n'est trouvée en cache. MODEL_ID pointe directement le
# repo HF (pas un chemin disque) : après download_sae.py, il est résolu depuis
# le cache HF par from_pretrained(local_files_only=True), portable entre
# machines. RELEASE_ID : le repo réel est "google/gemma-scope-2-{taille}-it",
# sans suffixe "-res".
_PRESETS = {
    # Couche 31 (~2/3 profondeur, 48 couches) plutôt que 24 (~0,5) : SPLARE
    # (Formal et al., NAVER Labs) situe la profondeur optimale à ~2/3 du modèle ;
    # layer 31 est aussi le SEUL résultat de balayage de couche nominalement
    # significatif mesuré dans ce dépôt (58,0% vs 45,3%, z=2,20, p=0,028,
    # RESULTS_TESTS.md §51, jamais répliqué sur une 2e graine). Largeur 16k
    # (pas 65k) à cette couche : c'est la configuration réellement testée en §51
    # (résultat lié à cette largeur précise, pas 65k à layer 31 -- jamais tourné) ;
    # coûte en couverture Neuronpedia (82,6% à 16k vs 87,8% à 65k, mesuré à
    # layer 24) mais reste la largeur
    # correspondant au résultat empirique retenu, pas une combinaison inédite.
    # Couche 40 (~2/3 profondeur, 62 couches) : même logique que 12b ci-dessous --
    # GemmaScope-2 27B publie resid_post curé à layers {16,31,40,53} (~25/50/65/85%
    # de profondeur, mêmes proportions que le jeu 12/24/31/41 de 12b), 40 est le
    # plus proche de 2/3. Jamais testé (nouvelle taille, ablation en cours).
    "27b":  ("google/gemma-3-27b-it", "gemma-scope-2-27b-it", "layer_40_width_16k_l0_medium", 40, 5376),
    "12b":  ("google/gemma-3-12b-it", "gemma-scope-2-12b-it", "layer_31_width_16k_l0_medium", 31, 3840),
    "4b":   ("google/gemma-3-4b-it",  "gemma-scope-2-4b-it",  "layer_17_width_16k_l0_medium", 17, 2560),
    "1b":   ("google/gemma-3-1b-it",  "gemma-scope-2-1b-it",  "layer_13_width_16k_l0_medium", 13, 1152),
    # google/gemma-3-270m-it (LM) + google/gemma-scope-2-270m-it (SAE, resid_post,
    # layer 12, largeur 65k confirmée via Neuronpedia). d_model=640 confirmé
    # empiriquement (w_enc.shape du SAE téléchargé). Profil réduit (6 Go VRAM)
    # pour validation locale du pipeline de bout en bout.
    "270m": ("google/gemma-3-270m-it", "gemma-scope-2-270m-it", "layer_12_width_65k_l0_medium", 12, 640),
}
_m, RELEASE_ID, _sae_default, LAYER, D_MODEL = _PRESETS.get(MODEL_SIZE, _PRESETS["270m"])
MODEL_ID  = os.environ.get("MODEL_ID", _m)
SAE_ID    = os.environ.get("SAE_ID", _sae_default)
HOOK_TYPE = os.environ.get("HOOK_TYPE", "resid_post")
# LAYER par défaut vient du preset MODEL_SIZE (31 pour 12b, RESULTS_TESTS.md
# §51) ; overridable pour tester les autres layers "curés" (12/24/41) publiés
# par GemmaScope-2 pour gemma-3-12b-it. SAE_ID doit être mis à jour en
# cohérence (le layer y est encodé dans le nom, ex. "layer_24_width_16k_l0_medium").
LAYER = int(os.environ.get("LAYER", LAYER))
LOCAL_SAE_ROOT = os.environ.get("LOCAL_SAE_DIR", f"./local_data/saes/{RELEASE_ID}")

# Juge LLM (odd_one_out_judge, local_gemma_judge), DÉCOUPLÉ de MODEL_ID
# (l'extracteur) -- recharger le même checkpoint Gemma-3 comme juge de ses
# propres features expose à un biais d'auto-préférence jamais isolé d'un
# simple effet de capacité (RESULTS_TESTS.md §43/§63/§65 : gemma-3-4b-it vs
# gemma-3-12b-it, même famille, écart de juge confirmé et robuste au seed
# mais ne tranche pas entre les deux explications -- un juge de famille
# différente était identifié comme manquant). Qwen3.8-27B (capacité
# comparable au palier 27b, famille différente) sert de juge par défaut --
# bf16, PAS la variante FP8 (`unsloth/Qwen3.8-27B-FP8`) : son inférence
# échoue sur les deux chemins possibles avec torch==2.6.0 (pin du dépôt),
# aucun rapport avec un paquet manquant -- cf. docstring de
# src/sae/judge.py::load_judge_model pour le détail des deux échecs.
JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "./models/Qwen3.8-27B")
SAE_SNAPSHOT   = os.environ.get("SAE_SNAPSHOT", "0" * 40)

# ─── Précision ───
# bf16 par défaut, y compris en local. Testé empiriquement : Gemma-3 a des activations
# "massives" documentées dans le residual stream (outliers ~1e5) qui dépassent le max
# représentable en fp16 (~65504) -> overflow silencieux vers inf/nan, qui contamine tout
# l'entraînement de SAEBoostResidualSAE (Loss=nan dès l'epoch 1, confirmé sur run local 270m).
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
    # "./datasets/...", pas "./local_data/datasets/..." (défaut historique jamais
    # atteint sur le disque réel -- tous les slurm/pipeline_runs/*.slurm surchargent
    # déjà cette variable vers le bon chemin ; corrigé ici pour que le défaut soit
    # utilisable sans surcharge, ex. src/sae/retrieval/latent_terms.py).
    "./datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet")
# Corpus emails EDF originaux + variantes augmentées (générées par
# scripts/run_augmentation.py) : emplacement canonique unique, utilisé par
# saev5.py, scripts/run_augmentation.py et scripts/baseline_gemmascope.py.
LOCAL_MAILS_PATH = os.environ.get("LOCAL_MAILS_PATH", "./local_data/emails/Mails.tsv")
LOCAL_AUGMENTED_MAILS_PATH = os.environ.get(
    "LOCAL_AUGMENTED_MAILS_PATH", "./local_data/emails/augmented_mails.jsonl")

# Labels Neuronpedia (cf. src/sae/neuronpedia_labels.py) : cache partagé entre TOUS
# les runs (indépendant de SAVE_DIR/CACHE_DIR), régénérable hors-cluster via
# fetch_neuronpedia_labels() mais réutilisé tel quel une fois présent -- jamais
# re-téléchargé, jamais dupliqué par run. Override par env si besoin d'un autre jeu.
# Largeur dérivée de SAE_ID (ex. "layer_24_width_65k_l0_medium" -> "65k") au lieu
# d'être figée en dur : varie selon MODEL_SIZE (cf. _PRESETS ci-dessus).
_WIDTH_MATCH = re.search(r"width_(\w+?)_l0", SAE_ID)
_SAE_WIDTH = _WIDTH_MATCH.group(1) if _WIDTH_MATCH else "16k"
NEURONPEDIA_LABELS_PATH = os.environ.get(
    "NEURONPEDIA_LABELS_PATH",
    f"./local_data/neuronpedia_labels/neuronpedia_labels_{LAYER}-gemmascope-2-res-{_SAE_WIDTH}.json")
