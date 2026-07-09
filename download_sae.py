"""
download_sae.py — Télécharge et met en cache localement le modèle Gemma-3 et le
SAE GemmaScope-2 correspondants à MODEL_SIZE (src/config.py), au lieu des valeurs
"gemma-scope-2-4b-it-res" figées en dur historiquement.

Usage :
    MODEL_SIZE=12b python download_sae.py   # ou 4b / 1b / 270m (cf. src/config.py _PRESETS)
"""
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("download_sae")

from src.config import (
    MODEL_SIZE, MODEL_ID, RELEASE_ID, SAE_ID, HOOK_TYPE,
    LOCAL_SAE_ROOT, SAE_SNAPSHOT, HF_TOKEN, CLUSTER_OFFLINE_MODE,
)

if CLUSTER_OFFLINE_MODE:
    # Réplique les patchs réseau cluster (proxy à certificat auto-signé) — voir saev5.py.
    import urllib3
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    import huggingface_hub.utils
    import huggingface_hub.file_download
    _old_get_session = huggingface_hub.utils.get_session

    def _patched_get_session():
        session = _old_get_session()
        session.verify = False
        return session

    huggingface_hub.utils.get_session = _patched_get_session
    huggingface_hub.file_download.get_session = _patched_get_session
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_model(model_id: str) -> None:
    """Télécharge/caches le tokenizer + les poids du LM (métadonnées uniquement,
    pas de chargement en mémoire/VRAM ici)."""
    from huggingface_hub import snapshot_download
    log.info(f"[Modèle] Téléchargement de {model_id} ...")
    snapshot_download(repo_id=model_id, token=HF_TOKEN)
    log.info("[Modèle] OK.")


def download_sae(release_id: str, hook_type: str, sae_id: str, local_sae_root: str,
                  snapshot: str) -> str:
    """
    Télécharge le sous-dossier {hook_type}/{sae_id} du repo HF google/{release_id}
    vers local_sae_root/snapshots/{snapshot}/{hook_type}/{sae_id}/, structure attendue
    par src.sae.gemma_scope_loader.load_gemma_scope_sae() / saev5.load_pretrained_sae().
    Retourne le chemin local du SAE téléchargé.
    """
    from huggingface_hub import snapshot_download
    repo_id = f"google/{release_id}"
    dest_snapshot_dir = os.path.join(local_sae_root, "snapshots", snapshot)
    pattern = f"{hook_type}/{sae_id}/*"
    log.info(f"[SAE] Téléchargement de {repo_id} (motif={pattern}) -> {dest_snapshot_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest_snapshot_dir,
        allow_patterns=[pattern],
        token=HF_TOKEN,
    )
    sae_dir = os.path.join(dest_snapshot_dir, hook_type, sae_id)
    if not os.path.isdir(sae_dir):
        raise FileNotFoundError(
            f"Le motif '{pattern}' n'a rien téléchargé depuis {repo_id}. "
            f"Vérifiez le nom exact de la release/sae_id (cf. la page HF du repo)."
        )
    log.info(f"[SAE] OK : {sae_dir}")
    return sae_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-only", action="store_true", help="Ne télécharge que le LM.")
    parser.add_argument("--sae-only", action="store_true", help="Ne télécharge que le SAE.")
    args = parser.parse_args()

    log.info(f"MODEL_SIZE={MODEL_SIZE}  MODEL_ID={MODEL_ID}")
    log.info(f"RELEASE_ID={RELEASE_ID}  SAE_ID={HOOK_TYPE}/{SAE_ID}")

    if not args.sae_only:
        download_model(MODEL_ID)
    if not args.model_only:
        download_sae(RELEASE_ID, HOOK_TYPE, SAE_ID, LOCAL_SAE_ROOT, SAE_SNAPSHOT)


if __name__ == "__main__":
    main()
