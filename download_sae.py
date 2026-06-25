# download_sae.py
import os
import urllib3

# 1. Variables classiques
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

# 2. Forcer le chargement des sous-modules pour appliquer le patch au bon endroit
import huggingface_hub.utils
import huggingface_hub.file_download

_old_get_session = huggingface_hub.utils.get_session

def patched_get_session():
    session = _old_get_session()
    session.verify = False  # Désactive la vérification sur la session HF
    return session

# Application chirurgicale des patchs
huggingface_hub.utils.get_session = patched_get_session
huggingface_hub.file_download.get_session = patched_get_session

# Nettoyage des alertes
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 3. Import tardif de SAE pour qu'il hérite du patch ci-dessus
from sae_lens import SAE

print("Téléchargement et mise en cache complète du SAE...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2-4b-it-res",
    sae_id="layer_17_width_16k_l0_medium",
    device="cpu"
)
print("Fichiers mis en cache avec succès !")