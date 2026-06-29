import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# 1. Isolation stricte hors-ligne
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/home/h21486/SAE/models/gemma-3-12b-it"
SAE_SNAPSHOT_PATH = "/home/h21486/SAE/saes/gemma-scope-2-12b-it-res/snapshots/0000000000000000000000000000000000000000/resid_post/layer_24_width_16k_l0_medium"

print(f"Vérification sur l'environnement : {DEVICE}")
print("=" * 60)

# ─── ÉTAPE 1 : CHARGEMENT DE GEMMA-3-12B-IT ───
try:
    print(f"1. Chargement du tokenizer depuis : {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    print(f"2. Chargement de Gemma-3-12B-IT (bf16) sur {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto" if DEVICE == "cuda" else None,
        local_files_only=True
    ).eval()
    print("   [+] Modèle et Tokenizer chargés avec succès !")
    print(f"   [+] Allocation VRAM actuelle : {torch.cuda.memory_allocated(0) / 1e9:.2f} GB" if DEVICE == "cuda" else "")
except Exception as e:
    print(f"   [X] ERREUR ÉTAPE 1 : Impossible de charger le LLM. Détails :\n{e}")
    model = None

print("-" * 60)

# ─── ÉTAPE 2 : CHARGEMENT DU SAE ASSOCIÉ (COUCHE 24) ───
try:
    print(f"3. Chargement du SAE depuis le snapshot déporté : {SAE_SNAPSHOT_PATH}")
    if not os.path.isdir(SAE_SNAPSHOT_PATH):
        raise FileNotFoundError(f"Le sous-dossier de destination n'existe pas : {SAE_SNAPSHOT_PATH}")
        
    # Utilisation de l'API native sae_lens pour charger un répertoire absolu offline
    sae = SAE.load_from_disk(SAE_SNAPSHOT_PATH, device=DEVICE)
    print("   [+] Sparse Autoencoder (SAE) chargé avec succès !")
    print(f"   [+] Dimensions : d_in={sae.cfg.d_in} | d_sae={sae.cfg.d_sae}")
except Exception as e:
    print(f"   [X] ERREUR ÉTAPE 2 : Impossible de charger le SAE. Détails :\n{e}")
    sae = None

print("=" * 60)

# ─── ÉTAPE 3 : TEST DE VALIDATION DU FORWARD COMPACT ───
if model is not None and sae is not None:
    print("4. Exécution d'une passe d'inférence de validation...")
    try:
        text = "L'énergie nucléaire chez EDF"
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Récupération de la couche résiduelle cible (Layer 24)
            hidden_states = outputs.hidden_states[24].to(torch.bfloat16)
            
            # Normalisation RMS locale standard
            epsilon = 1e-6
            rms = torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + epsilon)
            normalized_states = hidden_states * rms
            
            # Encodage SAE
            feature_acts = sae.encode(normalized_states)
            
        print("   [+] Inférence validée sans crash !")
        print(f"   [+] Shape du flux résiduel (Layer 24) : {hidden_states.shape} (Attendu: [1, T, 4096])")
        print(f"   [+] Shape des activations SAE : {feature_acts.shape} (Attendu: [1, T, 16384])")
    except Exception as e:
        print(f"   [X] ERREUR ÉTAPE 3 : Échec de l'inférence. Détails :\n{e}")

# Nettoyage des tenseurs
del model, sae
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print("Test terminé.")