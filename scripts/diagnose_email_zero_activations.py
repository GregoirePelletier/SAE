"""Diagnostic CPU-only : le run précédent (email_interp_embed_encode_test.py,
job 44413) a rapporté 0/65536 features actives sur 1500 mails -- trop extrême
pour être plausible même sous forte dérive de domaine. Inspecte le pickle
sauvegardé (activations RAW par token, pas l'agrégat) pour localiser si le
problème vient de l'encodage lui-même (SAE) ou de l'agrégation/binarize."""
import sys
sys.path.insert(0, "/home/h21486/SAE/external/interp_embed")

import numpy as np

from interp_embed.utils.helpers import safe_load_pkl

path = "/home/h21486/SAE/local_data/email_interp_embed/encoded_emails.pkl"
d = safe_load_pkl(path)

print("Keys:", list(d.keys()))
print("n rows:", len(d["rows"]))
n_none = sum(1 for r in d["rows"] if r is None)
print("n None rows:", n_none)

for i in range(5):
    row = d["rows"][i]
    if row is None:
        print(f"row {i}: None")
        continue
    print(f"row {i}: shape={row.shape}, nnz={row.nnz}, "
          f"max={row.data.max() if row.nnz else 0}, "
          f"min_nonzero={row.data.min() if row.nnz else 0}")

all_nnz = sum(r.nnz for r in d["rows"] if r is not None)
print(f"\nTotal nnz across all rows (raw per-token, uncompressed CSR): {all_nnz}")

# Vérifie aussi les aggregate_activations sauvegardées (max/sum par doc)
agg = d.get("aggregate_activations")
if agg:
    for i in range(5):
        a = agg[i]
        if a is None:
            continue
        for k, v in a.items():
            print(f"row {i} aggregate[{k!r}]: nnz={v.nnz if hasattr(v, 'nnz') else 'N/A'}, "
                  f"max={v.max() if hasattr(v, 'max') else 'N/A'}")

print("\nsae_metadata:", d.get("sae_metadata"))
