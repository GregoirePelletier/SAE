import json
import math
import os

import torch

SHARD_SIZE_GB = float(os.environ.get("SHARD_SIZE_GB", "4.0"))


def save_activations_sharded(acts: torch.Tensor, prefix: str) -> list[str]:
    acts_b16 = acts.to(torch.bfloat16)
    n_total = acts_b16.shape[0]
    bytes_total = acts_b16.numel() * 2
    n_shards = max(1, math.ceil(bytes_total / (SHARD_SIZE_GB * 1e9)))
    shard_size = math.ceil(n_total / n_shards)
    paths = []
    for i in range(n_shards):
        chunk = acts_b16[i * shard_size: (i + 1) * shard_size].clone()
        path = f"{prefix}_shard{i:03d}of{n_shards:03d}.pt"
        torch.save(chunk, path)
        paths.append(path)
    manifest = {"prefix": prefix, "n_shards": n_shards, "n_total": n_total,
                "dtype": "bfloat16", "d_in": acts_b16.shape[1], "paths": paths}
    with open(prefix + "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return paths


def load_activations_mmap(prefix: str) -> torch.Tensor:
    manifest_path = prefix + "_manifest.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest introuvable : {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    shards = [torch.load(p, weights_only=True, mmap=True) for p in manifest["paths"]]
    acts = torch.cat(shards, dim=0)
    print(f"  [sae_shared] Chargé {acts.shape} depuis {manifest['n_shards']} shards (mmap)")
    return acts
