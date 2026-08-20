"""Teste le mécanisme de reprise par shards de
src/sae/phrase_sae.py::extract_f2llm_embeddings (R1, AUDIT_SAE_2026-08.md
§2.3/§4.3). Le modèle F2LLM lui-même n'est pas chargé ici (GPU/poids requis,
hors périmètre d'un test CPU) -- ce test isole le mécanisme réellement
nouveau et risqué : écrire des shards + un checkpoint, puis les relire et les
reconcaténer dans le même ordre pour reconstruire le tenseur final, exactement
comme le fait la fonction (chargement des shards existants avant de reprendre
la boucle de calcul)."""
import os

import torch

from src.storage.checkpoint import read_checkpoint, write_checkpoint


def _write_shards(shard_dir, chunks):
    os.makedirs(shard_dir, exist_ok=True)
    for s, chunk in enumerate(chunks):
        torch.save(chunk, os.path.join(shard_dir, f"shard_{s:05d}.pt"))


def _load_shards(shard_dir, n_shards):
    return [torch.load(os.path.join(shard_dir, f"shard_{s:05d}.pt"), map_location="cpu")
            for s in range(n_shards)]


def test_shards_reload_in_order_reconstruct_original_tensor(tmp_path):
    torch.manual_seed(0)
    full = torch.randn(250, 8)
    chunks = [full[0:100], full[100:220], full[220:250]]  # tailles hétérogènes, comme un dernier shard partiel
    shard_dir = str(tmp_path / "emb_shards")
    _write_shards(shard_dir, chunks)

    reloaded = torch.cat(_load_shards(shard_dir, len(chunks)), dim=0)
    assert torch.equal(reloaded, full)


def test_checkpoint_next_idx_matches_cumulative_shard_length(tmp_path):
    """Invariant dont dépend la reprise : next_idx persisté == somme des
    longueurs des shards déjà écrits -- sinon la boucle reprendrait soit en
    sautant des phrases (perte silencieuse), soit en en retraitant (doublons)."""
    torch.manual_seed(0)
    shard_dir = str(tmp_path / "emb_shards")
    progress_path = str(tmp_path / "emb_shards.progress.json")
    os.makedirs(shard_dir, exist_ok=True)

    chunks = [torch.randn(100, 4), torch.randn(100, 4), torch.randn(37, 4)]
    cumulative = 0
    for s, chunk in enumerate(chunks):
        torch.save(chunk, os.path.join(shard_dir, f"shard_{s:05d}.pt"))
        cumulative += chunk.shape[0]
        write_checkpoint(progress_path, next_idx=cumulative, n_shards=s + 1)

    progress = read_checkpoint(progress_path)
    assert progress["next_idx"] == sum(c.shape[0] for c in chunks) == 237
    assert progress["n_shards"] == len(chunks)

    reloaded = torch.cat(_load_shards(shard_dir, progress["n_shards"]), dim=0)
    assert reloaded.shape[0] == progress["next_idx"]


def test_resume_simulation_matches_uninterrupted_run(tmp_path):
    """Simule exactement le flux de extract_f2llm_embeddings : une exécution
    interrompue après le 2e shard, puis reprise à partir de next_idx, doit
    produire le même tenseur final qu'une exécution continue -- sur des
    "embeddings" synthétiques déterministes (indice répété comme valeur,
    pour détecter tout décalage/doublon d'indices à la reconstruction)."""
    n_total, shard_size = 530, 100

    def fake_batch_embed(start, end):
        # "embedding" = l'indice répété -- un décalage ou doublon d'indices
        # se voit immédiatement dans les valeurs reconstruites.
        return torch.arange(start, end, dtype=torch.float32).unsqueeze(1).expand(-1, 3).clone()

    def run(shard_dir, progress_path, resume_from, all_embs_init, n_shards_init):
        all_embs = list(all_embs_init)
        n_shards_written = n_shards_init
        current_shard, current_count = [], 0

        def flush(next_idx):
            nonlocal current_shard, current_count, n_shards_written
            if not current_shard:
                return
            t = torch.cat(current_shard, dim=0)
            all_embs.append(t)
            torch.save(t, os.path.join(shard_dir, f"shard_{n_shards_written:05d}.pt"))
            n_shards_written += 1
            current_shard, current_count = [], 0
            write_checkpoint(progress_path, next_idx=next_idx, n_shards=n_shards_written)

        i = resume_from
        while i < n_total:
            batch_end = min(i + 32, n_total)
            emb = fake_batch_embed(i, batch_end)
            current_shard.append(emb)
            current_count += emb.shape[0]
            next_idx = batch_end
            if current_count >= shard_size:
                flush(next_idx)
            i = batch_end
        flush(n_total)
        return torch.cat(all_embs, dim=0)

    # Run continu (référence)
    ref_dir = str(tmp_path / "ref_shards")
    os.makedirs(ref_dir, exist_ok=True)
    reference = run(ref_dir, str(tmp_path / "ref.progress.json"), 0, [], 0)

    # Run "interrompu puis repris" : on simule la coupure en tronquant
    # manuellement après les 2 premiers shards plutôt qu'en cassant le process
    # (impossible à faire proprement dans un test), puis on relit le
    # checkpoint comme le ferait un second lancement.
    split_dir = str(tmp_path / "split_shards")
    os.makedirs(split_dir, exist_ok=True)
    progress_path = str(tmp_path / "split.progress.json")
    # Phase 1 : traite jusqu'à ~2 shards puis "coupe" (on arrête la boucle nous-mêmes).
    i, n_shards_written, current_shard, current_count = 0, 0, [], 0
    while i < 250:  # s'arrête après ~2 shards pleins (200 items), avant la fin
        batch_end = min(i + 32, 250)
        emb = fake_batch_embed(i, batch_end)
        current_shard.append(emb)
        current_count += emb.shape[0]
        if current_count >= shard_size:
            t = torch.cat(current_shard, dim=0)
            torch.save(t, os.path.join(split_dir, f"shard_{n_shards_written:05d}.pt"))
            n_shards_written += 1
            current_shard, current_count = [], 0
            write_checkpoint(progress_path, next_idx=batch_end, n_shards=n_shards_written)
        i = batch_end
    # "Coupure" ici -- current_shard partiel perdu, comme un vrai crash.

    # Phase 2 : reprise, exactement comme extract_f2llm_embeddings au démarrage.
    progress = read_checkpoint(progress_path)
    resume_from = progress["next_idx"]
    preloaded = [torch.load(os.path.join(split_dir, f"shard_{s:05d}.pt"), map_location="cpu")
                 for s in range(progress["n_shards"])]
    result = run(split_dir, progress_path, resume_from, preloaded, progress["n_shards"])

    assert torch.equal(result, reference)
