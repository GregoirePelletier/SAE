"""Teste le stockage shardé des fragments (item 3, AUDIT_SAE_2026-08.md
§2.2/§2.9) : ShardedFragmentWriter (SHARD_SIZE documents par fichier au lieu
d'un par document, utilisé côté extraction) et la lecture transparente
(load_fragment/list_fragment_ids/fragment_exists) qui doit voir aussi bien
les shards que les fichiers individuels legacy (écrits par save_fragment,
ex. ré-encodage) -- avec priorité au fichier individuel, la version la plus
récente d'un document déjà réécrit après extraction."""
import torch

from src.storage import fragment_store as fs
from src.storage.fragment_store import (
    ShardedFragmentWriter, save_fragment, load_fragment,
    list_fragment_ids, fragment_exists,
)


def _tiny_acts(t=3, d=8):
    acts = torch.zeros(t, d)
    acts[0, 1] = 0.5
    acts[1, 3] = 0.7
    return acts


def test_round_trip_across_multiple_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "SHARD_SIZE", 3)  # shards minuscules pour exercer plusieurs fichiers
    fdir = str(tmp_path)
    writer = ShardedFragmentWriter(fdir)
    for doc_id in range(7):  # couvre les shards 0 (0-2), 1 (3-5), 2 (6)
        writer.add(doc_id, token_strings=[f"tok{doc_id}"], acts_dense=_tiny_acts(), d_total=8)
    writer.close()

    ids = list_fragment_ids(fdir)
    assert ids == list(range(7))
    for doc_id in range(7):
        assert fragment_exists(fdir, doc_id)
        frag = load_fragment(fdir, doc_id)
        assert frag["token_strings"] == [f"tok{doc_id}"]
        assert frag["shape"] == (3, 8)


def test_resume_into_partial_shard_preserves_earlier_docs(tmp_path, monkeypatch):
    """Simule une coupure en cours de shard : un premier writer flush un
    shard partiel (2/3 documents), un second writer (nouveau process, même
    répertoire) reprend et complète -- le flush final doit contenir les
    TROIS documents, pas seulement le dernier ajouté par le second writer."""
    monkeypatch.setattr(fs, "SHARD_SIZE", 3)
    fdir = str(tmp_path)

    writer1 = ShardedFragmentWriter(fdir)
    writer1.add(0, token_strings=["a"], acts_dense=_tiny_acts(), d_total=8)
    writer1.add(1, token_strings=["b"], acts_dense=_tiny_acts(), d_total=8)
    writer1.flush()  # coupure ici -- le fichier shard_00000.pt contient déjà 0 et 1
    writer1._async_writer.close()  # ferme juste le thread, pas de flush supplémentaire

    writer2 = ShardedFragmentWriter(fdir)
    writer2.add(2, token_strings=["c"], acts_dense=_tiny_acts(), d_total=8)  # complète le shard
    writer2.close()

    ids = list_fragment_ids(fdir)
    assert ids == [0, 1, 2]
    for doc_id, expected in enumerate(["a", "b", "c"]):
        assert load_fragment(fdir, doc_id)["token_strings"] == [expected]


def test_individual_file_overrides_shard_content(tmp_path, monkeypatch):
    """Simule le ré-encodage : un document initialement écrit dans un shard
    (extraction) est ensuite réécrit en fichier individuel (save_fragment,
    comme le fait la passe de ré-encodage) -- load_fragment doit renvoyer la
    version individuelle, pas celle du shard."""
    monkeypatch.setattr(fs, "SHARD_SIZE", 3)
    fdir = str(tmp_path)

    writer = ShardedFragmentWriter(fdir)
    writer.add(0, token_strings=["original"], acts_dense=_tiny_acts(), d_total=8)
    writer.close()
    assert load_fragment(fdir, 0)["token_strings"] == ["original"]

    save_fragment(fdir, 0, token_strings=["reencoded"], acts_dense=_tiny_acts(d=12), d_total=12)
    frag = load_fragment(fdir, 0)
    assert frag["token_strings"] == ["reencoded"]
    assert frag["shape"] == (3, 12)


def test_read_cache_invalidated_after_writer_flush(tmp_path, monkeypatch):
    """Reproduit le scénario de péremption identifié : lire un shard (le
    met en cache) PUIS écrire dedans via un writer -- une lecture
    ultérieure du MÊME shard doit voir le contenu frais, pas l'ancien."""
    monkeypatch.setattr(fs, "SHARD_SIZE", 10)
    fdir = str(tmp_path)

    writer = ShardedFragmentWriter(fdir)
    writer.add(0, token_strings=["v1"], acts_dense=_tiny_acts(), d_total=8)
    writer.flush()

    assert load_fragment(fdir, 0)["token_strings"] == ["v1"]  # peuple le cache pour shard 0

    writer.add(1, token_strings=["v2"], acts_dense=_tiny_acts(), d_total=8)  # même shard (0)
    writer.flush()

    # Sans invalidation, load_fragment(1) irait chercher dans le cache
    # périmé (qui ne contenait que le doc 0 au moment de la lecture
    # précédente) et lèverait FileNotFoundError malgré l'écriture réussie.
    assert load_fragment(fdir, 1)["token_strings"] == ["v2"]
    writer.close()


def test_fragment_exists_false_for_unknown_doc(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "SHARD_SIZE", 5)
    fdir = str(tmp_path)
    writer = ShardedFragmentWriter(fdir)
    writer.add(0, token_strings=["a"], acts_dense=_tiny_acts(), d_total=8)
    writer.close()
    assert fragment_exists(fdir, 0)
    assert not fragment_exists(fdir, 99)
