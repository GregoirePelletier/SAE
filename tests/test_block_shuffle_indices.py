"""Teste src/sae/sae_shared.py::block_shuffle_indices -- remplace
train_idx[torch.randperm(len(train_idx))] par époque dans
load_or_train_extended_sae (AUDIT_SAE_2026-08.md, §2 Performance :
torch.randperm(100_000_000) réalloué à chaque époque)."""
import torch

from src.sae.sae_shared import block_shuffle_indices


def test_block_shuffle_is_a_valid_permutation():
    idx = torch.arange(10_000)
    out = block_shuffle_indices(idx, block_size=64)
    assert out.shape == idx.shape
    assert torch.equal(torch.sort(out).values, torch.sort(idx).values)


def test_block_shuffle_actually_shuffles():
    idx = torch.arange(10_000)
    out = block_shuffle_indices(idx, block_size=64)
    assert not torch.equal(out, idx)


def test_block_shuffle_reproducible_with_generator_seed():
    idx = torch.arange(5_000)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = block_shuffle_indices(idx, block_size=32, generator=g1)
    out2 = block_shuffle_indices(idx, block_size=32, generator=g2)
    assert torch.equal(out1, out2)


def test_block_shuffle_falls_back_to_full_randperm_when_n_leq_block_size():
    idx = torch.arange(20)
    out = block_shuffle_indices(idx, block_size=64)
    assert torch.equal(torch.sort(out).values, torch.sort(idx).values)


def test_block_shuffle_handles_non_multiple_of_block_size():
    # 10_003 n'est pas un multiple de 64 -- le dernier bloc est partiel.
    idx = torch.arange(10_003)
    out = block_shuffle_indices(idx, block_size=64)
    assert torch.equal(torch.sort(out).values, torch.sort(idx).values)


def test_block_shuffle_works_on_arbitrary_index_values():
    # train_idx n'est pas forcément 0..n-1 contigu (c'est un sous-ensemble
    # après le split de validation) -- vérifie que ça tient sur des valeurs
    # non triées/non contiguës.
    idx = torch.randperm(3_000)[:2_000]
    out = block_shuffle_indices(idx, block_size=128)
    assert torch.equal(torch.sort(out).values, torch.sort(idx).values)
