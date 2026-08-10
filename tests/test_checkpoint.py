"""Smoke-test PyTorch générique (save/load state_dict) -- ne teste aucun code
du dépôt (AUDIT_REPO_2026-08-07.md §4.3)."""
import torch

def test_save_load(tmp_path):

    model = torch.nn.Linear(128, 64)

    path = tmp_path / "model.pt"

    torch.save(model.state_dict(), path)

    model2 = torch.nn.Linear(128, 64)

    model2.load_state_dict(torch.load(path))

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)