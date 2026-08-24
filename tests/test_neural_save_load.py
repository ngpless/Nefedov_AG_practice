# -*- coding: utf-8 -*-
"""Тест сохранения и загрузки весов: предсказания идентичны."""
import torch

from neural_models import WideAndDeep


def test_state_dict_roundtrip(tmp_path):
    torch.manual_seed(7)
    model = WideAndDeep(20, 12, 8, [16, 8])
    users = torch.randint(0, 20, (10,))
    items = torch.randint(0, 12, (10,))
    model.eval()
    with torch.no_grad():
        before = model(users, items)

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    restored = WideAndDeep(20, 12, 8, [16, 8])
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    with torch.no_grad():
        after = restored(users, items)

    assert torch.allclose(before, after)
