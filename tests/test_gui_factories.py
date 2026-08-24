# -*- coding: utf-8 -*-
"""Тесты фабрик моделей графического интерфейса (без создания окна)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gui import MODEL_CLASSES  # noqa: E402


def test_four_architectures_registered():
    assert set(MODEL_CLASSES) == {"GMF", "MLP", "NCF", "Wide & Deep"}


def test_every_factory_builds_working_model():
    users = torch.randint(0, 10, (4,))
    items = torch.randint(0, 8, (4,))
    for name, factory in MODEL_CLASSES.items():
        model = factory(10, 8)
        out = model(users, items)
        assert out.shape == (4,), name
        assert torch.isfinite(out).all(), name
