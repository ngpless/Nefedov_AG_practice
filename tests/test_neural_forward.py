# -*- coding: utf-8 -*-
"""Тесты прямого прохода нейросетевых архитектур."""
import torch
import pytest

from neural_models import (GMF, MLP_Recommender,
                           NeuralCollaborativeFiltering, WideAndDeep)

N_USERS, N_ITEMS, BATCH = 30, 20, 16

ARCHITECTURES = [
    ("GMF", lambda: GMF(N_USERS, N_ITEMS, embed_dim=8)),
    ("MLP", lambda: MLP_Recommender(N_USERS, N_ITEMS, 8, [16, 8])),
    ("NCF", lambda: NeuralCollaborativeFiltering(N_USERS, N_ITEMS, 8, 8, [16, 8])),
    ("WideDeep", lambda: WideAndDeep(N_USERS, N_ITEMS, 8, [16, 8])),
]


@pytest.mark.parametrize("name,factory", ARCHITECTURES)
def test_forward_shape_and_finiteness(name, factory):
    model = factory()
    users = torch.randint(0, N_USERS, (BATCH,))
    items = torch.randint(0, N_ITEMS, (BATCH,))
    out = model(users, items)
    assert out.shape == (BATCH,), name
    assert torch.isfinite(out).all(), name


@pytest.mark.parametrize("name,factory", ARCHITECTURES)
def test_parameters_counted(name, factory):
    model = factory()
    assert model.count_parameters() > 0, name


@pytest.mark.parametrize("name,factory", ARCHITECTURES)
def test_backward_pass_produces_gradients(name, factory):
    model = factory()
    users = torch.randint(0, N_USERS, (BATCH,))
    items = torch.randint(0, N_ITEMS, (BATCH,))
    target = torch.rand(BATCH) * 4 + 1
    loss = torch.nn.functional.mse_loss(model(users, items), target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), name
