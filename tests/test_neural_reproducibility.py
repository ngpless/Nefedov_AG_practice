# -*- coding: utf-8 -*-
"""Тест воспроизводимости: одинаковый seed — одинаковая инициализация."""
import torch

from neural_models import GMF, NeuralModelTrainer


def _fresh_model():
    NeuralModelTrainer(device="cpu")  # конструктор фиксирует seed
    return GMF(25, 15, embed_dim=8)


def test_same_seed_same_initial_weights():
    m1 = _fresh_model()
    m2 = _fresh_model()
    for (n1, p1), (n2, p2) in zip(m1.state_dict().items(),
                                  m2.state_dict().items()):
        assert n1 == n2
        assert torch.allclose(p1, p2), n1
