# -*- coding: utf-8 -*-
"""Интеграционный тест: короткое обучение снижает ошибку модели."""
import numpy as np
import torch

from neural_models import GMF, NeuralModelTrainer


def _rmse(model, users, items, ratings):
    model.eval()
    with torch.no_grad():
        preds = torch.clamp(model(users, items), 1, 5)
    return float(torch.sqrt(torch.mean((preds - ratings) ** 2)))


def test_two_epochs_reduce_training_error(prepared):
    _, _, train, test, _ = prepared
    trainer = NeuralModelTrainer(device="cpu")
    n_users = int(train["user_idx"].max()) + 1
    n_items = int(train["item_idx"].max()) + 1
    model = GMF(n_users, n_items, embed_dim=8)

    users = torch.tensor(train["user_idx"].values, dtype=torch.long)
    items = torch.tensor(train["item_idx"].values, dtype=torch.long)
    ratings = torch.tensor(train["rating"].values, dtype=torch.float32)

    rmse_before = _rmse(model, users, items, ratings)

    train_loader, val_loader, _ = trainer.create_train_val_test_loaders(
        train, test, batch_size=128)
    trainer.train_model(model, train_loader, val_loader, val_loader,
                        epochs=2, lr=0.01, patience=5)

    rmse_after = _rmse(model, users, items, ratings)
    assert rmse_after < rmse_before
    assert np.isfinite(rmse_after)
