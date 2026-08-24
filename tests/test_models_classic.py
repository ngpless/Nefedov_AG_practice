# -*- coding: utf-8 -*-
"""Тесты классических моделей рекомендаций."""
import numpy as np
import pytest

from models import RecommenderModelTrainer


@pytest.fixture(scope="module")
def base_results(prepared):
    _, _, train, test, matrix = prepared
    trainer = RecommenderModelTrainer()
    return trainer.train_base_models(train, test, matrix), test


def test_all_seven_models_trained(base_results):
    results, _ = base_results
    assert len(results) == 7


def test_predictions_cover_test_set(base_results):
    results, test = base_results
    for name, res in results.items():
        assert len(res["predictions"]) == len(test), name


def test_metrics_are_finite_and_positive(base_results):
    results, _ = base_results
    for name, res in results.items():
        assert np.isfinite(res["rmse"]) and res["rmse"] > 0, name
        assert np.isfinite(res["mae"]) and res["mae"] > 0, name
        assert res["mae"] <= res["rmse"] + 1e-9, name


def test_predictions_within_rating_bounds(base_results):
    results, _ = base_results
    for name, res in results.items():
        preds = np.asarray(res["predictions"], dtype=float)
        assert preds.min() >= 1 - 1e-9, name
        assert preds.max() <= 5 + 1e-9, name
