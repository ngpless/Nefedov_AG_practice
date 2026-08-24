# -*- coding: utf-8 -*-
"""Тесты ансамблевых методов на фиктивных предсказаниях."""
import numpy as np
import pandas as pd

from ensemble_methods import RecommenderEnsembleTrainer


def _fake_test(n=200, seed=1):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"rating": rng.choice([1, 2, 3, 4, 5], n)})


def test_simple_average_is_mean_of_inputs():
    test = _fake_test()
    preds = {"a": np.full(len(test), 2.0), "b": np.full(len(test), 4.0)}
    res = RecommenderEnsembleTrainer().simple_average_ensemble(test, preds)
    assert np.allclose(res["predictions"], 3.0)


def test_weighted_average_prefers_better_model():
    test = _fake_test()
    y = test["rating"].values.astype(float)
    good = y + 0.1          # почти идеальные предсказания
    bad = np.full(len(test), 3.0)
    preds = {"good": good, "bad": bad}
    val_rmse = {"good": 0.1, "bad": 1.2}
    res = RecommenderEnsembleTrainer().weighted_average_ensemble(
        test, preds, val_rmse)
    w = dict(zip(preds.keys(), res["weights"]))
    assert w["good"] > w["bad"]


def test_voting_returns_valid_ratings():
    test = _fake_test()
    rng = np.random.RandomState(2)
    preds = {k: rng.uniform(1, 5, len(test)) for k in "abc"}
    res = RecommenderEnsembleTrainer().voting_ensemble(test, preds)
    out = np.asarray(res["predictions"])
    assert set(np.unique(out)).issubset({1.0, 2.0, 3.0, 4.0, 5.0})
