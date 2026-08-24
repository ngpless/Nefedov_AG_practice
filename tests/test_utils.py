# -*- coding: utf-8 -*-
"""Тесты служебных функций."""
import numpy as np

from utils import generate_recommendations_for_user


def test_recommendations_sorted_and_limited():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    recs = generate_recommendations_for_user(
        user_idx=0, model_predictions=scores, n_items=5, n_recommendations=3)
    assert len(recs) == 3
    values = [scores[i] for i in recs]
    assert values == sorted(values, reverse=True)


def test_recommendations_within_item_range():
    scores = np.random.RandomState(0).rand(20)
    recs = generate_recommendations_for_user(
        user_idx=0, model_predictions=scores, n_items=20, n_recommendations=10)
    assert all(0 <= i < 20 for i in recs)
