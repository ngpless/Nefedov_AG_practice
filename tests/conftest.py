# -*- coding: utf-8 -*-
"""Общие фикстуры тестов: небольшой синтетический набор оценок."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import RecommenderDataProcessor  # noqa: E402

N_USERS = 60
N_ITEMS = 40
N_RATINGS = 1500


@pytest.fixture(scope="session")
def ratings_df():
    """Синтетический DataFrame оценок с фиксированным seed."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "user_id": rng.randint(1, N_USERS + 1, N_RATINGS),
        "item_id": rng.randint(1, N_ITEMS + 1, N_RATINGS),
        "rating": rng.choice([1, 2, 3, 4, 5], N_RATINGS,
                             p=[0.05, 0.10, 0.25, 0.35, 0.25]),
        "timestamp": rng.randint(1_000_000_000, 1_600_000_000, N_RATINGS),
    }).drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)
    return df


@pytest.fixture(scope="session")
def prepared(ratings_df):
    """Процессор с маппингами, разбиением и train-матрицей."""
    proc = RecommenderDataProcessor(data_dir="data")
    df = proc.create_mappings(ratings_df.copy())
    train, test = proc.split_data(df, test_size=0.2)
    matrix = proc.create_user_item_matrix(train)
    return proc, df, train, test, matrix
