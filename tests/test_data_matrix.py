# -*- coding: utf-8 -*-
"""Тесты user-item матрицы: строится только по обучающей выборке."""


def test_matrix_shape(prepared):
    proc, _, _, _, matrix = prepared
    assert matrix.shape == (len(proc.user_mapping), len(proc.item_mapping))


def test_matrix_built_from_train_only(prepared):
    _, _, train, test, matrix = prepared
    # число ненулевых элементов равно числу обучающих оценок
    assert matrix.nnz == len(train)


def test_matrix_values_match_train(prepared):
    _, _, train, _, matrix = prepared
    sample = train.sample(20, random_state=0)
    for _, row in sample.iterrows():
        assert matrix[int(row["user_idx"]), int(row["item_idx"])] == row["rating"]


def test_test_ratings_absent_from_matrix(prepared):
    """Тестовые пары не должны попадать в матрицу (защита от утечки)."""
    _, _, train, test, matrix = prepared
    train_pairs = set(zip(train["user_idx"], train["item_idx"]))
    leaked = 0
    for _, row in test.head(100).iterrows():
        pair = (row["user_idx"], row["item_idx"])
        if pair not in train_pairs and matrix[int(pair[0]), int(pair[1])] != 0:
            leaked += 1
    assert leaked == 0
