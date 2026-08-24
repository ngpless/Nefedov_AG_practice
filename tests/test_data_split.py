# -*- coding: utf-8 -*-
"""Тесты разбиения на обучающую и тестовую выборки."""


def test_split_sizes(prepared):
    _, df, train, test, _ = prepared
    assert len(train) + len(test) == len(df)
    share = len(test) / len(df)
    assert 0.18 < share < 0.22


def test_split_no_overlap(prepared):
    _, df, train, test, _ = prepared
    inter = set(train.index) & set(test.index)
    assert not inter


def test_stratification_preserves_distribution(prepared):
    _, df, train, test, _ = prepared
    for rating in sorted(df["rating"].unique()):
        p_train = (train["rating"] == rating).mean()
        p_test = (test["rating"] == rating).mean()
        assert abs(p_train - p_test) < 0.03
