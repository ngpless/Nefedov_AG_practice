# -*- coding: utf-8 -*-
"""Тесты построения маппингов идентификаторов в плотные индексы."""


def test_indices_are_contiguous(prepared):
    proc, df, *_ = prepared
    users = sorted(proc.user_mapping.values())
    items = sorted(proc.item_mapping.values())
    assert users == list(range(len(users)))
    assert items == list(range(len(items)))


def test_mapping_is_bijective(prepared):
    proc, df, *_ = prepared
    assert len(set(proc.user_mapping.values())) == len(proc.user_mapping)
    assert len(set(proc.item_mapping.values())) == len(proc.item_mapping)


def test_dataframe_columns_added(prepared):
    _, df, *_ = prepared
    assert "user_idx" in df.columns
    assert "item_idx" in df.columns
    assert df["user_idx"].min() >= 0
    assert df["item_idx"].min() >= 0
