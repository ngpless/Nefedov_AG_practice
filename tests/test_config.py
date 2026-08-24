# -*- coding: utf-8 -*-
"""Проверка констант конфигурации проекта."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config


def test_seed_is_int():
    assert isinstance(config.RANDOM_SEED, int)


def test_split_fractions_valid():
    assert 0 < config.TEST_SIZE < 1
    assert 0 < config.VALIDATION_SIZE < 1
    # суммарно train_core остаётся больше половины данных
    assert (1 - config.TEST_SIZE) * (1 - config.VALIDATION_SIZE) > 0.5


def test_training_defaults_positive():
    assert config.DEFAULT_EPOCHS > 0
    assert config.DEFAULT_BATCH_SIZE > 0
    assert config.DEFAULT_LEARNING_RATE > 0
    assert config.EARLY_STOPPING_PATIENCE > 0


def test_rating_bounds():
    assert config.RATING_MIN < config.RATING_MAX
