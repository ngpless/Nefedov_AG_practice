"""
Ансамблевые методы рекомендательной системы.

Схема без утечки данных: обучающая выборка делится на core/validation,
базовые модели переобучаются на core, мета-модели (стекинг, блендинг,
оптимизация весов) подбираются по validation, а итоговая оценка всех
ансамблей выполняется на тестовой выборке, метки которой ни на одном
этапе не участвуют в обучении.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import time

from config import RANDOM_SEED, VALIDATION_SIZE


class RecommenderEnsembleTrainer:
    def __init__(self):
        self.ensemble_models = {}
        # Контекст holdout-схемы: заполняется в train_ensemble_models
        self._val_y = None
        self._val_preds = None
        self._test_preds_core = None
        self._val_rmse = None
        self._model_names = None

    # ------------------------------------------------------------------
    # Подготовка honest-предсказаний базовых моделей
    # ------------------------------------------------------------------
    def _build_holdout_predictions(self, train_core, val_data, test_data,
                                   n_users, n_items):
        """
        Обучает базовые модели на train_core и возвращает их предсказания
        на валидационной и тестовой выборках. Тестовые метки нигде не
        используются для обучения.
        """
        from models import RecommenderModelTrainer

        matrix_core = csr_matrix(
            (train_core['rating'].values,
             (train_core['user_idx'].values, train_core['item_idx'].values)),
            shape=(n_users, n_items)
        )

        print("\n  Обучение базовых моделей на core-выборке (для мета-моделей)...")
        val_trainer = RecommenderModelTrainer()
        val_results = val_trainer.train_base_models(train_core, val_data, matrix_core)

        test_trainer = RecommenderModelTrainer()
        test_results = test_trainer.train_base_models(train_core, test_data, matrix_core)

        val_preds = {n: r['predictions'] for n, r in val_results.items()}
        test_preds = {n: r['predictions'] for n, r in test_results.items()}
        val_rmse = {n: r['rmse'] for n, r in val_results.items()}

        return val_preds, test_preds, val_rmse

    # ------------------------------------------------------------------
    # Ансамбли без обучения на метках (используют полные test-предсказания)
    # ------------------------------------------------------------------
    def simple_average_ensemble(self, test_data, model_predictions):
        print("Обучение Simple Average Ensemble...")
        start_time = time.time()

        predictions_array = np.array(list(model_predictions.values()))
        ensemble_predictions = np.mean(predictions_array, axis=0)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], ensemble_predictions))
        mae = mean_absolute_error(test_data['rating'], ensemble_predictions)

        return {
            'name': 'Simple Average Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': ensemble_predictions
        }

    def weighted_average_ensemble(self, test_data, model_predictions, val_rmse):
        """Веса моделей вычисляются по RMSE на валидации, а не на тесте."""
        print("Обучение Weighted Average Ensemble...")
        start_time = time.time()

        weights = []
        for model_name in model_predictions.keys():
            rmse = val_rmse[model_name]
            weight = 1.0 / (rmse + 0.001)
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()

        predictions_array = np.array(list(model_predictions.values()))
        ensemble_predictions = np.average(predictions_array, axis=0, weights=weights)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], ensemble_predictions))
        mae = mean_absolute_error(test_data['rating'], ensemble_predictions)

        print(f"Веса моделей (по валидации): {dict(zip(model_predictions.keys(), np.round(weights, 4)))}")

        return {
            'name': 'Weighted Average Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': ensemble_predictions,
            'weights': weights
        }

    def voting_ensemble(self, test_data, model_predictions):
        print("Обучение Voting Ensemble...")
        start_time = time.time()

        predictions_array = np.array(list(model_predictions.values()))
        rounded_predictions = np.round(predictions_array)

        ensemble_predictions = []
        for i in range(predictions_array.shape[1]):
            votes = rounded_predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            most_voted = unique[np.argmax(counts)]
            ensemble_predictions.append(most_voted)

        ensemble_predictions = np.array(ensemble_predictions)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], ensemble_predictions))
        mae = mean_absolute_error(test_data['rating'], ensemble_predictions)

        return {
            'name': 'Voting Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': ensemble_predictions
        }

    def bagging_ensemble(self, train_data, test_data):
        print("Обучение Bagging Ensemble...")
        start_time = time.time()

        n_models = 5
        predictions_list = []
        rng = np.random.RandomState(RANDOM_SEED)

        for i in range(n_models):
            sample_indices = rng.choice(len(train_data), size=len(train_data), replace=True)
            train_sample = train_data.iloc[sample_indices]

            user_means = train_sample.groupby('user_id')['rating'].mean()
            item_means = train_sample.groupby('item_id')['rating'].mean()
            global_mean = train_sample['rating'].mean()

            sample_predictions = []
            for _, row in test_data.iterrows():
                user_mean = user_means.get(row['user_id'], global_mean)
                item_mean = item_means.get(row['item_id'], global_mean)
                pred = (user_mean + item_mean) / 2
                sample_predictions.append(pred)

            predictions_list.append(sample_predictions)

        ensemble_predictions = np.mean(predictions_list, axis=0)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], ensemble_predictions))
        mae = mean_absolute_error(test_data['rating'], ensemble_predictions)

        return {
            'name': 'Bagging Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': ensemble_predictions
        }

    # ------------------------------------------------------------------
    # Ансамбли с мета-обучением: fit на валидации, оценка на тесте
    # ------------------------------------------------------------------
    def stacking_ensemble(self, val_y, val_preds, test_y, test_preds):
        print("Обучение Stacking Ensemble...")
        start_time = time.time()

        names = self._model_names
        X_val = np.array([val_preds[n] for n in names]).T
        X_test = np.array([test_preds[n] for n in names]).T

        meta_model = LinearRegression()
        meta_model.fit(X_val, val_y)

        test_predictions = np.clip(meta_model.predict(X_test), 1, 5)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
        mae = mean_absolute_error(test_y, test_predictions)

        print(f"  Мета-модель обучена на валидации ({len(val_y)} примеров), оценка на тесте ({len(test_y)})")

        return {
            'name': 'Stacking Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': test_predictions,
            'meta_model': meta_model
        }

    def blending_ensemble(self, val_y, val_preds, test_y, test_preds):
        print("Обучение Blending Ensemble...")
        start_time = time.time()

        names = self._model_names
        X_val = np.array([val_preds[n] for n in names]).T
        X_test = np.array([test_preds[n] for n in names]).T

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X_val, val_y)

        test_predictions = np.clip(meta_model.predict(X_test), 1, 5)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
        mae = mean_absolute_error(test_y, test_predictions)

        return {
            'name': 'Blending Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': test_predictions
        }

    def optimize_ensemble_weights(self, test_data):
        """
        Оптимизация весов ансамбля (SLSQP): целевая функция — RMSE на
        валидационной выборке, итоговая оценка — на тестовой.
        Требует предварительного вызова train_ensemble_models.
        """
        print("Оптимизация весов ансамбля...")
        start_time = time.time()

        names = self._model_names
        X_val = np.array([self._val_preds[n] for n in names]).T
        y_val = self._val_y
        X_test = np.array([self._test_preds_core[n] for n in names]).T
        y_test = test_data['rating'].values
        n_models = len(names)

        def objective(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()
            predictions = np.clip(np.dot(X_val, weights), 1, 5)
            return np.sqrt(mean_squared_error(y_val, predictions))

        initial_weights = np.array([1.0 / (self._val_rmse[n] + 0.001) for n in names])
        initial_weights = initial_weights / initial_weights.sum()

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / optimal_weights.sum()

        test_predictions = np.clip(np.dot(X_test, optimal_weights), 1, 5)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        mae = mean_absolute_error(y_test, test_predictions)

        print("Оптимальные веса моделей (подобраны по валидации):")
        for name, weight in zip(names, optimal_weights):
            print(f"  {name}: {weight:.4f}")

        return {
            'name': 'Optimized Ensemble Weights',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': test_predictions,
            'optimal_weights': dict(zip(names, optimal_weights))
        }

    # ------------------------------------------------------------------
    # Общий запуск
    # ------------------------------------------------------------------
    def train_ensemble_models(self, train_data, test_data, user_item_matrix,
                              trained_models, n_users, n_items):
        results = {}

        # Предсказания базовых моделей, обученных на ПОЛНОМ train (для
        # ансамблей, не использующих метки: среднее, голосование)
        model_predictions = {name: model['predictions']
                             for name, model in trained_models.items()}

        # Honest-схема для мета-моделей: train -> core + validation
        train_core, val_data = train_test_split(
            train_data, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED,
            stratify=train_data['rating']
        )
        val_preds, test_preds_core, val_rmse = self._build_holdout_predictions(
            train_core, val_data, test_data, n_users, n_items
        )

        # Сохраняем контекст для optimize_ensemble_weights
        self._val_y = val_data['rating'].values
        self._val_preds = val_preds
        self._test_preds_core = test_preds_core
        self._val_rmse = val_rmse
        self._model_names = list(val_preds.keys())

        test_y = test_data['rating'].values

        ensemble_methods = [
            lambda: self.simple_average_ensemble(test_data, model_predictions),
            lambda: self.weighted_average_ensemble(test_data, model_predictions, val_rmse),
            lambda: self.stacking_ensemble(self._val_y, val_preds, test_y, test_preds_core),
            lambda: self.blending_ensemble(self._val_y, val_preds, test_y, test_preds_core),
            lambda: self.voting_ensemble(test_data, model_predictions),
            lambda: self.bagging_ensemble(train_data, test_data)
        ]

        for ensemble_func in ensemble_methods:
            try:
                result = ensemble_func()
                model_name = result['name']
                results[model_name] = result
                self.ensemble_models[model_name] = result

                print(f"{model_name}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, Время={result['training_time']:.2f}с")

            except Exception as e:
                print(f"Ошибка в ансамбле: {e}")

        return results

    def optimize_hyperparameters(self, train_data, test_data, user_item_matrix):
        print("Оптимизация гиперпараметров...")
        results = {}

        features = train_data[['user_idx', 'item_idx']].values
        targets = train_data['rating'].values
        test_features = test_data[['user_idx', 'item_idx']].values

        try:
            start_time = time.time()

            rf_params = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [5, 10]
            }

            rf = RandomForestRegressor(random_state=42)
            rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            rf_grid.fit(features, targets)

            rf_predictions = rf_grid.predict(test_features)
            rf_predictions = np.clip(rf_predictions, 1, 5)

            rf_rmse = np.sqrt(mean_squared_error(test_data['rating'], rf_predictions))
            rf_mae = mean_absolute_error(test_data['rating'], rf_predictions)

            results['Optimized Random Forest'] = {
                'name': 'Optimized Random Forest',
                'rmse': rf_rmse,
                'mae': rf_mae,
                'training_time': time.time() - start_time,
                'predictions': rf_predictions,
                'best_params': rf_grid.best_params_
            }

            print(f"Optimized Random Forest: RMSE={rf_rmse:.4f}, MAE={rf_mae:.4f}")
            print(f"Лучшие параметры: {rf_grid.best_params_}")

        except Exception as e:
            print(f"Ошибка в оптимизации: {e}")

        try:
            start_time = time.time()

            gb_params = {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            }

            gb = GradientBoostingRegressor(random_state=42)
            gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            gb_grid.fit(features, targets)

            gb_predictions = gb_grid.predict(test_features)
            gb_predictions = np.clip(gb_predictions, 1, 5)

            gb_rmse = np.sqrt(mean_squared_error(test_data['rating'], gb_predictions))
            gb_mae = mean_absolute_error(test_data['rating'], gb_predictions)

            results['Optimized Gradient Boosting'] = {
                'name': 'Optimized Gradient Boosting',
                'rmse': gb_rmse,
                'mae': gb_mae,
                'training_time': time.time() - start_time,
                'predictions': gb_predictions,
                'best_params': gb_grid.best_params_
            }

            print(f"Optimized Gradient Boosting: RMSE={gb_rmse:.4f}, MAE={gb_mae:.4f}")
            print(f"Лучшие параметры: {gb_grid.best_params_}")

        except Exception as e:
            print(f"Ошибка в оптимизации Gradient Boosting: {e}")

        return results
