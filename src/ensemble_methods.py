import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from scipy.optimize import minimize
import time

class RecommenderEnsembleTrainer:
    def __init__(self):
        self.ensemble_models = {}
        
    def simple_average_ensemble(self, train_data, test_data, model_predictions):
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
    
    def weighted_average_ensemble(self, train_data, test_data, model_predictions, trained_models):
        print("Обучение Weighted Average Ensemble...")
        start_time = time.time()
        
        weights = []
        for model_name in model_predictions.keys():
            rmse = trained_models[model_name]['rmse']
            weight = 1.0 / (rmse + 0.001)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        predictions_array = np.array(list(model_predictions.values()))
        ensemble_predictions = np.average(predictions_array, axis=0, weights=weights)
        
        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], ensemble_predictions))
        mae = mean_absolute_error(test_data['rating'], ensemble_predictions)
        
        print(f"Веса моделей: {dict(zip(model_predictions.keys(), weights))}")
        
        return {
            'name': 'Weighted Average Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': ensemble_predictions,
            'weights': weights
        }
    
    def stacking_ensemble(self, train_data, test_data, model_predictions):
        print("Обучение Stacking Ensemble...")
        start_time = time.time()

        X_stack = np.array(list(model_predictions.values())).T
        y_stack = test_data['rating'].values

        # Исправление утечки данных: разделяем на validation и holdout
        n_samples = len(y_stack)
        n_val = int(n_samples * 0.3)

        # Перемешиваем индексы
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        holdout_indices = indices[n_val:]

        X_val = X_stack[val_indices]
        y_val = y_stack[val_indices]
        X_holdout = X_stack[holdout_indices]
        y_holdout = y_stack[holdout_indices]

        # Обучаем мета-модель на validation
        meta_model = LinearRegression()
        meta_model.fit(X_val, y_val)

        # Оцениваем на holdout (честная оценка)
        holdout_predictions = meta_model.predict(X_holdout)
        holdout_predictions = np.clip(holdout_predictions, 1, 5)

        # Предсказания для всех данных (для использования в других ансамблях)
        full_predictions = meta_model.predict(X_stack)
        full_predictions = np.clip(full_predictions, 1, 5)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(y_holdout, holdout_predictions))
        mae = mean_absolute_error(y_holdout, holdout_predictions)

        print(f"  Validation size: {n_val}, Holdout size: {len(holdout_indices)}")

        return {
            'name': 'Stacking Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': full_predictions,
            'meta_model': meta_model
        }
    
    def blending_ensemble(self, train_data, test_data, model_predictions):
        print("Обучение Blending Ensemble...")
        start_time = time.time()
        
        blend_ratio = 0.3
        n_blend = int(len(test_data) * blend_ratio)
        
        X_blend = np.array(list(model_predictions.values())).T[:n_blend]
        y_blend = test_data['rating'].values[:n_blend]
        
        meta_model = LinearRegression()
        meta_model.fit(X_blend, y_blend)
        
        X_holdout = np.array(list(model_predictions.values())).T[n_blend:]
        ensemble_predictions_holdout = meta_model.predict(X_holdout)
        ensemble_predictions_holdout = np.clip(ensemble_predictions_holdout, 1, 5)
        
        X_full = np.array(list(model_predictions.values())).T
        ensemble_predictions_full = meta_model.predict(X_full)
        ensemble_predictions_full = np.clip(ensemble_predictions_full, 1, 5)
        
        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'].values[n_blend:], ensemble_predictions_holdout))
        mae = mean_absolute_error(test_data['rating'].values[n_blend:], ensemble_predictions_holdout)
        
        return {
            'name': 'Blending Ensemble',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': ensemble_predictions_full
        }
    
    def voting_ensemble(self, train_data, test_data, model_predictions):
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
    
    def bagging_ensemble(self, train_data, test_data, user_item_matrix):
        print("Обучение Bagging Ensemble...")
        start_time = time.time()
        
        n_models = 5
        predictions_list = []
        
        for i in range(n_models):
            sample_indices = np.random.choice(len(train_data), size=len(train_data), replace=True)
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
    
    def train_ensemble_models(self, train_data, test_data, user_item_matrix, trained_models):
        results = {}
        model_predictions = {}
        
        for name, model in trained_models.items():
            model_predictions[name] = model['predictions']
        
        ensemble_methods = [
            lambda: self.simple_average_ensemble(train_data, test_data, model_predictions),
            lambda: self.weighted_average_ensemble(train_data, test_data, model_predictions, trained_models),
            lambda: self.stacking_ensemble(train_data, test_data, model_predictions),
            lambda: self.blending_ensemble(train_data, test_data, model_predictions),
            lambda: self.voting_ensemble(train_data, test_data, model_predictions),
            lambda: self.bagging_ensemble(train_data, test_data, user_item_matrix)
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
    
    def optimize_ensemble_weights(self, train_data, test_data, model_predictions, trained_models):
        """
        Оптимизация весов ансамбля с использованием scipy.optimize.minimize
        для минимизации RMSE на валидационной выборке
        """
        print("Оптимизация весов ансамбля...")
        start_time = time.time()

        X_stack = np.array(list(model_predictions.values())).T
        y_true = test_data['rating'].values
        n_models = len(model_predictions)
        model_names = list(model_predictions.keys())

        # Разделяем на validation и holdout для честной оценки
        n_samples = len(y_true)
        n_val = int(n_samples * 0.3)

        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        holdout_indices = indices[n_val:]

        X_val = X_stack[val_indices]
        y_val = y_true[val_indices]
        X_holdout = X_stack[holdout_indices]
        y_holdout = y_true[holdout_indices]

        # Целевая функция - RMSE на валидационной выборке
        def objective(weights):
            weights = np.abs(weights)  # Веса должны быть положительными
            weights = weights / weights.sum()  # Нормализуем
            predictions = np.dot(X_val, weights)
            predictions = np.clip(predictions, 1, 5)
            return np.sqrt(mean_squared_error(y_val, predictions))

        # Начальные веса - обратные RMSE моделей
        initial_weights = []
        for name in model_names:
            rmse = trained_models[name]['rmse']
            initial_weights.append(1.0 / (rmse + 0.001))
        initial_weights = np.array(initial_weights)
        initial_weights = initial_weights / initial_weights.sum()

        # Ограничения: веса >= 0 и сумма = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        # Оптимизация
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

        # Оцениваем на holdout (честная оценка)
        holdout_predictions = np.dot(X_holdout, optimal_weights)
        holdout_predictions = np.clip(holdout_predictions, 1, 5)

        # Полные предсказания для всех данных
        full_predictions = np.dot(X_stack, optimal_weights)
        full_predictions = np.clip(full_predictions, 1, 5)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(y_holdout, holdout_predictions))
        mae = mean_absolute_error(y_holdout, holdout_predictions)

        print("Оптимальные веса моделей:")
        for name, weight in zip(model_names, optimal_weights):
            print(f"  {name}: {weight:.4f}")

        return {
            'name': 'Optimized Ensemble Weights',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': full_predictions,
            'optimal_weights': dict(zip(model_names, optimal_weights))
        }

    def optimize_hyperparameters(self, train_data, test_data, user_item_matrix):
        print("Оптимизация гиперпараметров...")
        results = {}

        # Сначала оптимизируем веса ансамбля если есть обученные модели
        if self.ensemble_models:
            try:
                # Собираем предсказания базовых моделей из ensemble_models
                # Нам нужны предсказания от родительского trainer'а
                pass  # Будет вызвано отдельно из main
            except Exception as e:
                print(f"Ошибка в оптимизации весов: {e}")

        try:
            start_time = time.time()

            features = train_data[['user_idx', 'item_idx']].values
            targets = train_data['rating'].values
            test_features = test_data[['user_idx', 'item_idx']].values

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