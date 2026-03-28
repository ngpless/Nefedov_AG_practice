import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import time
import traceback

class RecommenderModelTrainer:
    def __init__(self):
        self.trained_models = {}
        self.model_predictions = {}
        self.global_mean = None
        
    def mean_baseline(self, train_data, test_data):
        print("Обучение Mean Baseline...")
        start_time = time.time()
        
        global_mean = train_data['rating'].mean()
        user_means = train_data.groupby('user_id')['rating'].mean()
        item_means = train_data.groupby('item_id')['rating'].mean()
        
        predictions = []
        for _, row in test_data.iterrows():
            user_mean = user_means.get(row['user_id'], global_mean)
            item_mean = item_means.get(row['item_id'], global_mean)
            pred = (user_mean + item_mean) / 2
            predictions.append(pred)
        
        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)
        
        return {
            'name': 'Mean Baseline',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': predictions
        }
    
    def user_based_cf(self, train_data, test_data, user_item_matrix):
        print("Обучение User-Based Collaborative Filtering...")
        start_time = time.time()

        n_neighbors = 25
        global_mean = train_data['rating'].mean()

        # Преобразуем в dense для быстрых вычислений
        user_item_dense = user_item_matrix.toarray()

        # Вычисляем матрицу сходства пользователей один раз
        user_similarity = cosine_similarity(user_item_dense)
        np.fill_diagonal(user_similarity, 0)  # Убираем самосходство

        # Получаем индексы тестовых данных
        test_user_idx = test_data['user_idx'].values
        test_item_idx = test_data['item_idx'].values

        predictions = []

        # Для каждого уникального пользователя находим k ближайших соседей
        unique_users = np.unique(test_user_idx)
        user_neighbors = {}

        for user_idx in unique_users:
            sim_scores = user_similarity[user_idx]
            top_neighbors = np.argsort(sim_scores)[-n_neighbors:][::-1]
            user_neighbors[user_idx] = (top_neighbors, sim_scores[top_neighbors])

        # Векторизованное предсказание
        for i in range(len(test_data)):
            user_idx = test_user_idx[i]
            item_idx = test_item_idx[i]

            neighbors, similarities = user_neighbors[user_idx]
            neighbor_ratings = user_item_dense[neighbors, item_idx]

            # Только соседи, которые оценили этот товар
            mask = neighbor_ratings > 0
            if mask.sum() > 0:
                weighted_sum = np.sum(neighbor_ratings[mask] * similarities[mask])
                sim_sum = np.sum(similarities[mask])
                pred = weighted_sum / sim_sum if sim_sum > 0 else global_mean
            else:
                pred = global_mean

            predictions.append(pred)

        predictions = np.array(predictions)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)

        return {
            'name': 'User-Based CF',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': predictions
        }
    
    def item_based_cf(self, train_data, test_data, user_item_matrix):
        print("Обучение Item-Based Collaborative Filtering...")
        start_time = time.time()

        n_neighbors = 25
        global_mean = train_data['rating'].mean()

        # Преобразуем в dense для быстрых вычислений
        user_item_dense = user_item_matrix.toarray()

        # Вычисляем матрицу сходства товаров один раз
        item_similarity = cosine_similarity(user_item_dense.T)
        np.fill_diagonal(item_similarity, 0)  # Убираем самосходство

        # Получаем индексы тестовых данных
        test_user_idx = test_data['user_idx'].values
        test_item_idx = test_data['item_idx'].values

        # Для каждого уникального товара находим k ближайших соседей
        unique_items = np.unique(test_item_idx)
        item_neighbors = {}

        for item_idx in unique_items:
            sim_scores = item_similarity[item_idx]
            top_neighbors = np.argsort(sim_scores)[-n_neighbors:][::-1]
            item_neighbors[item_idx] = (top_neighbors, sim_scores[top_neighbors])

        predictions = []

        # Векторизованное предсказание
        for i in range(len(test_data)):
            user_idx = test_user_idx[i]
            item_idx = test_item_idx[i]

            neighbors, similarities = item_neighbors[item_idx]
            user_ratings = user_item_dense[user_idx, neighbors]

            # Только товары, которые пользователь оценил
            mask = user_ratings > 0
            if mask.sum() > 0:
                weighted_sum = np.sum(user_ratings[mask] * similarities[mask])
                sim_sum = np.sum(similarities[mask])
                pred = weighted_sum / sim_sum if sim_sum > 0 else global_mean
            else:
                pred = global_mean

            predictions.append(pred)

        predictions = np.array(predictions)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)

        return {
            'name': 'Item-Based CF',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': predictions
        }
    
    def matrix_factorization_svd(self, train_data, test_data, user_item_matrix):
        print("Обучение SVD Matrix Factorization...")
        start_time = time.time()

        # Преобразуем в dense матрицу
        user_item_dense = user_item_matrix.toarray().astype(float)

        # Вычисляем глобальное среднее (только по ненулевым элементам)
        mask = user_item_dense > 0
        global_mean = user_item_dense[mask].mean()
        self.global_mean = global_mean

        # Вычисляем смещения пользователей и товаров
        user_means = np.zeros(user_item_dense.shape[0])
        item_means = np.zeros(user_item_dense.shape[1])

        for u in range(user_item_dense.shape[0]):
            user_ratings = user_item_dense[u, mask[u]]
            if len(user_ratings) > 0:
                user_means[u] = user_ratings.mean() - global_mean

        for i in range(user_item_dense.shape[1]):
            item_ratings = user_item_dense[mask[:, i], i]
            if len(item_ratings) > 0:
                item_means[i] = item_ratings.mean() - global_mean

        # Центрируем матрицу (вычитаем среднее только из ненулевых элементов)
        centered_matrix = user_item_dense.copy()
        centered_matrix[mask] -= global_mean

        # Применяем SVD с большим количеством компонент
        n_components = min(100, min(centered_matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors = svd.fit_transform(centered_matrix)
        item_factors = svd.components_.T

        # Получаем индексы тестовых данных
        test_user_idx = test_data['user_idx'].values
        test_item_idx = test_data['item_idx'].values

        # Векторизованное предсказание
        predictions = np.array([
            np.dot(user_factors[u], item_factors[i]) + global_mean
            for u, i in zip(test_user_idx, test_item_idx)
        ])

        predictions = np.clip(predictions, 1, 5)

        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)

        return {
            'name': 'SVD Matrix Factorization',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': predictions
        }
    
    def nmf_model(self, train_data, test_data, user_item_matrix):
        print("Обучение NMF...")
        start_time = time.time()

        try:
            # Преобразуем разреженную матрицу в плотную
            dense_matrix = user_item_matrix.toarray().astype(float)
            global_mean = train_data['rating'].mean()

            # Создаём маску ненулевых элементов
            mask = dense_matrix > 0

            # Для NMF: заполняем пустые ячейки средним значением
            train_matrix = dense_matrix.copy()
            train_matrix[~mask] = global_mean

            # Применяем NMF
            n_components = min(30, min(train_matrix.shape) - 1)
            nmf = NMF(
                n_components=n_components,
                random_state=42,
                max_iter=500,
                init='nndsvda',
                solver='cd',  # Coordinate descent - более стабильный
                tol=1e-4
            )

            user_factors = nmf.fit_transform(train_matrix)
            item_factors = nmf.components_.T

            # Реконструируем матрицу
            reconstructed = np.dot(user_factors, item_factors.T)

            # Получаем индексы тестовых данных
            test_user_idx = test_data['user_idx'].values
            test_item_idx = test_data['item_idx'].values

            # Предсказания из реконструированной матрицы
            predictions = reconstructed[test_user_idx, test_item_idx]
            predictions = np.clip(predictions, 1, 5)

            training_time = time.time() - start_time
            rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
            mae = mean_absolute_error(test_data['rating'], predictions)

            print(f"NMF: успешно обучена, RMSE={rmse:.4f}")

            return {
                'name': 'NMF',
                'rmse': rmse,
                'mae': mae,
                'training_time': training_time,
                'predictions': predictions
            }

        except Exception as e:
            print(f"Ошибка в NMF: {str(e)}")
            print(traceback.format_exc())
            # Возвращаем fallback с глобальным средним
            global_mean = train_data['rating'].mean()
            predictions = np.full(len(test_data), global_mean)
            training_time = time.time() - start_time

            return {
                'name': 'NMF',
                'rmse': np.sqrt(mean_squared_error(test_data['rating'], predictions)),
                'mae': mean_absolute_error(test_data['rating'], predictions),
                'training_time': training_time,
                'predictions': predictions
            }
    
    def ridge_regression(self, train_data, test_data):
        print("Обучение Ridge Regression...")
        start_time = time.time()
        
        train_features = train_data[['user_idx', 'item_idx']].values
        train_targets = train_data['rating'].values
        test_features = test_data[['user_idx', 'item_idx']].values
        
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(train_features, train_targets)
        
        predictions = ridge.predict(test_features)
        predictions = np.clip(predictions, 1, 5)
        
        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)
        
        return {
            'name': 'Ridge Regression',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': predictions
        }
    
    def popularity_based(self, train_data, test_data):
        print("Обучение Popularity-Based...")
        start_time = time.time()
        
        item_popularity = train_data.groupby('item_id')['rating'].mean()
        global_mean = train_data['rating'].mean()
        
        predictions = []
        for _, row in test_data.iterrows():
            item_id = row['item_id']
            pred = item_popularity.get(item_id, global_mean)
            predictions.append(pred)
        
        training_time = time.time() - start_time
        rmse = np.sqrt(mean_squared_error(test_data['rating'], predictions))
        mae = mean_absolute_error(test_data['rating'], predictions)
        
        return {
            'name': 'Popularity-Based',
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'predictions': predictions
        }
    
    def train_base_models(self, train_data, test_data, user_item_matrix):
        results = {}
        
        models = [
            lambda: self.mean_baseline(train_data, test_data),
            lambda: self.user_based_cf(train_data, test_data, user_item_matrix),
            lambda: self.item_based_cf(train_data, test_data, user_item_matrix),
            lambda: self.matrix_factorization_svd(train_data, test_data, user_item_matrix),
            lambda: self.nmf_model(train_data, test_data, user_item_matrix),
            lambda: self.ridge_regression(train_data, test_data),
            lambda: self.popularity_based(train_data, test_data)
        ]
        
        for model_func in models:
            try:
                result = model_func()
                model_name = result['name']
                results[model_name] = result
                self.trained_models[model_name] = result
                self.model_predictions[model_name] = result['predictions']
                
                print(f"{model_name}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, Время={result['training_time']:.2f}с")
                
            except Exception as e:
                print(f"Ошибка в модели: {e}")
        
        return results