"""
Утилиты для рекомендательной системы

Содержит:
1. Функции для создания директорий и сохранения результатов
2. Метрики оценки качества рекомендаций (Precision@K, Recall@K, NDCG@K, Hit Rate)
3. Функции генерации рекомендаций
4. Вспомогательные функции

Автор: Нефедов Алексей Геннадьевич
Дата: 2025
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime


def create_directories():
    """
    Создание структуры директорий проекта

    Создаёт:
    - data/raw - сырые данные
    - data/processed - обработанные данные
    - results/figures - графики и визуализации
    - results/models - сохранённые модели
    - results/experiments - результаты экспериментов
    - notebooks - Jupyter notebooks
    """
    directories = [
        'data/raw',
        'data/processed',
        'results/figures',
        'results/models',
        'results/experiments',
        'results/experiments/figures',
        'notebooks'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Созданы необходимые директории проекта")


def save_recommendation_results(results, dataset_info, results_dir='results'):
    """
    Сохранение результатов экспериментов

    Args:
        results: словарь с результатами моделей
        dataset_info: информация о датасете
        results_dir: директория для сохранения
    """
    print("Сохранение результатов...")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)

    # Сводка результатов
    results_summary = {}
    for name, result in results.items():
        results_summary[name] = {
            'rmse': float(result['rmse']),
            'mae': float(result['mae']),
            'training_time': float(result['training_time'])
        }

        # Добавляем дополнительные метрики если есть
        if 'n_parameters' in result:
            results_summary[name]['n_parameters'] = int(result['n_parameters'])
        if 'epochs_trained' in result:
            results_summary[name]['epochs_trained'] = int(result['epochs_trained'])

    with open(os.path.join(results_dir, 'results_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # Информация о датасете
    dataset_info_serializable = {}
    for key, value in dataset_info.items():
        if isinstance(value, (int, float, str, bool)):
            dataset_info_serializable[key] = value
        elif isinstance(value, (list, dict)):
            dataset_info_serializable[key] = value
        elif isinstance(value, np.integer):
            dataset_info_serializable[key] = int(value)
        elif isinstance(value, np.floating):
            dataset_info_serializable[key] = float(value)
        else:
            dataset_info_serializable[key] = str(value)

    with open(os.path.join(results_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_info_serializable, f, indent=2, ensure_ascii=False)

    # Сохранение отдельных моделей
    for name, result in results.items():
        safe_name = name.replace(' ', '_').replace('-', '_').replace('&', 'and')

        model_data = {
            'name': result['name'] if 'name' in result else name,
            'rmse': float(result['rmse']),
            'mae': float(result['mae']),
            'training_time': float(result['training_time']),
            'timestamp': datetime.now().isoformat()
        }

        # Добавляем предсказания
        if 'predictions' in result:
            preds = result['predictions']
            if hasattr(preds, 'tolist'):
                model_data['predictions'] = preds.tolist()
            elif isinstance(preds, list):
                model_data['predictions'] = preds

        # Добавляем дополнительные поля
        for field in ['n_parameters', 'epochs_trained', 'architecture', 'best_params']:
            if field in result:
                model_data[field] = result[field]

        model_path = os.path.join(results_dir, 'models', f'{safe_name}.json')
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

    # Лучшая модель
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"Лучшая модель по RMSE: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")

    best_model_data = {
        'name': best_model[0],
        'rmse': float(best_model[1]['rmse']),
        'mae': float(best_model[1]['mae']),
        'training_time': float(best_model[1]['training_time']),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(results_dir, 'best_model.json'), 'w', encoding='utf-8') as f:
        json.dump(best_model_data, f, indent=2, ensure_ascii=False)

    print(f"Результаты сохранены в папке {results_dir}/")


def load_results(results_dir='results'):
    """
    Загрузка сохранённых результатов

    Args:
        results_dir: директория с результатами

    Returns:
        dict с результатами или None если файл не найден
    """
    try:
        with open(os.path.join(results_dir, 'results_summary.json'), 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print("Файл результатов не найден")
        return None


# ============================================================================
# МЕТРИКИ КАЧЕСТВА РЕКОМЕНДАЦИЙ
# ============================================================================

def calculate_precision_at_k(true_items, recommended_items, k):
    """
    Precision@K - доля релевантных рекомендаций в топ-K

    Формула: Precision@K = |{релевантные} ∩ {рекомендованные}| / K

    Args:
        true_items: список истинных релевантных товаров
        recommended_items: список рекомендованных товаров
        k: количество топ рекомендаций

    Returns:
        float: значение Precision@K
    """
    if k == 0:
        return 0.0

    recommended_k = recommended_items[:k]
    relevant_recommended = len(set(true_items) & set(recommended_k))

    return relevant_recommended / k


def calculate_recall_at_k(true_items, recommended_items, k):
    """
    Recall@K - доля найденных релевантных товаров

    Формула: Recall@K = |{релевантные} ∩ {рекомендованные}| / |{релевантные}|

    Args:
        true_items: список истинных релевантных товаров
        recommended_items: список рекомендованных товаров
        k: количество топ рекомендаций

    Returns:
        float: значение Recall@K
    """
    if len(true_items) == 0:
        return 0.0

    recommended_k = recommended_items[:k]
    relevant_recommended = len(set(true_items) & set(recommended_k))

    return relevant_recommended / len(true_items)


def calculate_ndcg_at_k(true_items, recommended_items, k):
    """
    NDCG@K - Normalized Discounted Cumulative Gain

    Учитывает позицию релевантных товаров в списке рекомендаций.
    Товары в начале списка вносят больший вклад.

    Формула:
    DCG@K = Σ(rel_i / log2(i+1)) для i от 1 до K
    NDCG@K = DCG@K / IDCG@K

    где IDCG - идеальный DCG (когда все релевантные товары в начале)

    Args:
        true_items: список истинных релевантных товаров
        recommended_items: список рекомендованных товаров
        k: количество топ рекомендаций

    Returns:
        float: значение NDCG@K в диапазоне [0, 1]
    """
    def dcg_at_k(scores, k):
        """Вычисление DCG"""
        scores = np.array(scores)[:k]
        if len(scores) == 0:
            return 0.0
        return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

    if k == 0 or len(true_items) == 0:
        return 0.0

    recommended_k = recommended_items[:k]
    relevance_scores = [1 if item in true_items else 0 for item in recommended_k]

    dcg = dcg_at_k(relevance_scores, k)

    # Идеальный DCG (все релевантные в начале)
    ideal_scores = [1] * min(len(true_items), k)
    idcg = dcg_at_k(ideal_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_hit_rate_at_k(true_items, recommended_items, k):
    """
    Hit Rate@K - была ли хотя бы одна релевантная рекомендация в топ-K

    Формула: HR@K = 1 если |{релевантные} ∩ {рекомендованные}| > 0, иначе 0

    Args:
        true_items: список истинных релевантных товаров
        recommended_items: список рекомендованных товаров
        k: количество топ рекомендаций

    Returns:
        float: 1.0 если есть попадание, 0.0 иначе
    """
    if k == 0 or len(true_items) == 0:
        return 0.0

    recommended_k = set(recommended_items[:k])
    hit = len(set(true_items) & recommended_k) > 0

    return 1.0 if hit else 0.0


def calculate_mrr(true_items, recommended_items):
    """
    MRR - Mean Reciprocal Rank

    Среднее обратное ранжирование - средняя величина обратная позиции
    первого релевантного товара.

    Формула: MRR = 1/rank первого релевантного товара

    Args:
        true_items: список истинных релевантных товаров
        recommended_items: список рекомендованных товаров

    Returns:
        float: значение MRR (0 если нет попаданий)
    """
    true_set = set(true_items)

    for i, item in enumerate(recommended_items):
        if item in true_set:
            return 1.0 / (i + 1)

    return 0.0


def calculate_coverage(all_items, all_recommendations):
    """
    Coverage - покрытие каталога товаров

    Доля уникальных товаров, которые были рекомендованы хотя бы одному пользователю.

    Формула: Coverage = |уникальные рекомендации| / |все товары|

    Args:
        all_items: множество всех товаров в каталоге
        all_recommendations: список всех рекомендаций для всех пользователей

    Returns:
        float: покрытие в диапазоне [0, 1]
    """
    if len(all_items) == 0:
        return 0.0

    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)

    return len(unique_recommended) / len(all_items)


def evaluate_recommendations(test_data, model, user_item_matrix, k_values=[5, 10, 20], n_users=100):
    """
    Полная оценка качества рекомендаций

    Args:
        test_data: тестовые данные
        model: обученная модель с методом predict
        user_item_matrix: матрица user-item
        k_values: список значений K для метрик
        n_users: количество пользователей для оценки

    Returns:
        dict: словарь со всеми метриками
    """
    print(f"Оценка рекомендаций для k = {k_values}")

    metrics = {}

    for k in k_values:
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        hit_rate_scores = []
        mrr_scores = []

        # Выбираем случайных пользователей
        unique_users = test_data['user_idx'].unique()
        sample_users = np.random.choice(unique_users, min(n_users, len(unique_users)), replace=False)

        for user_idx in sample_users:
            # Истинные товары пользователя (с высоким рейтингом)
            user_data = test_data[test_data['user_idx'] == user_idx]
            true_items = user_data[user_data['rating'] >= 4]['item_idx'].tolist()

            if len(true_items) == 0:
                continue

            # Генерируем рекомендации (товары которые пользователь не оценивал)
            user_rated = set(test_data[test_data['user_idx'] == user_idx]['item_idx'].tolist())
            all_items = set(range(user_item_matrix.shape[1]))
            candidate_items = list(all_items - user_rated)

            # Простая рекомендация по популярности если нет модели
            if len(candidate_items) == 0:
                continue

            # Случайный выбор для демонстрации (замените на реальные предсказания модели)
            np.random.shuffle(candidate_items)
            recommended_items = candidate_items[:k * 2]

            # Вычисляем метрики
            precision = calculate_precision_at_k(true_items, recommended_items, k)
            recall = calculate_recall_at_k(true_items, recommended_items, k)
            ndcg = calculate_ndcg_at_k(true_items, recommended_items, k)
            hit_rate = calculate_hit_rate_at_k(true_items, recommended_items, k)
            mrr = calculate_mrr(true_items, recommended_items)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            hit_rate_scores.append(hit_rate)
            mrr_scores.append(mrr)

        # Усреднение
        if precision_scores:
            metrics[f'Precision@{k}'] = np.mean(precision_scores)
            metrics[f'Recall@{k}'] = np.mean(recall_scores)
            metrics[f'NDCG@{k}'] = np.mean(ndcg_scores)
            metrics[f'HitRate@{k}'] = np.mean(hit_rate_scores)
            metrics[f'MRR@{k}'] = np.mean(mrr_scores)

    return metrics


def generate_recommendations_for_user(user_idx, model_predictions, n_items, n_recommendations=10):
    """
    Генерация рекомендаций для пользователя

    Args:
        user_idx: индекс пользователя
        model_predictions: предсказания модели или None
        n_items: общее количество товаров
        n_recommendations: количество рекомендаций

    Returns:
        list: список индексов рекомендованных товаров
    """
    if model_predictions is not None and hasattr(model_predictions, '__getitem__'):
        # Используем предсказания модели
        scores = model_predictions[user_idx] if len(model_predictions.shape) > 1 else model_predictions
        sorted_items = np.argsort(scores)[::-1]
        return sorted_items[:n_recommendations].tolist()
    else:
        # Случайные рекомендации
        all_items = list(range(n_items))
        np.random.shuffle(all_items)
        return all_items[:n_recommendations]


def print_metrics_table(metrics, title="Метрики качества рекомендаций"):
    """
    Красивый вывод таблицы метрик

    Args:
        metrics: словарь с метриками
        title: заголовок таблицы
    """
    print("\n" + "="*50)
    print(title)
    print("="*50)

    # Группируем по K
    k_values = set()
    metric_names = set()

    for key in metrics.keys():
        parts = key.split('@')
        if len(parts) == 2:
            metric_names.add(parts[0])
            k_values.add(int(parts[1]))

    k_values = sorted(k_values)
    metric_names = sorted(metric_names)

    # Заголовок
    header = f"{'Метрика':<15}"
    for k in k_values:
        header += f"{'@'+str(k):>10}"
    print(header)
    print("-"*50)

    # Строки
    for metric in metric_names:
        row = f"{metric:<15}"
        for k in k_values:
            key = f"{metric}@{k}"
            if key in metrics:
                row += f"{metrics[key]:>10.4f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    print("="*50)


def format_time(seconds):
    """Форматирование времени в читаемый вид"""
    if seconds < 60:
        return f"{seconds:.2f} сек"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} мин {int(secs)} сек"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} ч {int(minutes)} мин"


def get_model_summary(results):
    """
    Создание сводки по моделям

    Args:
        results: словарь с результатами

    Returns:
        pandas.DataFrame: сводная таблица
    """
    data = []
    for name, result in results.items():
        row = {
            'Модель': name,
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'Время': result['training_time']
        }
        if 'n_parameters' in result:
            row['Параметры'] = result['n_parameters']
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('RMSE')
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # Нумерация с 1

    return df
