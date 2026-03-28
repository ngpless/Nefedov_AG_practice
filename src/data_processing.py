"""
Модуль обработки данных для рекомендательной системы

Поддерживает:
1. Автоматическую загрузку датасета MovieLens 100K
2. Генерацию синтетических данных для тестирования
3. Предобработку и анализ данных
4. Создание user-item матриц

Автор: Нефедов Алексей Геннадьевич
Дата: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import os
import zipfile
import requests
from io import BytesIO
import time
import warnings
warnings.filterwarnings('ignore')


class RecommenderDataProcessor:
    """
    Класс для загрузки и обработки данных рекомендательной системы

    Поддерживает:
    - MovieLens 100K (автоматическая загрузка)
    - MovieLens 1M (автоматическая загрузка)
    - Синтетические данные для тестирования
    """

    # URLs для загрузки датасетов
    MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def __init__(self, data_dir='data'):
        """
        Args:
            data_dir: директория для хранения данных
        """
        self.data_dir = data_dir
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.item_info = None  # Информация о товарах (фильмах)

        # Создаём директории
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)

    def download_movielens_100k(self):
        """
        Загрузка датасета MovieLens 100K

        MovieLens 100K содержит:
        - 100,000 рейтингов (1-5)
        - 943 пользователя
        - 1,682 фильма
        - Каждый пользователь оценил минимум 20 фильмов

        Returns:
            DataFrame с рейтингами
        """
        print("Загрузка датасета MovieLens 100K...")

        data_path = os.path.join(self.data_dir, 'raw', 'ml-100k')

        # Проверяем, есть ли уже загруженные данные
        ratings_file = os.path.join(data_path, 'u.data')
        if os.path.exists(ratings_file):
            print("  Датасет уже загружен, используем локальную копию")
        else:
            print(f"  Скачивание с {self.MOVIELENS_100K_URL}...")

            try:
                response = requests.get(self.MOVIELENS_100K_URL, timeout=60)
                response.raise_for_status()

                # Распаковываем архив
                with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                    zip_file.extractall(os.path.join(self.data_dir, 'raw'))

                print("  Загрузка завершена успешно!")

            except Exception as e:
                print(f"  Ошибка загрузки: {e}")
                print("  Переключаемся на синтетические данные...")
                return self.generate_synthetic_data()

        # Загружаем рейтинги
        df = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )

        # Загружаем информацию о фильмах
        movies_file = os.path.join(data_path, 'u.item')
        if os.path.exists(movies_file):
            self.item_info = pd.read_csv(
                movies_file,
                sep='|',
                encoding='latin-1',
                names=['item_id', 'title', 'release_date', 'video_release_date',
                       'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                usecols=['item_id', 'title']
            )

        print(f"\nМетаданные датасета MovieLens 100K:")
        print(f"  Рейтингов: {len(df):,}")
        print(f"  Пользователей: {df['user_id'].nunique():,}")
        print(f"  Фильмов: {df['item_id'].nunique():,}")
        print(f"  Диапазон рейтингов: {df['rating'].min()} - {df['rating'].max()}")

        return df

    def download_movielens_1m(self):
        """
        Загрузка датасета MovieLens 1M

        MovieLens 1M содержит:
        - 1,000,209 рейтингов (1-5)
        - 6,040 пользователей
        - 3,706 фильмов

        Returns:
            DataFrame с рейтингами
        """
        print("Загрузка датасета MovieLens 1M...")

        data_path = os.path.join(self.data_dir, 'raw', 'ml-1m')

        ratings_file = os.path.join(data_path, 'ratings.dat')
        if os.path.exists(ratings_file):
            print("  Датасет уже загружен")
        else:
            print(f"  Скачивание с {self.MOVIELENS_1M_URL}...")

            try:
                response = requests.get(self.MOVIELENS_1M_URL, timeout=120)
                response.raise_for_status()

                with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                    zip_file.extractall(os.path.join(self.data_dir, 'raw'))

                print("  Загрузка завершена!")

            except Exception as e:
                print(f"  Ошибка загрузки: {e}")
                return self.generate_synthetic_data()

        df = pd.read_csv(
            ratings_file,
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python',
            encoding='latin-1'
        )

        print(f"\nМетаданные датасета MovieLens 1M:")
        print(f"  Рейтингов: {len(df):,}")
        print(f"  Пользователей: {df['user_id'].nunique():,}")
        print(f"  Фильмов: {df['item_id'].nunique():,}")

        return df

    def generate_synthetic_data(self, n_users=1500, n_items=2000, n_ratings=80000):
        """
        Генерация синтетического датасета для тестирования

        Создаёт реалистичное распределение рейтингов с:
        - Разной активностью пользователей
        - Разной популярностью товаров
        - Смещением к положительным рейтингам (типично для реальных данных)

        Args:
            n_users: количество пользователей
            n_items: количество товаров
            n_ratings: количество рейтингов

        Returns:
            DataFrame с синтетическими рейтингами
        """
        print("Генерация синтетического датасета...")

        np.random.seed(42)

        # Генерируем рейтинги
        user_ids = np.random.randint(1, n_users + 1, n_ratings)
        item_ids = np.random.randint(1, n_items + 1, n_ratings)

        # Реалистичное распределение рейтингов (смещение к 4-5)
        ratings = np.random.choice(
            [1, 2, 3, 4, 5],
            n_ratings,
            p=[0.05, 0.10, 0.25, 0.35, 0.25]
        )

        timestamps = np.random.randint(1000000000, 1600000000, n_ratings)

        df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps
        })

        # Удаляем дубликаты
        df = df.drop_duplicates(subset=['user_id', 'item_id'])

        # Фильтруем активных пользователей и популярные товары
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()

        active_users = user_counts[user_counts >= 10].index
        popular_items = item_counts[item_counts >= 5].index

        df = df[df['user_id'].isin(active_users) & df['item_id'].isin(popular_items)]

        print(f"\nСинтетический датасет создан:")
        print(f"  Рейтингов: {len(df):,}")
        print(f"  Пользователей: {df['user_id'].nunique():,}")
        print(f"  Товаров: {df['item_id'].nunique():,}")

        return df

    def load_dataset(self, dataset='movielens-100k'):
        """
        Загрузка датасета

        Args:
            dataset: тип датасета
                - 'movielens-100k': MovieLens 100K
                - 'movielens-1m': MovieLens 1M
                - 'synthetic': синтетические данные

        Returns:
            DataFrame с рейтингами
        """
        if dataset == 'movielens-100k':
            return self.download_movielens_100k()
        elif dataset == 'movielens-1m':
            return self.download_movielens_1m()
        elif dataset == 'synthetic':
            return self.generate_synthetic_data()
        else:
            print(f"Неизвестный датасет: {dataset}, используем MovieLens 100K")
            return self.download_movielens_100k()

    def analyze_dataset(self, df):
        """
        Детальный анализ датасета

        Args:
            df: DataFrame с рейтингами

        Returns:
            dict с метриками датасета
        """
        print("\n" + "="*60)
        print("АНАЛИЗ ДАТАСЕТА")
        print("="*60)

        n_users = df['user_id'].nunique()
        n_items = df['item_id'].nunique()
        n_ratings = len(df)

        # Разреженность матрицы
        sparsity = 1 - (n_ratings / (n_users * n_items))

        # Статистика рейтингов
        rating_stats = df['rating'].describe()

        # Статистика по пользователям
        user_stats = df.groupby('user_id').size()

        # Статистика по товарам
        item_stats = df.groupby('item_id').size()

        info = {
            'n_users': n_users,
            'n_items': n_items,
            'n_ratings': n_ratings,
            'sparsity': sparsity,
            'sparsity_percent': sparsity * 100,
            'rating_mean': df['rating'].mean(),
            'rating_std': df['rating'].std(),
            'rating_min': df['rating'].min(),
            'rating_max': df['rating'].max(),
            'rating_distribution': df['rating'].value_counts().to_dict(),
            'avg_ratings_per_user': user_stats.mean(),
            'min_ratings_per_user': user_stats.min(),
            'max_ratings_per_user': user_stats.max(),
            'median_ratings_per_user': user_stats.median(),
            'avg_ratings_per_item': item_stats.mean(),
            'min_ratings_per_item': item_stats.min(),
            'max_ratings_per_item': item_stats.max(),
            'median_ratings_per_item': item_stats.median()
        }

        # Вывод статистики
        print(f"\nОбщая информация:")
        print(f"  Пользователей: {n_users:,}")
        print(f"  Товаров/фильмов: {n_items:,}")
        print(f"  Рейтингов: {n_ratings:,}")
        print(f"  Разреженность матрицы: {sparsity*100:.2f}%")

        print(f"\nСтатистика рейтингов:")
        print(f"  Среднее: {info['rating_mean']:.2f}")
        print(f"  Стд. отклонение: {info['rating_std']:.2f}")
        print(f"  Диапазон: [{info['rating_min']}, {info['rating_max']}]")

        print(f"\nРаспределение рейтингов:")
        for rating in sorted(info['rating_distribution'].keys()):
            count = info['rating_distribution'][rating]
            pct = count / n_ratings * 100
            bar = '#' * int(pct / 2)
            print(f"  {rating}: {count:,} ({pct:.1f}%) {bar}")

        print(f"\nАктивность пользователей (рейтингов на пользователя):")
        print(f"  Среднее: {info['avg_ratings_per_user']:.1f}")
        print(f"  Медиана: {info['median_ratings_per_user']:.1f}")
        print(f"  Мин/Макс: {info['min_ratings_per_user']}/{info['max_ratings_per_user']}")

        print(f"\nПопулярность товаров (рейтингов на товар):")
        print(f"  Среднее: {info['avg_ratings_per_item']:.1f}")
        print(f"  Медиана: {info['median_ratings_per_item']:.1f}")
        print(f"  Мин/Макс: {info['min_ratings_per_item']}/{info['max_ratings_per_item']}")

        return info

    def create_mappings(self, df):
        """
        Создание маппингов ID → индексы

        Преобразует оригинальные ID пользователей и товаров
        в непрерывные индексы [0, N-1] для использования в моделях.

        Args:
            df: DataFrame с рейтингами

        Returns:
            DataFrame с добавленными столбцами user_idx, item_idx
        """
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())

        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}

        df = df.copy()
        df['user_idx'] = df['user_id'].map(self.user_mapping)
        df['item_idx'] = df['item_id'].map(self.item_mapping)

        print(f"\nСоздан маппинг:")
        print(f"  Пользователи: {len(self.user_mapping)} (0-{len(self.user_mapping)-1})")
        print(f"  Товары: {len(self.item_mapping)} (0-{len(self.item_mapping)-1})")

        return df

    def create_user_item_matrix(self, df):
        """
        Создание разреженной user-item матрицы

        Матрица R размера (n_users × n_items), где R[u,i] = рейтинг
        пользователя u для товара i (0 если нет рейтинга).

        Args:
            df: DataFrame с рейтингами (должен содержать user_idx, item_idx)

        Returns:
            scipy.sparse.csr_matrix - разреженная матрица
        """
        print("\nСоздание user-item матрицы...")
        start_time = time.time()

        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        user_item_matrix = csr_matrix(
            (df['rating'].values, (df['user_idx'].values, df['item_idx'].values)),
            shape=(n_users, n_items)
        )

        creation_time = time.time() - start_time

        # Статистика матрицы
        nnz = user_item_matrix.nnz  # Non-zero элементы
        density = nnz / (n_users * n_items) * 100
        memory_mb = user_item_matrix.data.nbytes / (1024 * 1024)

        print(f"  Размер: {n_users} x {n_items}")
        print(f"  Ненулевых элементов: {nnz:,}")
        print(f"  Плотность: {density:.2f}%")
        print(f"  Память: {memory_mb:.2f} MB")
        print(f"  Время создания: {creation_time:.2f} сек")

        return user_item_matrix

    def split_data(self, df, test_size=0.2, random_state=42, stratify=True):
        """
        Разделение данных на обучающую и тестовую выборки

        Args:
            df: DataFrame с рейтингами
            test_size: доля тестовой выборки
            random_state: seed для воспроизводимости
            stratify: использовать стратификацию по рейтингам

        Returns:
            train_data, test_data - два DataFrame
        """
        print(f"\nРазделение данных (test_size={test_size})...")

        if stratify:
            train_data, test_data = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df['rating']
            )
        else:
            train_data, test_data = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )

        print(f"  Обучающая выборка: {len(train_data):,} рейтингов")
        print(f"  Тестовая выборка: {len(test_data):,} рейтингов")

        # Проверка распределения рейтингов
        train_dist = train_data['rating'].value_counts(normalize=True).sort_index()
        test_dist = test_data['rating'].value_counts(normalize=True).sort_index()

        print("\n  Распределение рейтингов (Train / Test):")
        for rating in sorted(df['rating'].unique()):
            train_pct = train_dist.get(rating, 0) * 100
            test_pct = test_dist.get(rating, 0) * 100
            print(f"    {rating}: {train_pct:.1f}% / {test_pct:.1f}%")

        return train_data, test_data

    def process_data(self, dataset='movielens-100k', test_size=0.2):
        """
        Полный пайплайн обработки данных

        Args:
            dataset: тип датасета
            test_size: размер тестовой выборки

        Returns:
            train_data, test_data, user_item_matrix, dataset_info
        """
        print("\n" + "="*60)
        print("ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ")
        print("="*60)

        # Загрузка
        df = self.load_dataset(dataset)

        # Анализ
        dataset_info = self.analyze_dataset(df)

        # Маппинги
        df = self.create_mappings(df)

        # User-item матрица
        user_item_matrix = self.create_user_item_matrix(df)

        # Разделение
        train_data, test_data = self.split_data(df, test_size)

        # Добавляем информацию в dataset_info
        dataset_info['n_users_mapped'] = len(self.user_mapping)
        dataset_info['n_items_mapped'] = len(self.item_mapping)
        dataset_info['train_size'] = len(train_data)
        dataset_info['test_size'] = len(test_data)

        print("\n" + "="*60)
        print("ОБРАБОТКА ДАННЫХ ЗАВЕРШЕНА")
        print("="*60)

        return train_data, test_data, user_item_matrix, dataset_info

    def get_item_name(self, item_idx):
        """
        Получить название товара по индексу

        Args:
            item_idx: индекс товара

        Returns:
            str: название товара
        """
        if self.item_info is None:
            return f"Item_{item_idx}"

        item_id = self.reverse_item_mapping.get(item_idx)
        if item_id is None:
            return f"Item_{item_idx}"

        item_row = self.item_info[self.item_info['item_id'] == item_id]
        if len(item_row) > 0:
            return item_row['title'].values[0]

        return f"Item_{item_id}"

    def get_user_history(self, df, user_idx, top_n=10):
        """
        Получить историю рейтингов пользователя

        Args:
            df: DataFrame с рейтингами
            user_idx: индекс пользователя
            top_n: количество записей

        Returns:
            DataFrame с историей
        """
        user_data = df[df['user_idx'] == user_idx].sort_values('rating', ascending=False)

        if self.item_info is not None:
            user_data = user_data.merge(
                self.item_info[['item_id', 'title']],
                on='item_id',
                how='left'
            )

        return user_data.head(top_n)


# Для обратной совместимости с существующим кодом
def load_dataset():
    """Функция для обратной совместимости"""
    processor = RecommenderDataProcessor()
    return processor.load_dataset('movielens-100k')
