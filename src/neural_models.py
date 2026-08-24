"""
Нейросетевые модели для рекомендательных систем на PyTorch

Модуль содержит реализации:
1. Neural Collaborative Filtering (NCF) - нейросетевая коллаборативная фильтрация
2. Autoencoder - автоэнкодер для рекомендаций
3. Wide & Deep Network - комбинация линейной и глубокой моделей
4. GMF (Generalized Matrix Factorization) - обобщённая матричная факторизация
5. MLP для рекомендаций - многослойный перцептрон

Автор: Нефедов Алексей Геннадьевич
Дата: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import time

from config import RANDOM_SEED, VALIDATION_SIZE, EARLY_STOPPING_PATIENCE
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ДАТАСЕТЫ ДЛЯ PyTorch
# ============================================================================

class RatingDataset(Dataset):
    """
    PyTorch Dataset для рейтинговых данных

    Преобразует данные в формат, подходящий для обучения нейросетей:
    - user_idx: индекс пользователя
    - item_idx: индекс товара
    - rating: целевой рейтинг
    """

    def __init__(self, user_ids, item_ids, ratings):
        """
        Args:
            user_ids: массив индексов пользователей
            item_ids: массив индексов товаров
            ratings: массив рейтингов
        """
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class AutoencoderDataset(Dataset):
    """
    Dataset для автоэнкодера

    Для каждого пользователя создаётся вектор его рейтингов по всем товарам
    """

    def __init__(self, user_item_matrix, user_indices):
        """
        Args:
            user_item_matrix: разреженная матрица user-item
            user_indices: индексы пользователей для выборки
        """
        self.matrix = user_item_matrix.toarray().astype(np.float32)
        self.user_indices = user_indices

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        user_idx = self.user_indices[idx]
        user_vector = torch.FloatTensor(self.matrix[user_idx])
        return user_vector, user_vector  # input = target для автоэнкодера


# ============================================================================
# АРХИТЕКТУРЫ НЕЙРОННЫХ СЕТЕЙ
# ============================================================================

class GMF(nn.Module):
    """
    Generalized Matrix Factorization (GMF)

    Обобщённая матричная факторизация - нейросетевой аналог классического SVD.

    Архитектура:
    ┌─────────────────────────────────────────────────────────────┐
    │  User ID → Embedding(n_users, embed_dim) → user_embed       │
    │  Item ID → Embedding(n_items, embed_dim) → item_embed       │
    │                                                             │
    │  output = user_embed * item_embed  (поэлементное умножение) │
    │  prediction = Linear(embed_dim, 1)                          │
    └─────────────────────────────────────────────────────────────┘

    Математическое описание:
    -----------------------
    p_u = E_user[u]  - эмбеддинг пользователя u
    q_i = E_item[i]  - эмбеддинг товара i

    ŷ_ui = σ(h^T (p_u ⊙ q_i))

    где:
    - ⊙ - поэлементное умножение (Hadamard product)
    - h - весовой вектор выходного слоя
    - σ - функция активации (linear для регрессии)
    """

    def __init__(self, n_users, n_items, embed_dim=64):
        """
        Args:
            n_users: количество уникальных пользователей
            n_items: количество уникальных товаров
            embed_dim: размерность эмбеддингов
        """
        super(GMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        # Эмбеддинги пользователей и товаров
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # Выходной слой
        self.output_layer = nn.Linear(embed_dim, 1)

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов методом Xavier"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, user_ids, item_ids):
        """
        Прямой проход

        Args:
            user_ids: тензор индексов пользователей [batch_size]
            item_ids: тензор индексов товаров [batch_size]

        Returns:
            predictions: предсказанные рейтинги [batch_size]
        """
        user_embed = self.user_embedding(user_ids)  # [batch, embed_dim]
        item_embed = self.item_embedding(item_ids)  # [batch, embed_dim]

        # Поэлементное умножение (Hadamard product)
        element_product = user_embed * item_embed  # [batch, embed_dim]

        # Предсказание
        output = self.output_layer(element_product)  # [batch, 1]

        return output.squeeze()

    def count_parameters(self):
        """Подсчёт количества обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP_Recommender(nn.Module):
    """
    Multi-Layer Perceptron для рекомендаций

    Многослойный перцептрон, который обучается предсказывать рейтинги
    на основе конкатенации эмбеддингов пользователя и товара.

    Архитектура:
    ┌─────────────────────────────────────────────────────────────┐
    │  User ID → Embedding → user_embed                           │
    │  Item ID → Embedding → item_embed                           │
    │                                                             │
    │  concat = [user_embed, item_embed]                          │
    │                                                             │
    │  Hidden Layer 1: Linear → BatchNorm → ReLU → Dropout        │
    │  Hidden Layer 2: Linear → BatchNorm → ReLU → Dropout        │
    │  ...                                                        │
    │  Output Layer: Linear → prediction                          │
    └─────────────────────────────────────────────────────────────┘

    Математическое описание:
    -----------------------
    z_0 = [p_u, q_i]  - конкатенация эмбеддингов

    Для каждого скрытого слоя l:
    a_l = W_l · z_{l-1} + b_l  (линейное преобразование)
    z_l = ReLU(BatchNorm(a_l))  (нормализация и активация)
    z_l = Dropout(z_l, p)  (регуляризация)

    Выход:
    ŷ_ui = W_out · z_L + b_out
    """

    def __init__(self, n_users, n_items, embed_dim=64, hidden_layers=[128, 64, 32], dropout=0.2):
        """
        Args:
            n_users: количество пользователей
            n_items: количество товаров
            embed_dim: размерность эмбеддингов
            hidden_layers: список размеров скрытых слоёв
            dropout: вероятность dropout для регуляризации
        """
        super(MLP_Recommender, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_layers = hidden_layers

        # Эмбеддинги
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        # Построение MLP слоёв
        layers = []
        input_dim = embed_dim * 2  # конкатенация user + item

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*layers)

        # Выходной слой
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        """Инициализация весов"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids, item_ids):
        """Прямой проход"""
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        # Конкатенация эмбеддингов
        concat = torch.cat([user_embed, item_embed], dim=-1)

        # Прохождение через MLP
        mlp_output = self.mlp_layers(concat)

        # Предсказание
        output = self.output_layer(mlp_output)

        return output.squeeze()

    def count_parameters(self):
        """Подсчёт параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF)

    Комбинация GMF и MLP для захвата как линейных, так и нелинейных
    взаимодействий между пользователями и товарами.

    Архитектура:
    ┌─────────────────────────────────────────────────────────────┐
    │                    GMF Path                                 │
    │  User → GMF_Embed → ┐                                       │
    │                     ├→ element_product                      │
    │  Item → GMF_Embed → ┘                                       │
    │                                                             │
    │                    MLP Path                                 │
    │  User → MLP_Embed → ┐                                       │
    │                     ├→ concat → MLP layers                  │
    │  Item → MLP_Embed → ┘                                       │
    │                                                             │
    │  NeuMF = concat(GMF_output, MLP_output)                     │
    │  prediction = Linear(NeuMF)                                 │
    └─────────────────────────────────────────────────────────────┘

    Математическое описание:
    -----------------------
    GMF компонент:
    φ_GMF = p_u^G ⊙ q_i^G

    MLP компонент:
    z_1 = ReLU(W_1 · [p_u^M, q_i^M] + b_1)
    z_2 = ReLU(W_2 · z_1 + b_2)
    ...
    φ_MLP = z_L

    Объединение:
    ŷ_ui = σ(h^T · [φ_GMF, φ_MLP])

    где p_u^G, q_i^G - эмбеддинги для GMF
        p_u^M, q_i^M - эмбеддинги для MLP
    """

    def __init__(self, n_users, n_items, gmf_embed_dim=32, mlp_embed_dim=32,
                 mlp_hidden_layers=[64, 32, 16], dropout=0.2):
        """
        Args:
            n_users: количество пользователей
            n_items: количество товаров
            gmf_embed_dim: размерность GMF эмбеддингов
            mlp_embed_dim: размерность MLP эмбеддингов
            mlp_hidden_layers: размеры скрытых слоёв MLP
            dropout: вероятность dropout
        """
        super(NeuralCollaborativeFiltering, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.gmf_embed_dim = gmf_embed_dim
        self.mlp_embed_dim = mlp_embed_dim
        self.mlp_hidden_layers = mlp_hidden_layers

        # GMF эмбеддинги
        self.gmf_user_embedding = nn.Embedding(n_users, gmf_embed_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, gmf_embed_dim)

        # MLP эмбеддинги
        self.mlp_user_embedding = nn.Embedding(n_users, mlp_embed_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, mlp_embed_dim)

        # MLP слои
        mlp_layers = []
        input_dim = mlp_embed_dim * 2

        for hidden_dim in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*mlp_layers)

        # NeuMF слой (объединение GMF и MLP)
        neumf_input_dim = gmf_embed_dim + mlp_hidden_layers[-1]
        self.neumf_layer = nn.Linear(neumf_input_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Инициализация весов"""
        nn.init.xavier_uniform_(self.gmf_user_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_item_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_user_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_item_embedding.weight)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, user_ids, item_ids):
        """Прямой проход"""
        # GMF путь
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item  # поэлементное умножение

        # MLP путь
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_concat = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp_layers(mlp_concat)

        # Объединение GMF и MLP
        neumf_input = torch.cat([gmf_output, mlp_output], dim=-1)

        # Финальное предсказание
        output = self.neumf_layer(neumf_input)

        return output.squeeze()

    def count_parameters(self):
        """Подсчёт параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Autoencoder(nn.Module):
    """
    Автоэнкодер для рекомендательных систем

    Обучается восстанавливать вектор рейтингов пользователя,
    при этом латентное представление захватывает предпочтения пользователя.

    Архитектура:
    ┌─────────────────────────────────────────────────────────────┐
    │  ENCODER (сжатие):                                          │
    │  Input(n_items) → Dense(512) → Dense(256) → Dense(latent)   │
    │                                                             │
    │  DECODER (восстановление):                                  │
    │  Latent → Dense(256) → Dense(512) → Output(n_items)         │
    └─────────────────────────────────────────────────────────────┘

    Математическое описание:
    -----------------------
    Encoder:
    h_1 = ReLU(W_1 · x + b_1)
    h_2 = ReLU(W_2 · h_1 + b_2)
    z = W_z · h_2 + b_z  (латентное представление)

    Decoder:
    h'_1 = ReLU(W'_1 · z + b'_1)
    h'_2 = ReLU(W'_2 · h'_1 + b'_2)
    x̂ = W_out · h'_2 + b_out  (восстановленный вектор)

    Функция потерь (только для известных рейтингов):
    L = Σ_{i: r_i > 0} (r_i - r̂_i)²
    """

    def __init__(self, n_items, hidden_dims=[512, 256], latent_dim=64, dropout=0.3):
        """
        Args:
            n_items: количество товаров (размерность входа)
            hidden_dims: размеры скрытых слоёв энкодера
            latent_dim: размерность латентного пространства
            dropout: вероятность dropout
        """
        super(Autoencoder, self).__init__()

        self.n_items = n_items
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        input_dim = n_items

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (зеркальная структура)
        decoder_layers = []
        input_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        decoder_layers.append(nn.Linear(input_dim, n_items))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self):
        """Инициализация весов"""
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        """
        Прямой проход

        Args:
            x: вектор рейтингов пользователя [batch, n_items]

        Returns:
            reconstructed: восстановленный вектор [batch, n_items]
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        """Получить латентное представление"""
        return self.encoder(x)

    def decode(self, z):
        """Декодировать из латентного пространства"""
        return self.decoder(z)

    def count_parameters(self):
        """Подсчёт параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WideAndDeep(nn.Module):
    """
    Wide & Deep Network

    Комбинирует "широкую" линейную модель для запоминания
    и "глубокую" нейросеть для обобщения.

    Архитектура:
    ┌─────────────────────────────────────────────────────────────┐
    │  WIDE (линейная модель):                                    │
    │  [user_id, item_id] → Linear → wide_output                  │
    │                                                             │
    │  DEEP (нейросеть):                                          │
    │  User → Embed ─┐                                            │
    │                ├→ concat → Dense → Dense → deep_output      │
    │  Item → Embed ─┘                                            │
    │                                                             │
    │  prediction = wide_output + deep_output                     │
    └─────────────────────────────────────────────────────────────┘

    Математическое описание:
    -----------------------
    Wide компонент:
    y_wide = w^T · x + b

    где x - вектор признаков (one-hot для user и item)

    Deep компонент:
    a_0 = [p_u, q_i]
    a_l = ReLU(W_l · a_{l-1} + b_l)
    y_deep = W_out · a_L + b_out

    Финальное предсказание:
    ŷ = y_wide + y_deep
    """

    def __init__(self, n_users, n_items, embed_dim=32, deep_layers=[64, 32], dropout=0.2):
        """
        Args:
            n_users: количество пользователей
            n_items: количество товаров
            embed_dim: размерность эмбеддингов для deep части
            deep_layers: размеры скрытых слоёв deep части
            dropout: вероятность dropout
        """
        super(WideAndDeep, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.deep_layers_config = deep_layers

        # Wide часть (линейная модель на эмбеддингах)
        self.wide_user_embedding = nn.Embedding(n_users, 1)
        self.wide_item_embedding = nn.Embedding(n_items, 1)
        self.wide_bias = nn.Parameter(torch.zeros(1))

        # Deep часть - эмбеддинги
        self.deep_user_embedding = nn.Embedding(n_users, embed_dim)
        self.deep_item_embedding = nn.Embedding(n_items, embed_dim)

        # Deep часть - MLP
        deep_mlp = []
        input_dim = embed_dim * 2

        for hidden_dim in deep_layers:
            deep_mlp.append(nn.Linear(input_dim, hidden_dim))
            deep_mlp.append(nn.BatchNorm1d(hidden_dim))
            deep_mlp.append(nn.ReLU())
            deep_mlp.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        deep_mlp.append(nn.Linear(input_dim, 1))
        self.deep_layers = nn.Sequential(*deep_mlp)

        self._init_weights()

    def _init_weights(self):
        """Инициализация весов"""
        nn.init.xavier_uniform_(self.wide_user_embedding.weight)
        nn.init.xavier_uniform_(self.wide_item_embedding.weight)
        nn.init.xavier_uniform_(self.deep_user_embedding.weight)
        nn.init.xavier_uniform_(self.deep_item_embedding.weight)

    def forward(self, user_ids, item_ids):
        """Прямой проход"""
        # Wide часть
        wide_user = self.wide_user_embedding(user_ids).squeeze()
        wide_item = self.wide_item_embedding(item_ids).squeeze()
        wide_output = wide_user + wide_item + self.wide_bias

        # Deep часть
        deep_user = self.deep_user_embedding(user_ids)
        deep_item = self.deep_item_embedding(item_ids)
        deep_concat = torch.cat([deep_user, deep_item], dim=-1)
        deep_output = self.deep_layers(deep_concat).squeeze()

        # Комбинация Wide и Deep
        output = wide_output + deep_output

        return output

    def count_parameters(self):
        """Подсчёт параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# ТРЕНЕР НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ
# ============================================================================

class NeuralModelTrainer:
    """
    Класс для обучения и оценки нейросетевых моделей рекомендаций

    Поддерживает:
    - Обучение с early stopping
    - Валидацию на отложенной выборке
    - Отслеживание метрик
    - Сохранение лучшей модели
    """

    def __init__(self, device=None):
        """
        Args:
            device: устройство для вычислений ('cuda' или 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Фиксация генераторов случайных чисел: инициализация весов и
        # порядок батчей воспроизводимы между запусками
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        print(f"Используется устройство: {self.device}")

        self.trained_models = {}
        self.training_history = {}

    def create_data_loaders(self, train_data, test_data, batch_size=256):
        """
        Создание DataLoader для обучения

        Args:
            train_data: DataFrame с обучающими данными
            test_data: DataFrame с тестовыми данными
            batch_size: размер батча

        Returns:
            train_loader, test_loader
        """
        train_dataset = RatingDataset(
            train_data['user_idx'].values,
            train_data['item_idx'].values,
            train_data['rating'].values
        )

        test_dataset = RatingDataset(
            test_data['user_idx'].values,
            test_data['item_idx'].values,
            test_data['rating'].values
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def create_train_val_test_loaders(self, train_data, test_data, batch_size=256,
                                      val_ratio=VALIDATION_SIZE):
        """
        Разбивает обучающую выборку на core/validation и создаёт три DataLoader.
        Валидация используется для early stopping и подбора числа эпох —
        тестовая выборка в обучении не участвует.
        """
        from sklearn.model_selection import train_test_split as _tts

        train_core, val_data = _tts(train_data, test_size=val_ratio,
                                    random_state=RANDOM_SEED,
                                    stratify=train_data['rating'])

        train_loader, val_loader = self.create_data_loaders(train_core, val_data, batch_size)
        _, test_loader = self.create_data_loaders(train_core, test_data, batch_size)

        return train_loader, val_loader, test_loader

    def train_model(self, model, train_loader, val_loader, test_loader,
                    epochs=50, lr=0.001,
                    weight_decay=1e-5, patience=EARLY_STOPPING_PATIENCE, min_delta=0.001):
        """
        Обучение модели с early stopping.

        Early stopping и scheduler работают по ВАЛИДАЦИОННОЙ выборке
        (отделённой от train); тестовая выборка используется один раз —
        для итоговой оценки качества.

        Args:
            model: модель PyTorch
            train_loader: DataLoader для обучения
            val_loader: DataLoader валидации (мониторинг early stopping)
            test_loader: DataLoader для итоговой оценки
            epochs: максимальное число эпох
            lr: скорость обучения
            weight_decay: коэффициент L2 регуляризации
            patience: терпение для early stopping
            min_delta: минимальное улучшение для early stopping

        Returns:
            dict с историей обучения и метриками
        """
        model = model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=0.5, patience=5)

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'epochs_trained': 0
        }

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        start_time = time.time()

        for epoch in range(epochs):
            # Обучение
            model.train()
            train_losses = []

            for user_ids, item_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)

                optimizer.zero_grad()
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

            # Оценка на валидационной выборке
            model.eval()
            val_losses = []
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for user_ids, item_ids, ratings in val_loader:
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)

                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    val_losses.append(loss.item())

                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(ratings.cpu().numpy())

            # Метрики
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_rmse = np.sqrt(train_loss)
            val_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            history['epochs_trained'] = epoch + 1

            # Learning rate scheduler
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Эпоха {epoch+1}/{epochs}: "
                      f"Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")

            if patience_counter >= patience:
                print(f"  Early stopping на эпохе {epoch+1}")
                break

        # Восстанавливаем лучшую модель
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        history['training_time'] = training_time

        # Финальная оценка
        model.eval()
        final_predictions = []
        final_targets = []

        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)

                predictions = model(user_ids, item_ids)
                final_predictions.extend(predictions.cpu().numpy())
                final_targets.extend(ratings.numpy())

        final_predictions = np.clip(final_predictions, 1, 5)

        history['final_rmse'] = np.sqrt(mean_squared_error(final_targets, final_predictions))
        history['final_mae'] = mean_absolute_error(final_targets, final_predictions)
        history['predictions'] = final_predictions

        return history

    def train_gmf(self, train_data, test_data, n_users, n_items,
                  embed_dim=64, epochs=50, batch_size=256, lr=0.001):
        """Обучение GMF модели"""
        print("Обучение GMF (Generalized Matrix Factorization)...")

        model = GMF(n_users, n_items, embed_dim)
        print(f"  Параметров: {model.count_parameters():,}")

        train_loader, val_loader, test_loader = self.create_train_val_test_loaders(
            train_data, test_data, batch_size)
        history = self.train_model(model, train_loader, val_loader, test_loader, epochs, lr)

        result = {
            'name': 'GMF',
            'rmse': history['final_rmse'],
            'mae': history['final_mae'],
            'training_time': history['training_time'],
            'predictions': history['predictions'],
            'epochs_trained': history['epochs_trained'],
            'n_parameters': model.count_parameters(),
            'history': history,
            'model': model
        }

        self.trained_models['GMF'] = result
        print(f"GMF: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, "
              f"Время={result['training_time']:.2f}с")

        return result

    def train_mlp(self, train_data, test_data, n_users, n_items,
                  embed_dim=64, hidden_layers=[128, 64, 32],
                  epochs=50, batch_size=256, lr=0.001):
        """Обучение MLP модели"""
        print("Обучение MLP Recommender...")

        model = MLP_Recommender(n_users, n_items, embed_dim, hidden_layers)
        print(f"  Параметров: {model.count_parameters():,}")
        print(f"  Архитектура: {hidden_layers}")

        train_loader, val_loader, test_loader = self.create_train_val_test_loaders(
            train_data, test_data, batch_size)
        history = self.train_model(model, train_loader, val_loader, test_loader, epochs, lr)

        result = {
            'name': 'MLP Recommender',
            'rmse': history['final_rmse'],
            'mae': history['final_mae'],
            'training_time': history['training_time'],
            'predictions': history['predictions'],
            'epochs_trained': history['epochs_trained'],
            'n_parameters': model.count_parameters(),
            'architecture': hidden_layers,
            'history': history,
            'model': model
        }

        self.trained_models['MLP Recommender'] = result
        print(f"MLP: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, "
              f"Время={result['training_time']:.2f}с")

        return result

    def train_ncf(self, train_data, test_data, n_users, n_items,
                  gmf_embed_dim=32, mlp_embed_dim=32, mlp_hidden_layers=[64, 32, 16],
                  epochs=50, batch_size=256, lr=0.001):
        """Обучение NCF модели"""
        print("Обучение Neural Collaborative Filtering (NCF)...")

        model = NeuralCollaborativeFiltering(n_users, n_items, gmf_embed_dim,
                                              mlp_embed_dim, mlp_hidden_layers)
        print(f"  Параметров: {model.count_parameters():,}")

        train_loader, val_loader, test_loader = self.create_train_val_test_loaders(
            train_data, test_data, batch_size)
        history = self.train_model(model, train_loader, val_loader, test_loader, epochs, lr)

        result = {
            'name': 'Neural Collaborative Filtering',
            'rmse': history['final_rmse'],
            'mae': history['final_mae'],
            'training_time': history['training_time'],
            'predictions': history['predictions'],
            'epochs_trained': history['epochs_trained'],
            'n_parameters': model.count_parameters(),
            'history': history,
            'model': model
        }

        self.trained_models['NCF'] = result
        print(f"NCF: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, "
              f"Время={result['training_time']:.2f}с")

        return result

    def train_wide_and_deep(self, train_data, test_data, n_users, n_items,
                            embed_dim=32, deep_layers=[64, 32],
                            epochs=50, batch_size=256, lr=0.001):
        """Обучение Wide & Deep модели"""
        print("Обучение Wide & Deep Network...")

        model = WideAndDeep(n_users, n_items, embed_dim, deep_layers)
        print(f"  Параметров: {model.count_parameters():,}")

        train_loader, val_loader, test_loader = self.create_train_val_test_loaders(
            train_data, test_data, batch_size)
        history = self.train_model(model, train_loader, val_loader, test_loader, epochs, lr)

        result = {
            'name': 'Wide & Deep',
            'rmse': history['final_rmse'],
            'mae': history['final_mae'],
            'training_time': history['training_time'],
            'predictions': history['predictions'],
            'epochs_trained': history['epochs_trained'],
            'n_parameters': model.count_parameters(),
            'history': history,
            'model': model
        }

        self.trained_models['Wide & Deep'] = result
        print(f"Wide & Deep: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, "
              f"Время={result['training_time']:.2f}с")

        return result

    def train_all_neural_models(self, train_data, test_data, n_users, n_items,
                                 epochs=50, batch_size=256):
        """Обучение всех нейросетевых моделей"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ")
        print("="*60)

        results = {}

        # GMF
        results['GMF'] = self.train_gmf(train_data, test_data, n_users, n_items,
                                        epochs=epochs, batch_size=batch_size)

        # MLP
        results['MLP Recommender'] = self.train_mlp(train_data, test_data, n_users, n_items,
                                                     epochs=epochs, batch_size=batch_size)

        # NCF
        results['NCF'] = self.train_ncf(train_data, test_data, n_users, n_items,
                                        epochs=epochs, batch_size=batch_size)

        # Wide & Deep
        results['Wide & Deep'] = self.train_wide_and_deep(train_data, test_data, n_users, n_items,
                                                           epochs=epochs, batch_size=batch_size)

        # Сводка результатов
        print("\n" + "-"*60)
        print("СВОДКА РЕЗУЛЬТАТОВ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ:")
        print("-"*60)

        for name, result in sorted(results.items(), key=lambda x: x[1]['rmse']):
            print(f"{name}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, "
                  f"Параметров={result['n_parameters']:,}")

        return results
