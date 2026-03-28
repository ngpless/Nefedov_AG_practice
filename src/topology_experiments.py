"""
Модуль экспериментов с топологией нейронных сетей

Реализует требования практики:
1. Экспериментальный подбор характеристик сети
2. Анализ влияния числа слоёв на качество
3. Анализ влияния размера эмбеддингов
4. Расчёт скорости обучения как функции топологии
5. Визуализация результатов экспериментов

Автор: Нефедов Алексей Геннадьевич
Дата: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Импортируем модели
from neural_models import (
    RatingDataset, GMF, MLP_Recommender,
    NeuralCollaborativeFiltering, WideAndDeep
)


class TopologyExperiments:
    """
    Класс для проведения экспериментов с топологией нейронных сетей

    Эксперименты:
    1. Влияние числа скрытых слоёв
    2. Влияние размера эмбеддингов
    3. Влияние размера скрытых слоёв
    4. Анализ скорости обучения vs качества
    5. Подбор оптимальных гиперпараметров
    """

    def __init__(self, n_users, n_items, device=None, results_dir='results/experiments'):
        """
        Args:
            n_users: количество пользователей
            n_items: количество товаров
            device: устройство для вычислений
            results_dir: директория для сохранения результатов
        """
        self.n_users = n_users
        self.n_items = n_items

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)

        self.experiment_results = {}

        print(f"TopologyExperiments инициализирован")
        print(f"  Устройство: {self.device}")
        print(f"  Пользователей: {n_users}, Товаров: {n_items}")

    def create_data_loaders(self, train_data, test_data, batch_size=256):
        """Создание DataLoader"""
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

    def train_and_evaluate(self, model, train_loader, test_loader,
                           epochs=30, lr=0.001, patience=5):
        """
        Обучение модели и оценка

        Returns:
            dict с метриками: rmse, mae, training_time, epochs_trained, n_parameters
        """
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        best_test_loss = float('inf')
        patience_counter = 0
        epochs_trained = 0

        start_time = time.time()

        for epoch in range(epochs):
            # Обучение
            model.train()
            for user_ids, item_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)

                optimizer.zero_grad()
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Оценка
            model.eval()
            test_losses = []
            with torch.no_grad():
                for user_ids, item_ids, ratings in test_loader:
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)

                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    test_losses.append(loss.item())

            test_loss = np.mean(test_losses)
            epochs_trained = epoch + 1

            # Early stopping
            if test_loss < best_test_loss - 0.001:
                best_test_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        training_time = time.time() - start_time

        # Финальная оценка
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)

                predictions = model(user_ids, item_ids)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.numpy())

        all_predictions = np.clip(all_predictions, 1, 5)

        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'rmse': rmse,
            'mae': mae,
            'training_time': training_time,
            'epochs_trained': epochs_trained,
            'n_parameters': n_parameters
        }

    def experiment_hidden_layers(self, train_data, test_data,
                                  embed_dim=32, epochs=30, batch_size=256):
        """
        Эксперимент 1: Влияние числа скрытых слоёв

        Исследуем, как количество слоёв влияет на:
        - Качество модели (RMSE)
        - Время обучения
        - Количество параметров

        Архитектуры:
        - 1 слой: [64]
        - 2 слоя: [128, 64]
        - 3 слоя: [256, 128, 64]
        - 4 слоя: [512, 256, 128, 64]
        """
        print("\n" + "="*60)
        print("ЭКСПЕРИМЕНТ 1: ВЛИЯНИЕ ЧИСЛА СКРЫТЫХ СЛОЁВ")
        print("="*60)

        layer_configs = [
            [64],                      # 1 слой
            [128, 64],                 # 2 слоя
            [256, 128, 64],            # 3 слоя
            [512, 256, 128, 64],       # 4 слоя
            [512, 256, 128, 64, 32],   # 5 слоёв
        ]

        train_loader, test_loader = self.create_data_loaders(train_data, test_data, batch_size)

        results = []

        for config in layer_configs:
            n_layers = len(config)
            print(f"\nТестируем {n_layers} слой(ев): {config}")

            model = MLP_Recommender(
                self.n_users, self.n_items,
                embed_dim=embed_dim,
                hidden_layers=config,
                dropout=0.2
            )

            metrics = self.train_and_evaluate(
                model, train_loader, test_loader, epochs
            )

            result = {
                'n_layers': n_layers,
                'config': config,
                **metrics
            }
            results.append(result)

            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Параметров: {metrics['n_parameters']:,}")
            print(f"  Время: {metrics['training_time']:.2f}с")

        self.experiment_results['hidden_layers'] = results
        return results

    def experiment_embedding_size(self, train_data, test_data,
                                   hidden_layers=[128, 64], epochs=30, batch_size=256):
        """
        Эксперимент 2: Влияние размера эмбеддингов

        Эмбеддинги - это плотные векторные представления пользователей и товаров.
        Больший размер может захватить больше информации, но увеличивает
        риск переобучения и время обучения.

        Тестируемые размеры: 8, 16, 32, 64, 128
        """
        print("\n" + "="*60)
        print("ЭКСПЕРИМЕНТ 2: ВЛИЯНИЕ РАЗМЕРА ЭМБЕДДИНГОВ")
        print("="*60)

        embed_sizes = [8, 16, 32, 64, 128]

        train_loader, test_loader = self.create_data_loaders(train_data, test_data, batch_size)

        results = []

        for embed_dim in embed_sizes:
            print(f"\nРазмер эмбеддинга: {embed_dim}")

            model = MLP_Recommender(
                self.n_users, self.n_items,
                embed_dim=embed_dim,
                hidden_layers=hidden_layers,
                dropout=0.2
            )

            metrics = self.train_and_evaluate(
                model, train_loader, test_loader, epochs
            )

            result = {
                'embed_dim': embed_dim,
                **metrics
            }
            results.append(result)

            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Параметров: {metrics['n_parameters']:,}")
            print(f"  Время: {metrics['training_time']:.2f}с")

        self.experiment_results['embedding_size'] = results
        return results

    def experiment_model_comparison(self, train_data, test_data,
                                     epochs=30, batch_size=256):
        """
        Эксперимент 3: Сравнение архитектур

        Сравниваем разные архитектуры нейросетей:
        - GMF (Generalized Matrix Factorization)
        - MLP (Multi-Layer Perceptron)
        - NCF (Neural Collaborative Filtering)
        - Wide & Deep
        """
        print("\n" + "="*60)
        print("ЭКСПЕРИМЕНТ 3: СРАВНЕНИЕ АРХИТЕКТУР")
        print("="*60)

        train_loader, test_loader = self.create_data_loaders(train_data, test_data, batch_size)

        models_config = [
            ('GMF', GMF(self.n_users, self.n_items, embed_dim=64)),
            ('MLP', MLP_Recommender(self.n_users, self.n_items,
                                    embed_dim=32, hidden_layers=[128, 64, 32])),
            ('NCF', NeuralCollaborativeFiltering(self.n_users, self.n_items,
                                                  gmf_embed_dim=32, mlp_embed_dim=32,
                                                  mlp_hidden_layers=[64, 32, 16])),
            ('Wide & Deep', WideAndDeep(self.n_users, self.n_items,
                                        embed_dim=32, deep_layers=[64, 32])),
        ]

        results = []

        for name, model in models_config:
            print(f"\nТестируем: {name}")

            metrics = self.train_and_evaluate(
                model, train_loader, test_loader, epochs
            )

            result = {
                'model': name,
                **metrics
            }
            results.append(result)

            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  Параметров: {metrics['n_parameters']:,}")
            print(f"  Время: {metrics['training_time']:.2f}с")

        self.experiment_results['model_comparison'] = results
        return results

    def experiment_training_speed(self, train_data, test_data,
                                   epochs=30, batch_size=256):
        """
        Эксперимент 4: Анализ скорости обучения

        Исследуем зависимость времени обучения от:
        - Количества параметров модели
        - Топологии сети
        - Размера батча

        Это важное требование практики: "Рассчитать скорость обучения
        нейросети как функцию от её топологии"
        """
        print("\n" + "="*60)
        print("ЭКСПЕРИМЕНТ 4: СКОРОСТЬ ОБУЧЕНИЯ VS ТОПОЛОГИЯ")
        print("="*60)

        train_loader, test_loader = self.create_data_loaders(train_data, test_data, batch_size)

        # Разные конфигурации с разным числом параметров
        configs = [
            ('Small (1 слой)', {'embed_dim': 16, 'hidden_layers': [32]}),
            ('Medium (2 слоя)', {'embed_dim': 32, 'hidden_layers': [64, 32]}),
            ('Large (3 слоя)', {'embed_dim': 64, 'hidden_layers': [128, 64, 32]}),
            ('XLarge (4 слоя)', {'embed_dim': 64, 'hidden_layers': [256, 128, 64, 32]}),
            ('XXLarge (5 слоёв)', {'embed_dim': 128, 'hidden_layers': [512, 256, 128, 64, 32]}),
        ]

        results = []

        for name, config in configs:
            print(f"\n{name}")

            model = MLP_Recommender(
                self.n_users, self.n_items,
                **config,
                dropout=0.2
            )

            n_params = model.count_parameters()

            # Замеряем время на 1 эпоху
            model = model.to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            epoch_times = []

            for epoch in range(min(5, epochs)):  # Только 5 эпох для замера
                start = time.time()
                model.train()
                for user_ids, item_ids, ratings in train_loader:
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)

                    optimizer.zero_grad()
                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    loss.backward()
                    optimizer.step()

                epoch_time = time.time() - start
                epoch_times.append(epoch_time)

            avg_epoch_time = np.mean(epoch_times)

            result = {
                'config': name,
                'n_parameters': n_params,
                'avg_epoch_time': avg_epoch_time,
                'time_per_1k_params': avg_epoch_time / (n_params / 1000),
                **config
            }
            results.append(result)

            print(f"  Параметров: {n_params:,}")
            print(f"  Время на эпоху: {avg_epoch_time:.3f}с")
            print(f"  Время на 1K параметров: {result['time_per_1k_params']:.5f}с")

        self.experiment_results['training_speed'] = results
        return results

    def experiment_batch_size(self, train_data, test_data, epochs=20):
        """
        Эксперимент 5: Влияние размера батча

        Размер батча влияет на:
        - Скорость сходимости
        - Стабильность обучения
        - Использование памяти
        """
        print("\n" + "="*60)
        print("ЭКСПЕРИМЕНТ 5: ВЛИЯНИЕ РАЗМЕРА БАТЧА")
        print("="*60)

        batch_sizes = [64, 128, 256, 512, 1024]

        results = []

        for batch_size in batch_sizes:
            print(f"\nРазмер батча: {batch_size}")

            train_loader, test_loader = self.create_data_loaders(
                train_data, test_data, batch_size
            )

            model = MLP_Recommender(
                self.n_users, self.n_items,
                embed_dim=32,
                hidden_layers=[128, 64],
                dropout=0.2
            )

            metrics = self.train_and_evaluate(
                model, train_loader, test_loader, epochs
            )

            result = {
                'batch_size': batch_size,
                **metrics
            }
            results.append(result)

            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Время: {metrics['training_time']:.2f}с")

        self.experiment_results['batch_size'] = results
        return results

    def run_all_experiments(self, train_data, test_data, epochs=30, batch_size=256):
        """
        Запуск всех экспериментов

        Returns:
            dict со всеми результатами
        """
        print("\n" + "="*60)
        print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ С ТОПОЛОГИЕЙ")
        print("="*60)

        start_time = time.time()

        # Эксперимент 1: Число слоёв
        self.experiment_hidden_layers(train_data, test_data, epochs=epochs, batch_size=batch_size)

        # Эксперимент 2: Размер эмбеддингов
        self.experiment_embedding_size(train_data, test_data, epochs=epochs, batch_size=batch_size)

        # Эксперимент 3: Сравнение архитектур
        self.experiment_model_comparison(train_data, test_data, epochs=epochs, batch_size=batch_size)

        # Эксперимент 4: Скорость обучения
        self.experiment_training_speed(train_data, test_data, epochs=epochs, batch_size=batch_size)

        # Эксперимент 5: Размер батча
        self.experiment_batch_size(train_data, test_data, epochs=min(20, epochs))

        total_time = time.time() - start_time

        print("\n" + "="*60)
        print(f"ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ за {total_time:.1f} секунд")
        print("="*60)

        # Сохраняем результаты
        self.save_results()

        return self.experiment_results

    def save_results(self):
        """Сохранение результатов в JSON"""
        results_file = os.path.join(self.results_dir, 'topology_experiments.json')

        # Конвертируем numpy типы в Python типы
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_serializable = {}
        for exp_name, exp_results in self.experiment_results.items():
            results_serializable[exp_name] = [
                {k: convert(v) for k, v in r.items()}
                for r in exp_results
            ]

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)

        print(f"Результаты сохранены в {results_file}")

    def plot_all_experiments(self):
        """
        Визуализация всех экспериментов

        Создаёт графики для отчёта
        """
        print("\nСоздание визуализаций...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Эксперименты с топологией нейронной сети', fontsize=14)

        # График 1: Число слоёв vs RMSE
        if 'hidden_layers' in self.experiment_results:
            ax = axes[0, 0]
            data = self.experiment_results['hidden_layers']
            layers = [r['n_layers'] for r in data]
            rmse = [r['rmse'] for r in data]
            ax.plot(layers, rmse, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Число скрытых слоёв')
            ax.set_ylabel('RMSE')
            ax.set_title('Влияние числа слоёв на качество')
            ax.grid(True, alpha=0.3)

        # График 2: Размер эмбеддингов vs RMSE
        if 'embedding_size' in self.experiment_results:
            ax = axes[0, 1]
            data = self.experiment_results['embedding_size']
            embed = [r['embed_dim'] for r in data]
            rmse = [r['rmse'] for r in data]
            ax.plot(embed, rmse, 'go-', linewidth=2, markersize=8)
            ax.set_xlabel('Размер эмбеддинга')
            ax.set_ylabel('RMSE')
            ax.set_title('Влияние размера эмбеддингов')
            ax.grid(True, alpha=0.3)

        # График 3: Сравнение архитектур
        if 'model_comparison' in self.experiment_results:
            ax = axes[0, 2]
            data = self.experiment_results['model_comparison']
            models = [r['model'] for r in data]
            rmse = [r['rmse'] for r in data]
            colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
            bars = ax.bar(models, rmse, color=colors)
            ax.set_ylabel('RMSE')
            ax.set_title('Сравнение архитектур')
            ax.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars, rmse):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # График 4: Параметры vs Время обучения
        if 'training_speed' in self.experiment_results:
            ax = axes[1, 0]
            data = self.experiment_results['training_speed']
            params = [r['n_parameters'] / 1000 for r in data]  # в тысячах
            times = [r['avg_epoch_time'] for r in data]
            ax.scatter(params, times, s=100, c='red', alpha=0.7)
            for i, r in enumerate(data):
                ax.annotate(r['config'].split()[0], (params[i], times[i]),
                           textcoords="offset points", xytext=(5,5), fontsize=8)
            ax.set_xlabel('Параметров (тыс.)')
            ax.set_ylabel('Время на эпоху (с)')
            ax.set_title('Скорость обучения vs Параметры')
            ax.grid(True, alpha=0.3)

        # График 5: Размер батча vs Время
        if 'batch_size' in self.experiment_results:
            ax = axes[1, 1]
            data = self.experiment_results['batch_size']
            batch = [r['batch_size'] for r in data]
            times = [r['training_time'] for r in data]
            rmse = [r['rmse'] for r in data]

            ax2 = ax.twinx()
            l1 = ax.plot(batch, times, 'b-o', label='Время обучения')
            l2 = ax2.plot(batch, rmse, 'r-s', label='RMSE')

            ax.set_xlabel('Размер батча')
            ax.set_ylabel('Время (с)', color='blue')
            ax2.set_ylabel('RMSE', color='red')
            ax.set_title('Влияние размера батча')

            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')

        # График 6: Сводная таблица лучших параметров
        if 'model_comparison' in self.experiment_results:
            ax = axes[1, 2]
            ax.axis('off')

            data = self.experiment_results['model_comparison']
            best = min(data, key=lambda x: x['rmse'])

            text = "ЛУЧШАЯ КОНФИГУРАЦИЯ\n\n"
            text += f"Модель: {best['model']}\n"
            text += f"RMSE: {best['rmse']:.4f}\n"
            text += f"MAE: {best['mae']:.4f}\n"
            text += f"Параметров: {best['n_parameters']:,}\n"
            text += f"Время обучения: {best['training_time']:.1f}с"

            ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='center', horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()

        # Сохранение
        fig_path = os.path.join(self.results_dir, 'figures', 'topology_experiments.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Графики сохранены в {fig_path}")

        # Дополнительный график: детальное сравнение слоёв
        self._plot_layers_detail()

    def _plot_layers_detail(self):
        """Детальный график анализа слоёв"""
        if 'hidden_layers' not in self.experiment_results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        data = self.experiment_results['hidden_layers']
        layers = [r['n_layers'] for r in data]

        # RMSE vs Layers
        ax = axes[0]
        rmse = [r['rmse'] for r in data]
        ax.bar(layers, rmse, color='steelblue', alpha=0.8)
        ax.set_xlabel('Число слоёв')
        ax.set_ylabel('RMSE')
        ax.set_title('Качество модели')
        for i, (l, r) in enumerate(zip(layers, rmse)):
            ax.text(l, r + 0.01, f'{r:.3f}', ha='center', fontsize=9)

        # Parameters vs Layers
        ax = axes[1]
        params = [r['n_parameters'] / 1000 for r in data]
        ax.bar(layers, params, color='coral', alpha=0.8)
        ax.set_xlabel('Число слоёв')
        ax.set_ylabel('Параметры (тыс.)')
        ax.set_title('Сложность модели')
        for i, (l, p) in enumerate(zip(layers, params)):
            ax.text(l, p + 1, f'{p:.0f}K', ha='center', fontsize=9)

        # Training time vs Layers
        ax = axes[2]
        times = [r['training_time'] for r in data]
        ax.bar(layers, times, color='mediumseagreen', alpha=0.8)
        ax.set_xlabel('Число слоёв')
        ax.set_ylabel('Время обучения (с)')
        ax.set_title('Время обучения')
        for i, (l, t) in enumerate(zip(layers, times)):
            ax.text(l, t + 0.5, f'{t:.1f}с', ha='center', fontsize=9)

        plt.suptitle('Детальный анализ влияния числа скрытых слоёв', fontsize=12)
        plt.tight_layout()

        fig_path = os.path.join(self.results_dir, 'figures', 'layers_analysis.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Детальный график слоёв сохранён в {fig_path}")

    def get_best_configuration(self):
        """
        Получить лучшую конфигурацию на основе экспериментов

        Returns:
            dict с оптимальными параметрами
        """
        best_config = {}

        # Лучшее число слоёв
        if 'hidden_layers' in self.experiment_results:
            data = self.experiment_results['hidden_layers']
            best = min(data, key=lambda x: x['rmse'])
            best_config['hidden_layers'] = best['config']
            best_config['n_layers'] = best['n_layers']

        # Лучший размер эмбеддингов
        if 'embedding_size' in self.experiment_results:
            data = self.experiment_results['embedding_size']
            best = min(data, key=lambda x: x['rmse'])
            best_config['embed_dim'] = best['embed_dim']

        # Лучшая архитектура
        if 'model_comparison' in self.experiment_results:
            data = self.experiment_results['model_comparison']
            best = min(data, key=lambda x: x['rmse'])
            best_config['best_model'] = best['model']
            best_config['best_rmse'] = best['rmse']

        # Лучший размер батча
        if 'batch_size' in self.experiment_results:
            data = self.experiment_results['batch_size']
            # Баланс между качеством и скоростью
            best = min(data, key=lambda x: x['rmse'])
            best_config['batch_size'] = best['batch_size']

        return best_config

    def print_summary(self):
        """Вывод сводки результатов"""
        print("\n" + "="*60)
        print("СВОДКА ЭКСПЕРИМЕНТОВ С ТОПОЛОГИЕЙ")
        print("="*60)

        if 'hidden_layers' in self.experiment_results:
            print("\n1. Влияние числа слоёв:")
            for r in self.experiment_results['hidden_layers']:
                print(f"   {r['n_layers']} слой(ев): RMSE={r['rmse']:.4f}, "
                      f"Параметров={r['n_parameters']:,}")

        if 'embedding_size' in self.experiment_results:
            print("\n2. Влияние размера эмбеддингов:")
            for r in self.experiment_results['embedding_size']:
                print(f"   embed_dim={r['embed_dim']}: RMSE={r['rmse']:.4f}")

        if 'model_comparison' in self.experiment_results:
            print("\n3. Сравнение архитектур:")
            for r in sorted(self.experiment_results['model_comparison'],
                          key=lambda x: x['rmse']):
                print(f"   {r['model']}: RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}")

        if 'training_speed' in self.experiment_results:
            print("\n4. Скорость обучения:")
            for r in self.experiment_results['training_speed']:
                print(f"   {r['config']}: {r['avg_epoch_time']:.3f}с/эпоха, "
                      f"{r['n_parameters']:,} параметров")

        best = self.get_best_configuration()
        print("\n" + "-"*60)
        print("РЕКОМЕНДУЕМАЯ КОНФИГУРАЦИЯ:")
        print("-"*60)
        for key, value in best.items():
            print(f"  {key}: {value}")
