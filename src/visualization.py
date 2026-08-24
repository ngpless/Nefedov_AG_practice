"""
Модуль визуализации для рекомендательной системы

Создаёт графики и визуализации для:
1. Анализа данных
2. Сравнения моделей
3. Результатов обучения нейросетей
4. Экспериментов с топологией

Автор: Нефедов Алексей Геннадьевич
Дата: 2025
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Настройка шрифтов для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'


class RecommenderVisualizer:
    """
    Класс для создания визуализаций рекомендательной системы

    Методы:
    - create_data_analysis_plots: анализ данных
    - create_results_visualization: результаты моделей
    - create_neural_network_plots: визуализации для нейросетей
    - create_training_history_plot: история обучения
    """

    def __init__(self, results_dir='results'):
        """
        Args:
            results_dir: директория для сохранения графиков
        """
        plt.style.use('default')
        sns.set_palette("husl")

        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')

        os.makedirs(self.figures_dir, exist_ok=True)

    def create_data_analysis_plots(self, train_data, user_item_matrix, dataset_info):
        """
        Создание визуализаций анализа данных

        Args:
            train_data: DataFrame с обучающими данными
            user_item_matrix: разреженная матрица user-item
            dataset_info: словарь с информацией о датасете
        """
        print("Создание визуализаций анализа данных...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Анализ датасета рекомендательной системы', fontsize=16)

        # 1. Распределение рейтингов
        sns.countplot(data=train_data, x='rating', ax=axes[0, 0], palette='Blues_d')
        axes[0, 0].set_title('Распределение рейтингов')
        axes[0, 0].set_xlabel('Рейтинг')
        axes[0, 0].set_ylabel('Количество')

        # 2. Активность пользователей
        user_activity = train_data.groupby('user_id').size()
        axes[0, 1].hist(user_activity, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 1].set_title('Распределение активности пользователей')
        axes[0, 1].set_xlabel('Количество рейтингов на пользователя')
        axes[0, 1].set_ylabel('Количество пользователей')
        axes[0, 1].axvline(user_activity.mean(), color='red', linestyle='--',
                          label=f'Среднее: {user_activity.mean():.1f}')
        axes[0, 1].legend()

        # 3. Популярность товаров
        item_popularity = train_data.groupby('item_id').size()
        axes[0, 2].hist(item_popularity, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 2].set_title('Распределение популярности товаров')
        axes[0, 2].set_xlabel('Количество рейтингов на товар')
        axes[0, 2].set_ylabel('Количество товаров')
        axes[0, 2].axvline(item_popularity.mean(), color='red', linestyle='--',
                          label=f'Среднее: {item_popularity.mean():.1f}')
        axes[0, 2].legend()

        # 4. Средние рейтинги пользователей
        user_avg_ratings = train_data.groupby('user_id')['rating'].mean()
        axes[1, 0].hist(user_avg_ratings, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
        axes[1, 0].set_title('Распределение средних рейтингов пользователей')
        axes[1, 0].set_xlabel('Средний рейтинг')
        axes[1, 0].set_ylabel('Количество пользователей')

        # 5. Средние рейтинги товаров
        item_avg_ratings = train_data.groupby('item_id')['rating'].mean()
        axes[1, 1].hist(item_avg_ratings, bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
        axes[1, 1].set_title('Распределение средних рейтингов товаров')
        axes[1, 1].set_xlabel('Средний рейтинг')
        axes[1, 1].set_ylabel('Количество товаров')

        # 6. Разреженность матрицы
        n_ratings = dataset_info['n_ratings']
        n_possible = dataset_info['n_users'] * dataset_info['n_items']
        n_empty = n_possible - n_ratings

        labels = ['Заполненные\nячейки', 'Пустые\nячейки']
        sizes = [n_ratings, n_empty]
        colors = ['#66b3ff', '#ff9999']
        explode = (0.05, 0)

        axes[1, 2].pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct='%1.2f%%', shadow=True, startangle=90)
        axes[1, 2].set_title(f'Разреженность матрицы\n({n_ratings:,} из {n_possible:,})')

        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, 'data_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Сохранено: {fig_path}")

        # Дополнительно: корреляционная карта
        self.create_correlation_heatmap(train_data)

    def create_correlation_heatmap(self, train_data):
        """Создание корреляционной тепловой карты"""
        print("Создание корреляционной тепловой карты...")

        # Выбираем топ-20 популярных товаров для корреляции
        top_items = train_data.groupby('item_id').size().nlargest(20).index

        pivot_data = train_data[train_data['item_id'].isin(top_items)].pivot_table(
            index='user_id', columns='item_id', values='rating'
        ).fillna(0)

        plt.figure(figsize=(12, 10))
        correlation_matrix = pivot_data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='RdYlBu_r',
                    center=0, square=True, linewidths=0.5)
        plt.title('Корреляционная матрица топ-20 популярных товаров', fontsize=14)

        fig_path = os.path.join(self.figures_dir, 'correlation_heatmap.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Сохранено: {fig_path}")

    def create_results_visualization(self, all_results):
        """
        Создание визуализаций результатов всех моделей

        Args:
            all_results: словарь с результатами моделей
        """
        print("Создание визуализаций результатов...")

        self.plot_model_comparison(all_results)
        self.plot_training_times(all_results)
        self.plot_error_distribution(all_results)
        self.create_performance_summary_table(all_results)

    def plot_model_comparison(self, results):
        """Сравнение моделей по метрикам RMSE и MAE"""
        model_names = []
        rmse_values = []
        mae_values = []

        for name, result in results.items():
            model_names.append(name)
            rmse_values.append(result['rmse'])
            mae_values.append(result['mae'])

        # Сортируем по RMSE
        sorted_indices = np.argsort(rmse_values)
        model_names = [model_names[i] for i in sorted_indices]
        rmse_values = [rmse_values[i] for i in sorted_indices]
        mae_values = [mae_values[i] for i in sorted_indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # RMSE
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names)))
        bars1 = ax1.barh(range(len(model_names)), rmse_values, alpha=0.8, color=colors)
        ax1.set_xlabel('RMSE', fontsize=12)
        ax1.set_ylabel('Модели', fontsize=12)
        ax1.set_title('Сравнение моделей по RMSE (меньше - лучше)', fontsize=14)
        ax1.set_yticks(range(len(model_names)))
        ax1.set_yticklabels(model_names)

        for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)

        # MAE
        bars2 = ax2.barh(range(len(model_names)), mae_values, alpha=0.8, color=colors)
        ax2.set_xlabel('MAE', fontsize=12)
        ax2.set_ylabel('Модели', fontsize=12)
        ax2.set_title('Сравнение моделей по MAE (меньше - лучше)', fontsize=14)
        ax2.set_yticks(range(len(model_names)))
        ax2.set_yticklabels(model_names)

        for i, (bar, val) in enumerate(zip(bars2, mae_values)):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, 'model_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Сохранено: {fig_path}")

    def plot_training_times(self, results):
        """Сравнение времени обучения моделей"""
        model_names = []
        training_times = []

        for name, result in results.items():
            model_names.append(name)
            training_times.append(result['training_time'])

        # Сортируем по времени
        sorted_indices = np.argsort(training_times)
        model_names = [model_names[i] for i in sorted_indices]
        training_times = [training_times[i] for i in sorted_indices]

        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
        bars = plt.barh(range(len(model_names)), training_times, alpha=0.8, color=colors)

        plt.xlabel('Время обучения (секунды)', fontsize=12)
        plt.ylabel('Модели', fontsize=12)
        plt.title('Сравнение времени обучения моделей', fontsize=14)
        plt.yticks(range(len(model_names)), model_names)

        for bar, time_val in zip(bars, training_times):
            plt.text(time_val + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{time_val:.2f}с', va='center', fontsize=9)

        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, 'training_times.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Сохранено: {fig_path}")

    def plot_error_distribution(self, results):
        """Распределение ошибок и scatter-плоты"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        model_names = list(results.keys())
        rmse_values = [results[n]['rmse'] for n in model_names]
        mae_values = [results[n]['mae'] for n in model_names]
        training_times = [results[n]['training_time'] for n in model_names]

        # 1. RMSE vs MAE
        ax = axes[0, 0]
        scatter = ax.scatter(rmse_values, mae_values, c=range(len(model_names)),
                            cmap='tab20', s=100, alpha=0.8)
        ax.set_xlabel('RMSE')
        ax.set_ylabel('MAE')
        ax.set_title('Соотношение RMSE и MAE')
        ax.grid(True, alpha=0.3)

        # Подписи для точек
        for i, name in enumerate(model_names):
            short_name = name[:15] + '...' if len(name) > 15 else name
            ax.annotate(short_name, (rmse_values[i], mae_values[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=7)

        # 2. Время vs RMSE
        ax = axes[0, 1]
        ax.scatter(training_times, rmse_values, c=range(len(model_names)),
                  cmap='tab20', s=100, alpha=0.8)
        ax.set_xlabel('Время обучения (с)')
        ax.set_ylabel('RMSE')
        ax.set_title('Компромисс: Время обучения vs Качество')
        ax.grid(True, alpha=0.3)

        # 3. Гистограмма RMSE
        ax = axes[1, 0]
        ax.hist(rmse_values, bins=max(5, len(rmse_values)//3), alpha=0.7,
               color='steelblue', edgecolor='black')
        ax.axvline(np.mean(rmse_values), color='red', linestyle='--',
                  label=f'Среднее: {np.mean(rmse_values):.4f}')
        ax.set_xlabel('RMSE')
        ax.set_ylabel('Количество моделей')
        ax.set_title('Распределение RMSE')
        ax.legend()

        # 4. Топ-5 моделей
        ax = axes[1, 1]
        sorted_idx = np.argsort(rmse_values)[:5]
        top_names = [model_names[i] for i in sorted_idx]
        top_rmse = [rmse_values[i] for i in sorted_idx]

        colors = plt.cm.Greens(np.linspace(0.4, 0.8, 5))
        bars = ax.barh(range(5), top_rmse, color=colors)
        ax.set_yticks(range(5))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('RMSE')
        ax.set_title('Топ-5 лучших моделей')
        ax.invert_yaxis()

        for bar, val in zip(bars, top_rmse):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, 'error_distribution.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Сохранено: {fig_path}")

    def create_performance_summary_table(self, results):
        """Создание сводной таблицы результатов"""
        print("Создание сводной таблицы результатов...")

        summary_data = []
        for name, result in results.items():
            row = {
                'Модель': name,
                'RMSE': f"{result['rmse']:.4f}",
                'MAE': f"{result['mae']:.4f}",
                'Время (с)': f"{result['training_time']:.2f}"
            }

            # Добавляем количество параметров если есть
            if 'n_parameters' in result:
                row['Параметры'] = f"{result['n_parameters']:,}"

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('RMSE')

        # Создаём таблицу как изображение
        fig, ax = plt.subplots(figsize=(14, max(6, len(summary_df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Стилизация заголовков
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Чередование цветов строк
        for i in range(1, len(summary_df) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8F4F8')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')

        plt.title('Сводная таблица результатов моделей\n(отсортировано по RMSE)',
                 fontsize=14, fontweight='bold', pad=20)

        fig_path = os.path.join(self.figures_dir, 'performance_summary.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Сохраняем также в CSV
        csv_path = os.path.join(self.results_dir, 'performance_summary.csv')
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"  Сохранено: {fig_path}")
        print(f"  Сохранено: {csv_path}")

    def plot_training_history(self, history, model_name='Model'):
        """
        Визуализация истории обучения нейросети

        Args:
            history: словарь с историей (train_loss, val_loss, train_rmse, val_rmse)
            model_name: название модели
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        ax = axes[0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'{model_name}: История Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # RMSE
        ax = axes[1]
        ax.plot(epochs, history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
        ax.plot(epochs, history['val_rmse'], 'r-', label='Validation RMSE', linewidth=2)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{model_name}: История RMSE')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, f'training_history_{model_name.replace(" ", "_")}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  История обучения сохранена: {fig_path}")

    def plot_neural_vs_classical(self, classical_results, neural_results):
        """
        Сравнение классических и нейросетевых моделей

        Args:
            classical_results: результаты классических моделей
            neural_results: результаты нейросетевых моделей
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Подготовка данных
        classical_names = list(classical_results.keys())
        classical_rmse = [classical_results[n]['rmse'] for n in classical_names]

        neural_names = list(neural_results.keys())
        neural_rmse = [neural_results[n]['rmse'] for n in neural_names]

        # График 1: Сравнение RMSE
        ax = axes[0]
        x_classical = range(len(classical_names))
        x_neural = range(len(classical_names), len(classical_names) + len(neural_names))

        bars1 = ax.bar(x_classical, classical_rmse, label='Классические модели',
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x_neural, neural_rmse, label='Нейросетевые модели',
                      color='coral', alpha=0.8)

        all_names = classical_names + neural_names
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('RMSE')
        ax.set_title('Сравнение RMSE: Классические vs Нейросетевые модели')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # График 2: Средние значения
        ax = axes[1]
        categories = ['Классические\nмодели', 'Нейросетевые\nмодели']
        means = [np.mean(classical_rmse), np.mean(neural_rmse)]
        stds = [np.std(classical_rmse), np.std(neural_rmse)]
        mins = [np.min(classical_rmse), np.min(neural_rmse)]

        x = np.arange(len(categories))
        width = 0.25

        bars1 = ax.bar(x - width, means, width, label='Среднее RMSE',
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, mins, width, label='Лучшее RMSE',
                      color='mediumseagreen', alpha=0.8)
        bars3 = ax.bar(x + width, stds, width, label='Стд. отклонение',
                      color='coral', alpha=0.8)

        ax.set_ylabel('Значение')
        ax.set_title('Статистика по типам моделей')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Подписи значений
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, 'neural_vs_classical.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Сохранено: {fig_path}")

    def create_final_report_figure(self, all_results, dataset_info, best_model_name):
        """
        Создание финальной сводной фигуры для отчёта

        Args:
            all_results: все результаты
            dataset_info: информация о датасете
            best_model_name: название лучшей модели
        """
        fig = plt.figure(figsize=(16, 12))

        # Заголовок
        fig.suptitle('Рекомендательная система: Сводный отчёт\n'
                    f'Датасет: {dataset_info["n_users"]} пользователей, '
                    f'{dataset_info["n_items"]} товаров, '
                    f'{dataset_info["n_ratings"]} рейтингов',
                    fontsize=14, fontweight='bold')

        # Сетка 2x2
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Топ-10 моделей по RMSE
        ax1 = fig.add_subplot(gs[0, 0])
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['rmse'])[:10]
        names = [r[0] for r in sorted_results]
        rmse = [r[1]['rmse'] for r in sorted_results]

        colors = ['green' if n == best_model_name else 'steelblue' for n in names]
        bars = ax1.barh(range(len(names)), rmse, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel('RMSE')
        ax1.set_title('Топ-10 моделей по RMSE')
        ax1.invert_yaxis()

        for bar, val in zip(bars, rmse):
            ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)

        # 2. Распределение RMSE по типам моделей
        ax2 = fig.add_subplot(gs[0, 1])

        # Классифицируем модели
        neural_models = ['GMF', 'MLP', 'NCF', 'Wide & Deep', 'MLP Recommender',
                        'Neural Collaborative Filtering']
        ensemble_models = ['Ensemble', 'Stacking', 'Blending', 'Voting', 'Bagging',
                          'Average', 'Weighted']

        categories = {'Базовые': [], 'Ансамбли': [], 'Нейросети': []}

        for name, result in all_results.items():
            if any(nn in name for nn in neural_models):
                categories['Нейросети'].append(result['rmse'])
            elif any(en in name for en in ensemble_models):
                categories['Ансамбли'].append(result['rmse'])
            else:
                categories['Базовые'].append(result['rmse'])

        cat_names = []
        cat_means = []
        cat_mins = []

        for cat, values in categories.items():
            if values:
                cat_names.append(cat)
                cat_means.append(np.mean(values))
                cat_mins.append(np.min(values))

        x = np.arange(len(cat_names))
        width = 0.35

        ax2.bar(x - width/2, cat_means, width, label='Среднее', color='steelblue', alpha=0.8)
        ax2.bar(x + width/2, cat_mins, width, label='Лучшее', color='mediumseagreen', alpha=0.8)
        ax2.set_ylabel('RMSE')
        ax2.set_xticks(x)
        ax2.set_xticklabels(cat_names)
        ax2.set_title('RMSE по типам моделей')
        ax2.legend()

        # 3. Лучшая модель - детали
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')

        best_result = all_results[best_model_name]
        text = f"ЛУЧШАЯ МОДЕЛЬ\n\n"
        text += f"Название: {best_model_name}\n"
        text += f"RMSE: {best_result['rmse']:.4f}\n"
        text += f"MAE: {best_result['mae']:.4f}\n"
        text += f"Время обучения: {best_result['training_time']:.2f} с\n"

        if 'n_parameters' in best_result:
            text += f"Параметров: {best_result['n_parameters']:,}\n"

        ax3.text(0.5, 0.5, text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                family='monospace')

        # 4. Статистика датасета
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        stats_text = "СТАТИСТИКА ДАТАСЕТА\n\n"
        stats_text += f"Пользователей: {dataset_info['n_users']:,}\n"
        stats_text += f"Товаров: {dataset_info['n_items']:,}\n"
        stats_text += f"Рейтингов: {dataset_info['n_ratings']:,}\n"
        stats_text += f"Разреженность: {dataset_info['sparsity']*100:.2f}%\n"
        stats_text += f"Средний рейтинг: {dataset_info['rating_mean']:.2f}\n"
        stats_text += f"\nВсего моделей: {len(all_results)}"

        ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                family='monospace')

        fig_path = os.path.join(self.figures_dir, 'final_report.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Финальный отчёт сохранён: {fig_path}")
