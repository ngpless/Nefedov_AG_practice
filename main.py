"""
Рекомендательная система с применением нейронных сетей

Главный скрипт проекта для производственной практики.
Тема: "Применение нейронных сетей для рекомендации товаров и услуг"

Выполняет полный цикл:
1. Загрузка и анализ данных (MovieLens 100K)
2. Обучение классических моделей рекомендаций
3. Обучение нейросетевых моделей (GMF, MLP, NCF, Wide&Deep)
4. Применение ансамблевых методов
5. Эксперименты с топологией нейросетей
6. Визуализация и сохранение результатов

Автор: Нефедов Алексей Геннадьевич
Направление: 09.03.03 Прикладная информатика
Профиль: Искусственный интеллект и анализ данных
Дата: 2025

Запуск:
    python main.py
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Импорт модулей проекта
from data_processing import RecommenderDataProcessor
from models import RecommenderModelTrainer
from ensemble_methods import RecommenderEnsembleTrainer
from visualization import RecommenderVisualizer
from utils import save_recommendation_results, create_directories

# Проверяем доступность PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch не установлен. Нейросетевые модели будут пропущены.")
    print("Установите: pip install torch")


def main():
    """
    Главная функция - полный пайплайн рекомендательной системы
    """
    print("="*70)
    print("РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА С НЕЙРОННЫМИ СЕТЯМИ")
    print("Производственная практика - Эксплуатационная практика")
    print("Нефедов Алексей Геннадьевич")
    print("="*70)

    # Создаём директории
    create_directories()
    start_time = time.time()

    try:
        # =====================================================================
        # ЭТАП 1: ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ
        # =====================================================================
        print("\n" + "="*70)
        print("ЭТАП 1: ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ")
        print("="*70)

        processor = RecommenderDataProcessor(data_dir='data')

        # Загрузка MovieLens 100K (автоматическая загрузка из интернета)
        train_data, test_data, user_item_matrix, dataset_info = processor.process_data(
            dataset='movielens-100k',
            test_size=0.2
        )

        n_users = len(processor.user_mapping)
        n_items = len(processor.item_mapping)

        print(f"\nДанные загружены:")
        print(f"  Пользователей: {n_users}")
        print(f"  Товаров: {n_items}")
        print(f"  Обучающая выборка: {len(train_data)} рейтингов")
        print(f"  Тестовая выборка: {len(test_data)} рейтингов")

        # =====================================================================
        # ЭТАП 2: АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ДАННЫХ
        # =====================================================================
        print("\n" + "="*70)
        print("ЭТАП 2: АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ДАННЫХ")
        print("="*70)

        visualizer = RecommenderVisualizer(results_dir='results')
        visualizer.create_data_analysis_plots(train_data, user_item_matrix, dataset_info)

        # =====================================================================
        # ЭТАП 3: ОБУЧЕНИЕ КЛАССИЧЕСКИХ МОДЕЛЕЙ
        # =====================================================================
        print("\n" + "="*70)
        print("ЭТАП 3: ОБУЧЕНИЕ КЛАССИЧЕСКИХ МОДЕЛЕЙ РЕКОМЕНДАЦИЙ")
        print("="*70)

        model_trainer = RecommenderModelTrainer()
        base_results = model_trainer.train_base_models(train_data, test_data, user_item_matrix)

        print(f"\nОбучено {len(base_results)} классических моделей")

        # =====================================================================
        # ЭТАП 4: ПРИМЕНЕНИЕ АНСАМБЛЕВЫХ МЕТОДОВ
        # =====================================================================
        print("\n" + "="*70)
        print("ЭТАП 4: ПРИМЕНЕНИЕ АНСАМБЛЕВЫХ МЕТОДОВ")
        print("="*70)

        ensemble_trainer = RecommenderEnsembleTrainer()
        ensemble_results = ensemble_trainer.train_ensemble_models(
            train_data, test_data, user_item_matrix, model_trainer.trained_models,
            n_users, n_items
        )

        print(f"\nОбучено {len(ensemble_results)} ансамблевых моделей")

        # =====================================================================
        # ЭТАП 5: ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ
        # =====================================================================
        print("\n" + "="*70)
        print("ЭТАП 5: ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ")
        print("="*70)

        optimized_results = ensemble_trainer.optimize_hyperparameters(
            train_data, test_data, user_item_matrix
        )

        # Оптимизация весов ансамбля (веса подбираются по валидации,
        # оценка — на тесте; контекст подготовлен в train_ensemble_models)
        optimized_weights_result = ensemble_trainer.optimize_ensemble_weights(test_data)
        optimized_results['Optimized Ensemble Weights'] = optimized_weights_result

        print(f"\nОптимизировано {len(optimized_results)} моделей")

        # =====================================================================
        # ЭТАП 6: ОБУЧЕНИЕ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ
        # =====================================================================
        neural_results = {}

        if PYTORCH_AVAILABLE:
            print("\n" + "="*70)
            print("ЭТАП 6: ОБУЧЕНИЕ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ")
            print("="*70)

            from neural_models import NeuralModelTrainer

            neural_trainer = NeuralModelTrainer()
            neural_results = neural_trainer.train_all_neural_models(
                train_data, test_data, n_users, n_items,
                epochs=50,
                batch_size=256
            )

            # Визуализация истории обучения для каждой модели
            for name, result in neural_results.items():
                if 'history' in result:
                    visualizer.plot_training_history(result['history'], name)

            # Сохранение весов обученных моделей (требование программы ГИА:
            # модель должна загружаться из файла и воспроизводить метрики)
            os.makedirs(os.path.join('results', 'models'), exist_ok=True)
            for name, result in neural_results.items():
                if 'model' in result:
                    safe_name = name.replace(' ', '_').replace('&', 'and')
                    weights_path = os.path.join('results', 'models', f'{safe_name}.pt')
                    torch.save(result['model'].state_dict(), weights_path)
                    print(f"  Веса сохранены: {weights_path}")

            # Лучшая нейросетевая модель дублируется в корень как model.pt
            best_neural_name = min(neural_results.items(), key=lambda x: x[1]['rmse'])[0]
            best_neural_model = neural_results[best_neural_name]['model']
            torch.save(best_neural_model.state_dict(), 'model.pt')
            print(f"  Лучшая нейросетевая модель ({best_neural_name}) сохранена: model.pt")

            print(f"\nОбучено {len(neural_results)} нейросетевых моделей")

        else:
            print("\n" + "="*70)
            print("ЭТАП 6: НЕЙРОСЕТЕВЫЕ МОДЕЛИ ПРОПУЩЕНЫ (PyTorch не установлен)")
            print("="*70)

        # =====================================================================
        # ЭТАП 7: ЭКСПЕРИМЕНТЫ С ТОПОЛОГИЕЙ
        # =====================================================================
        if PYTORCH_AVAILABLE:
            print("\n" + "="*70)
            print("ЭТАП 7: ЭКСПЕРИМЕНТЫ С ТОПОЛОГИЕЙ НЕЙРОСЕТЕЙ")
            print("="*70)

            from topology_experiments import TopologyExperiments
            from sklearn.model_selection import train_test_split

            # Архитектура выбирается по ВАЛИДАЦИОННОЙ выборке (из train),
            # тест в выборе конфигурации не участвует
            topo_train, topo_val = train_test_split(
                train_data, test_size=0.15, random_state=42,
                stratify=train_data['rating']
            )

            topology_exp = TopologyExperiments(n_users, n_items)
            topology_results = topology_exp.run_all_experiments(
                topo_train, topo_val,
                epochs=30,
                batch_size=256
            )

            # Визуализация экспериментов
            topology_exp.plot_all_experiments()
            topology_exp.print_summary()

            # Получаем лучшую конфигурацию
            best_config = topology_exp.get_best_configuration()
            print(f"\nРекомендуемая конфигурация: {best_config}")

        # =====================================================================
        # ЭТАП 8: ОБЪЕДИНЕНИЕ И ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
        # =====================================================================
        print("\n" + "="*70)
        print("ЭТАП 8: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("="*70)

        # Объединяем все результаты
        all_results = {
            **base_results,
            **ensemble_results,
            **optimized_results,
            **neural_results
        }

        # Создаём визуализации
        visualizer.create_results_visualization(all_results)

        # Сравнение классических и нейросетевых моделей
        if neural_results:
            classical_results = {**base_results, **ensemble_results, **optimized_results}
            visualizer.plot_neural_vs_classical(classical_results, neural_results)

        # Финальный отчёт
        best_model_name = min(all_results.items(), key=lambda x: x[1]['rmse'])[0]
        visualizer.create_final_report_figure(all_results, dataset_info, best_model_name)

        # =====================================================================
        # ЭТАП 9: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
        # =====================================================================
        print("\n" + "="*70)
        print("ЭТАП 9: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*70)

        save_recommendation_results(all_results, dataset_info)

        # =====================================================================
        # ИТОГОВАЯ СТАТИСТИКА
        # =====================================================================
        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
        print("="*70)

        print(f"\nИтоговая статистика:")
        print(f"  Всего моделей обучено: {len(all_results)}")
        print(f"  - Классических: {len(base_results)}")
        print(f"  - Ансамблевых: {len(ensemble_results)}")
        print(f"  - Оптимизированных: {len(optimized_results)}")
        print(f"  - Нейросетевых: {len(neural_results)}")

        print(f"\nЛучшая модель: {best_model_name}")
        print(f"  RMSE: {all_results[best_model_name]['rmse']:.4f}")
        print(f"  MAE: {all_results[best_model_name]['mae']:.4f}")

        print(f"\nОбщее время выполнения: {total_time:.1f} секунд ({total_time/60:.1f} минут)")
        print(f"\nРезультаты сохранены в папке 'results/'")
        print(f"Графики сохранены в папке 'results/figures/'")

        # Топ-5 моделей
        print("\n" + "-"*50)
        print("ТОП-5 ЛУЧШИХ МОДЕЛЕЙ:")
        print("-"*50)
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['rmse'])[:5]
        for i, (name, result) in enumerate(sorted_models, 1):
            print(f"  {i}. {name}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")

        return True

    except Exception as e:
        print(f"\nОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nПроверьте:")
        print("  1. Установлены ли все зависимости: pip install -r requirements.txt")
        print("  2. Есть ли доступ к интернету для загрузки датасета")
        return False


def run_quick_test():
    """
    Быстрый тест на синтетических данных (без загрузки MovieLens)
    """
    print("="*70)
    print("БЫСТРЫЙ ТЕСТ НА СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("="*70)

    create_directories()

    processor = RecommenderDataProcessor(data_dir='data')
    train_data, test_data, user_item_matrix, dataset_info = processor.process_data(
        dataset='synthetic',
        test_size=0.2
    )

    model_trainer = RecommenderModelTrainer()
    base_results = model_trainer.train_base_models(train_data, test_data, user_item_matrix)

    print("\nБыстрый тест завершён!")
    print(f"Обучено моделей: {len(base_results)}")

    best = min(base_results.items(), key=lambda x: x[1]['rmse'])
    print(f"Лучшая модель: {best[0]} (RMSE: {best[1]['rmse']:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Рекомендательная система с нейронными сетями')
    parser.add_argument('--quick', action='store_true', help='Быстрый тест на синтетических данных')

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        success = main()
        if success:
            print("\n" + "="*70)
            print("Система готова для защиты практики!")
            print("Откройте notebooks/report.ipynb для детального отчёта")
            print("="*70)
        else:
            print("\nНеобходимо исправить ошибки перед продолжением")
            sys.exit(1)
