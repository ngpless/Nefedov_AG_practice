# Воспроизведение экспериментов

## Полный эксперимент

```bash
python main.py
```

Обучает 20 моделей (7 классических, 4 нейросетевые, 9 ансамблевых и
оптимизированных), выполняет эксперименты с топологией и сохраняет
результаты в `results/`. Время выполнения на CPU — около 12 минут.

## Ускоренные варианты

```bash
python main.py --skip-topology          # без экспериментов с топологией
python main.py --epochs 10              # меньше эпох обучения нейросетей
python main.py --batch-size 512         # крупнее батч
python main.py --quick                  # быстрый тест на синтетических данных
```

## Ожидаемые контрольные значения

При фиксированном seed (42) полный запуск воспроизводит метрики
на тестовой выборке:

| Модель | RMSE | MAE |
|---|---|---|
| Stacking Ensemble | 0,9246 | 0,7304 |
| Blending Ensemble | 0,9246 | 0,7304 |
| Wide & Deep | 0,9573 | 0,7593 |
| GMF | 0,9677 | 0,7528 |

## Проверка обученной модели без обучения

Веса лучшей нейросетевой модели сохранены в `model.pt`
(нумерованные копии — `model1.pt`…`model4.pt`). Загрузка:

```python
import torch, sys
sys.path.insert(0, "src")
from neural_models import WideAndDeep

model = WideAndDeep(943, 1682, 32, [64, 32])
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

## Модульные тесты

```bash
pip install -r requirements-dev.txt
python -m pytest
```
