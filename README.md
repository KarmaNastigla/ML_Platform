# 🚀 Universal ML Experimentation Platform

**Платформа полного цикла машинного обучения:** загрузи датасет, очисти данные, обучи модель, разберись почему она так решает — и задеплой. Всё в одном интерфейсе, без написания кода.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.5-6C63FF)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E)

---

## Зачем этот проект

Большинство ML-инструментов решают одну часть задачи: MLflow — трекинг экспериментов, AutoSklearn — подбор модели, FastAPI — деплой. Чтобы связать их вместе, нужно самому писать склейку.

Эта платформа закрывает весь цикл в единый UX: от сырого CSV до задеплоенного API — без переключения инструментов. При этом под каждым решением видна логика: какие признаки важны, почему модель предсказала именно так, насколько стабильна оценка качества.

---

## Возможности

### 📊 Анализ и очистка данных
- Автоматический отчёт о качестве: пропуски, выбросы, типы данных для каждого признака
- Удаление столбцов по настраиваемому порогу пропусков — с превью что будет удалено
- Сглаживание выбросов с настраиваемым множителем IQR (по умолчанию 1.5, рекомендация прямо в UI)
- Заполнение пропусков на уровне каждого столбца: Медиана / Среднее / Мода / Константа
- Создание новых признаков по математической формуле: `(SibSp + Parch) * Fare`
- Интерактивные графики: корреляционная матрица, гистограммы, scatter, boxplot
- Счётчик изменений датасета (Δ строк / Δ столбцов от исходного)

### ⚙️ Обучение классических ML-моделей
- **Автоопределение типа задачи** — `float`-таргет или >20 уникальных значений → регрессия; иначе → классификация
- **Bayesian оптимизация гиперпараметров** через Optuna — находит хорошие параметры за 20–50 итераций
- **K-Fold Cross-Validation** — опциональный режим с настраиваемым числом фолдов
- **4 алгоритма:** Random Forest, Gradient Boosting, Logistic Regression / Ridge, Voting Ensemble
- **📈 Learning Curve** — диагностика переобучения/недообучения с автоматической интерпретацией

### 🧠 Нейронные сети (ИНС)
- **sklearn MLP** — полносвязная сеть, нулевые новые зависимости; кривая лосса из `loss_curve_`
- **PyTorch MLP** — кастомная архитектура с BatchNorm и Dropout, early stopping, GPU-поддержка
- **TabNet** — трансформер для таблиц с встроенной интерпретируемостью через механизм внимания
- **📉 Loss Curve** — график Train Loss / Val Loss по эпохам с маркером лучшей эпохи для всех трёх архитектур
- Автоматическая интерпретация кривых: переобучение / недообучение / хороший баланс
- Сравнение ИНС vs классический ML на общих метриках

### 🔍 Объяснимость (XAI)
- **SHAP Waterfall** — для любого предсказания: какой признак на сколько сдвинул результат от базового уровня
- **График важности признаков** — топ-15 по impurity decrease (деревья) или абсолютному весу коэффициента
- **Confusion Matrix** — с нормировкой по строкам и двойной аннотацией: числа + процент

### 🧪 Встроенный тестер
- Умные виджеты: `selectbox` для категориальных и дискретных числовых признаков, `number_input` для непрерывных
- Дефолтные значения — медианы из обучающей выборки
- SHAP-объяснение строится автоматически для введённых данных

### 📜 История экспериментов
- Журнал всех запусков: классические модели и нейросети в одной таблице
- Визуальное сравнение любой метрики по всем экспериментам
- **Генератор воспроизводимого `.py`-скрипта** — теперь для классического ML и нейросетей отдельно

### 🛠 Деплой
- One-click генерация FastAPI-сервиса + Dockerfile + docker-compose.yml
- Продакшен-образ ~300 МБ (без Streamlit/Optuna/SHAP)

---

## Быстрый старт

```bash
# 1. Клонируй репозиторий
git clone https://github.com/KarmaNastigla/ML_Platform
cd ML_Platform

# 2. Создай виртуальное окружение (опционально)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# 3. Установи зависимости
pip install -r requirements.txt

# 4. (Опционально) Нейросети
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-версия
pip install pytorch-tabnet                                             # TabNet

# 5. Запусти платформу
streamlit run app.py
```

Браузер откроется на `http://localhost:8501`. Загрузи любой CSV-файл в боковое меню.

---

## Структура проекта

```
universal-ml-platform/
│
├── app.py                    # Streamlit UI: 4 вкладки + sidebar
├── ml_engine.py              # ML-ядро: Pipeline, Optuna, SHAP, Learning Curve
├── nn_engine.py              # Нейросети: sklearn MLP, PyTorch MLP, TabNet
├── deploy_generator.py       # Генерирует FastAPI + Dockerfile в папку deploy/
├── requirements.txt          # Dev-зависимости
│
├── tests/
│   └── test_ml_engine.py     # 37 юнит-тестов (pytest)
│
├── .streamlit/
│   └── config.toml           # maxUploadSize = 1024 МБ
│
└── deploy/                   # ← создаётся автоматически
    ├── api.py                # FastAPI: /health и /predict
    ├── Dockerfile
    ├── docker-compose.yml
    ├── model.pkl
    └── requirements.txt      # Только: fastapi, uvicorn, sklearn, pandas, joblib
```

---

## Кривые обучения

### Классические модели (вкладка ⚙️)
Learning Curve показывает качество модели (Accuracy или R²) при увеличении размера обучающей выборки от 20% до 100%:

- 🟢 **Хороший баланс** — Train–Val разрыв ≤ 0.05
- 🔴 **Переобучение** — разрыв > 0.15
- 🟡 **Недообучение** — Val Accuracy < 0.6

### Нейросети (вкладка 🧠)
Loss Curve показывает Train Loss и Val Loss по эпохам:

- Вертикальная зелёная линия — лучшая эпоха (минимум Val Loss)
- Early stopping срабатывает если Val Loss не улучшается N эпох
- Для sklearn MLP: `loss_curve_` (train) и `1 - validation_scores_` (val proxy)
- Для PyTorch MLP и TabNet: реальные лоссы по каждой эпохе

---

## Генератор Python-скриптов

Во вкладке **"📜 История экспериментов"** выбери любой запуск → **"⬇️ Сгенерировать и скачать .py скрипт"**.

Платформа определяет тип эксперимента и генерирует соответствующий скрипт:

**Классический ML** (`ml_solution_*.py`):
```
1. Загрузка данных
2. EDA (describe, value_counts, корреляции, выбросы IQR)
3. Очистка данных    ← только шаги применённые в UI
4. Подготовка признаков (ColumnTransformer)
5. Обучение          ← параметры Optuna вшиты напрямую
6. Оценка
7. Сохранение модели
```

**Нейронная сеть** (`nn_solution_*.py`):
```
1. Загрузка данных
2. Препроцессинг (StandardScaler + OrdinalEncoder)
3. Обучение нейросети:
   - sklearn MLP:  MLPClassifier/MLPRegressor + matplotlib loss curve
   - PyTorch MLP:  полная архитектура MLP + цикл обучения + early stopping + loss curve
   - TabNet:       TabNetClassifier/TabNetRegressor + feature importances
4. Оценка
5. Сохранение
```

---

## Нейросети: детали архитектур

| Архитектура | Зависимость | Особенности |
|-------------|-------------|-------------|
| sklearn MLP | sklearn (уже установлен) | Быстро, без новых пакетов. `loss_curve_` для визуализации |
| PyTorch MLP | `pip install torch` | BatchNorm + Dropout, ReduceLROnPlateau scheduler, GPU-поддержка |
| TabNet | `pip install pytorch-tabnet` | Sequential attention — встроенная интерпретируемость. Конкурент Gradient Boosting |

### Проблемы с PyTorch на Windows (CUDA DLL)
Если видишь ошибку `OSError: ... c10.dll`:
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Причина: установлена CUDA-версия torch без GPU/драйверов. CPU-версия работает везде.

---

## Поддерживаемые алгоритмы

| Алгоритм | Задача | Оптимизируемые параметры |
|----------|--------|--------------------------|
| Random Forest | классификация / регрессия | `n_estimators`, `max_depth` |
| Gradient Boosting | классификация / регрессия | `n_estimators`, `learning_rate`, `max_depth` |
| Logistic Regression → Ridge | классификация / регрессия¹ | `C` / `alpha` |
| Voting Ensemble | классификация / регрессия | фиксированные (RF + GB + LR/Ridge) |
| sklearn MLP | классификация / регрессия | `hidden_layers`, `max_iter` |
| PyTorch MLP | классификация / регрессия | `hidden_dims`, `dropout`, `lr`, `max_epochs` |
| TabNet | классификация / регрессия | `n_steps`, `n_d`, `n_a`, `max_epochs` |

¹ При регрессии Logistic Regression автоматически заменяется на Ridge.

---

## Метрики

| Задача | Метрики |
|--------|---------|
| Классификация | Accuracy, Precision (macro), Recall (macro), Confusion Matrix |
| Регрессия | R², MAE, RMSE |
| При включённом CV (классика) | + mean ± std по всем фолдам |

---

## Технические решения

### `learning_curve` из sklearn — почему нужна отдельная оценка
Стандартный `cross_val_score` даёт одно число: метрику на 100% данных. `learning_curve` запускает оценку на 5 подвыборках (20%–100%) — видно не только итоговое качество, но и как оно растёт. Если train-кривая высокая, а val-кривая низкая — классический сигнал переобучения.

### `loss_curve_` и `validation_scores_` в sklearn MLP
sklearn MLP не предоставляет epoch-by-epoch историю как PyTorch. Но при `early_stopping=True`:
- `loss_curve_` — лосс на обучении по итерациям (всегда доступен)
- `validation_scores_` — accuracy или r2 на val (10% от train)

Поскольку `validation_scores_` — это score (выше лучше), а на графике нужен loss — конвертируем: `val_loss = 1 - val_score`.

### Optuna вместо GridSearch
GridSearch с 3 параметрами по 10 значений = 1000 обучений. Optuna (TPE Sampler) — байесовская оптимизация: каждый trial учитывает результаты предыдущих. На Titanic это разница между ~2 минутами и ~20 секундами при сопоставимом качестве.

### SHAP: TreeExplainer для деревьев, LinearExplainer для линейных
`TreeExplainer` использует внутреннюю структуру дерева — точный и быстрый. Универсальный `KernelExplainer` перебирает подмножества признаков — работает минутами. Выбор explainer'а автоматический.

### Лог очистки для воспроизводимости
Каждое нажатие кнопки очистки записывается в `cleaning_log`. При обучении лог фиксируется. Генератор скрипта переводит лог в исполняемый Python-код — шаг за шагом, в том порядке, в котором они применялись.

---

## Деплой модели

```bash
cd deploy
docker-compose up --build
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"Pclass": 1, "Sex": "female", "Age": 28, "Fare": 100}}'
# → {"prediction": 1, "task_type": "classification"}
```

---

## Тесты

```bash
pytest tests/ -v
```

37 тестов: автоопределение типа задачи, все 4 алгоритма на классификации и регрессии, SHAP для каждой комбинации, CV-режим, сохранение/загрузка модели.

---

## Требования

- Python 3.10+
- 4 ГБ RAM (рекомендуется 8 ГБ для датасетов >100k строк)
- PyTorch: опционально, CPU-версия работает без GPU

---

## Лицензия

[MIT](LICENSE) — свободное использование и модификация.