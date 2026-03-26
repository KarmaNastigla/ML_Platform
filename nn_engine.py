"""
nn_engine.py — Движок нейронных сетей для Universal ML Platform.

Поддерживает три архитектуры для ТАБЛИЧНЫХ данных:
  1. MLPClassifier / MLPRegressor (sklearn)        — полносвязная сеть, нулевые зависимости
  2. TabNet (pytorch-tabnet)                        — трансформер для таблиц
  3. PyTorchMLP (torch)                             — простая кастомная нейросеть

Все модели:
  - Принимают тот же DataFrame что и UniversalMLEngine
  - Возвращают те же ключи метрик (Accuracy / R² и т.д.)
  - Имеют generate_human_explanation() для интерпретации
  - Поддерживают save_model() в тот же формат pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
import joblib
import warnings

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: препроцессор (одинаков для всех нейросетей)
# ──────────────────────────────────────────────────────────────────────────────
def _build_preprocessor(num_cols, cat_cols):
    """
    Строит ColumnTransformer аналогичный ml_engine.py.
    Нейросети особенно чувствительны к масштабу — StandardScaler обязателен

    """
    numeric_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),          # нормализация критична для нейросетей
    ])
    categorical_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_tf, num_cols),
        ('cat', categorical_tf, cat_cols),
    ])

def detect_task_type(y: pd.Series) -> str:
    """Та же логика что в UniversalMLEngine — копируем для независимости модуля"""
    if y.dtype in ['float64', 'float32']:
        return 'regression'
    if y.nunique() > 20:
        return 'regression'
    return 'classification'


# ──────────────────────────────────────────────────────────────────────────────
# 1. sklearn MLP
# ──────────────────────────────────────────────────────────────────────────────
class SklearnMLPEngine:
    """
    Многослойный персептрон через sklearn.
    Преимущества: нет новых зависимостей, работает везде, быстро.
    Ограничения: нет GPU, нет батч-обучения для больших данных.
    """
    def __init__(self, hidden_layers=(128, 64), max_iter=300, learning_rate_init=0.001):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.pipeline = None
        self.task_type = None
        self.features = []
        self.class_labels = None
        self.conf_matrix = None
        self.best_params = {
            'hidden_layers': hidden_layers,
            'max_iter': max_iter,
            'lr': learning_rate_init,
        }

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str):
        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.features = list(X.columns)
        self.task_type = detect_task_type(y)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        preprocessor = _build_preprocessor(num_cols, cat_cols)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.task_type == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                max_iter=self.max_iter,
                learning_rate_init=self.learning_rate_init,
                early_stopping=True,  # останавливаем при отсутствии прогресса
                validation_fraction=0.1,  # 10% train идёт на валидацию внутри
                random_state=42,
                verbose=False,
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                max_iter=self.max_iter,
                learning_rate_init=self.learning_rate_init,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                verbose=False,
            )

        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)

        return self._compute_metrics(y_test, preds, y)

    def _compute_metrics(self, y_test, preds, y_full):
        if self.task_type == 'classification':
            self.class_labels = sorted(y_full.unique().tolist())
            self.conf_matrix = confusion_matrix(y_test, preds, labels=self.class_labels)
            return {
                'Accuracy': round(accuracy_score(y_test, preds), 3),
                'Precision': round(precision_score(y_test, preds, average='macro', zero_division=0), 3),
                'Recall': round(recall_score(y_test, preds, average='macro', zero_division=0), 3),
            }
        else:
            self.class_labels = None
            self.conf_matrix = None
            return {
                'R²': round(r2_score(y_test, preds), 3),
                'MAE': round(mean_absolute_error(y_test, preds), 3),
                'RMSE': round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            }

    def generate_human_explanation(self):
        model = self.pipeline.named_steps['model']
        arch = " → ".join([str(self.hidden_layers[0])]
                          + [str(s) for s in self.hidden_layers[1:]])
        n_iter = getattr(model, 'n_iter_', self.max_iter)
        tl = "классификации" if self.task_type == 'classification' else "регрессии"
        return (
            f"Sklearn **MLP** (задача {tl}). "
            f"Архитектура: вход → **{arch}** → выход. "
            f"Обучение остановлено на итерации **{n_iter}** (early stopping). "
            f"Активация: ReLU. Оптимизатор: Adam (lr={self.learning_rate_init})."
        )

    def save_model(self, path="model.pkl"):
        joblib.dump({
            "model": self.pipeline, "features": self.features,
            "task_type": self.task_type, "class_labels": self.class_labels,
            "nn_type": "sklearn_mlp",
        }, path)


# ──────────────────────────────────────────────────────────────────────────────
# 2. TabNet (pytorch-tabnet)
# ──────────────────────────────────────────────────────────────────────────────
class TabNetEngine:
    """
    TabNet — трансформер для табличных данных (Arik & Pfister, Google, 2019).

    Ключевые идеи:
    - Sequential attention: на каждом шаге сеть выбирает какие признаки использовать
    - Feature importances из внимания — встроенная интерпретируемость
    - Обычно сопоставим с Gradient Boosting или лучше на больших датасетах

    Требует: pip install pytorch-tabnet

    """
    def __init__(self, n_steps=3, n_d=16, n_a=16, max_epochs=100, patience=15):
        # n_steps: количество шагов последовательного внимания
        # n_d, n_a: размерности пространств решений и внимания
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.max_epochs = max_epochs
        self.patience = patience       # early stopping: остановка если нет улучшения N эпох
        self.model = None
        self.task_type = None
        self.features = []
        self.class_labels = None
        self.conf_matrix = None
        self.feature_importances_ = None
        self.best_params = {
            'n_steps': n_steps, 'n_d': n_d, 'n_a': n_a,
            'max_epochs': max_epochs, 'patience': patience,
        }

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        except ImportError:
            raise ImportError(
                "pytorch-tabnet не установлен. Запусти: pip install pytorch-tabnet"
            )

        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.features = list(X.columns)
        self.task_type = detect_task_type(y)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Препроцессинг вручную — TabNet не принимает Pipeline sklearn
        preprocessor = _build_preprocessor(num_cols, cat_cols)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test2, y_val, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

        # Трансформируем данные в numpy-массивы — TabNet не принимает DataFrame
        X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
        X_val_t = preprocessor.transform(X_val).astype(np.float32)
        X_test_t = preprocessor.transform(X_test2).astype(np.float32)

        # Сохраняем препроцессор для инференса
        self._preprocessor = preprocessor

        if self.task_type == 'classification':
            # LabelEncoder: TabNet требует числовые метки от 0 до N-1
            self._le = LabelEncoder()
            y_train_enc = self._le.fit_transform(y_train)
            y_val_enc = self._le.transform(y_val)
            y_test_enc = self._le.transform(y_test2)
            self.class_labels = sorted(y.unique().tolist())

            self.model = TabNetClassifier(
                n_steps=self.n_steps, n_d=self.n_d, n_a=self.n_a,
                optimizer_fn=__import__('torch').optim.Adam,
                optimizer_params={'lr': 2e-3},
                scheduler_params={'gamma': 0.95, 'step_size': 20},
                scheduler_fn=__import__('torch').optim.lr_scheduler.StepLR,
                verbose=0,
                seed=42,
            )
            self.model.fit(
                X_train_t, y_train_enc,
                eval_set=[(X_val_t, y_val_enc)],
                eval_name=['val'],
                eval_metric=['accuracy'],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=256,
            )
            preds_enc = self.model.predict(X_test_t)
            preds = self._le.inverse_transform(preds_enc)
            y_test_orig = y_test2

        else:
            self._le = None
            y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
            y_val_np = y_val.values.reshape(-1, 1).astype(np.float32)

            self.model = TabNetRegressor(
                n_steps=self.n_steps, n_d=self.n_d, n_a=self.n_a,
                optimizer_fn=__import__('torch').optim.Adam,
                optimizer_params={'lr': 2e-3},
                verbose=0,
                seed=42,
            )
            self.model.fit(
                X_train_t, y_train_np,
                eval_set=[(X_val_t, y_val_np)],
                eval_name=['val'],
                eval_metric=['mse'],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=256,
            )
            preds = self.model.predict(X_test_t).ravel()
            y_test_orig = y_test2

            # Feature importances из механизма внимания TabNet
        self.feature_importances_ = self.model.feature_importances_

        return self._compute_metrics(y_test_orig, preds, y)

    def _compute_metrics(self, y_test, preds, y_full):
        if self.task_type == 'classification':
            self.conf_matrix = confusion_matrix(y_test, preds, labels=self.class_labels)
            return {
                'Accuracy': round(accuracy_score(y_test, preds), 3),
                'Precision': round(precision_score(y_test, preds, average='macro', zero_division=0), 3),
                'Recall': round(recall_score(y_test, preds, average='macro', zero_division=0), 3),
            }
        else:
            self.conf_matrix = None
            return {
                'R²': round(r2_score(y_test, preds), 3),
                'MAE': round(mean_absolute_error(y_test, preds), 3),
                'RMSE': round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            }

    def generate_human_explanation(self):
        tl = "классификации" if self.task_type == 'classification' else "регрессии"
        top_features = ""
        if self.feature_importances_ is not None and len(self.features) > 0:
            fi = sorted(zip(self.features, self.feature_importances_),
                        key=lambda x: x[1], reverse=True)
            top_features = (f" Внимание сети сосредоточено на **{fi[0][0]}**"
                            + (f" и **{fi[1][0]}**" if len(fi) > 1 else "") + ".")
        return (
            f"**TabNet** (задача {tl}). "
            f"Архитектура: {self.n_steps} шагов последовательного внимания, "
            f"n_d={self.n_d}, n_a={self.n_a}."
            f"{top_features} "
            f"TabNet выбирает признаки на каждом шаге — интерпретируемость встроена в архитектуру."
        )

    def save_model(self, path="model.pkl"):
        # TabNet сохраняется через joblib вместе с препроцессором
        joblib.dump({
            "tabnet_model": self.model,
            "preprocessor": self._preprocessor,
            "label_encoder": self._le,
            "features": self.features,
            "task_type": self.task_type,
            "class_labels": self.class_labels,
            "nn_type": "tabnet",
        }, path)


