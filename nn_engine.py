"""
nn_engine.py — Движок нейронных сетей для Universal ML Platform.

Поддерживает три архитектуры для табличных данных:

  1. SklearnMLPEngine  — sklearn MLPClassifier/Regressor
       + нет новых зависимостей; loss_curve_ для визуализации
       – нет GPU; медленнее PyTorch на больших данных

  2. PyTorchMLPEngine  — кастомный MLP на PyTorch
       + BatchNorm + Dropout; ReduceLROnPlateau; GPU-поддержка
       + epoch_callback для live-графика в Streamlit
       – требует pip install torch

  3. TabNetEngine      — трансформер для таблиц (Arik & Pfister, Google 2019)
       + встроенная интерпретируемость через sequential attention
       + feature_importances_ из механизма внимания
       – требует pip install pytorch-tabnet

Все три движка реализуют одинаковый интерфейс:
    engine.train_and_evaluate(df, target_col, epoch_callback=cb) → dict метрик
    engine.generate_human_explanation() → str
    engine.save_model(path)
    engine.train_history → {'train_loss': [...], 'val_loss': [...], 'n_iter': N}

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
    r2_score, mean_absolute_error, mean_squared_error,
)
import joblib
import warnings

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# Общие вспомогательные функции (используются всеми тремя движками)
# ══════════════════════════════════════════════════════════════════════════════
def _build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Строит стандартный ColumnTransformer для нейросетей.

    Нейросети ОСОБЕННО чувствительны к масштабу входных данных:
    без StandardScaler градиенты в разных нейронах могут отличаться на порядки,
    что приводит к нестабильному обучению и медленной сходимости.

    """
    numeric_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    categorical_tf = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_tf, num_cols),
        ('cat', categorical_tf, cat_cols),
    ])


def detect_task_type(y: pd.Series) -> str:
    """Определяет тип задачи по распределению целевой переменной"""
    if y.dtype in ['float64', 'float32']:
        return 'regression'
    if y.nunique() > 20:
        return 'regression'
    return 'classification'


def _compute_metrics_common(task_type, y_test, preds, y_full, class_labels):
    """
    Вычисляет метрики — общий код для всех трёх движков.
    Возвращает (metrics_dict, class_labels, conf_matrix)

    """
    if task_type == 'classification':
        labels = sorted(y_full.unique().tolist())
        cm = confusion_matrix(y_test, preds, labels=labels)
        metrics = {
            'Accuracy':  round(accuracy_score(y_test, preds), 3),
            'Precision': round(precision_score(
                y_test, preds, average='macro', zero_division=0), 3),
            'Recall':    round(recall_score(
                y_test, preds, average='macro', zero_division=0), 3),
        }
        return metrics, labels, cm
    else:
        metrics = {
            'R²':   round(r2_score(y_test, preds), 3),
            'MAE':  round(mean_absolute_error(y_test, preds), 3),
            'RMSE': round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
        }
        return metrics, None, None


# ══════════════════════════════════════════════════════════════════════════════
# 1. sklearn MLP
# ══════════════════════════════════════════════════════════════════════════════
class SklearnMLPEngine:
    """
    Многослойный персептрон через sklearn.neural_network.

    Преимущества:
      - нет новых зависимостей (sklearn уже установлен)
      - early_stopping=True останавливает обучение при отсутствии прогресса
      - loss_curve_ и validation_scores_ доступны для визуализации

    Ограничения:
      - нет GPU (только CPU)
      - нет BatchNorm, нет гибкого Dropout
      - для больших датасетов (>500k строк) медленнее PyTorch

    Параметры:
      hidden_layers:       кортеж с числом нейронов в каждом слое: (128,) или (128, 64)
      max_iter:            максимум итераций (эпох); early stopping может остановить раньше
      learning_rate_init:  начальный lr для Adam-подобного оптимизатора sklearn

    """
    def __init__(self, hidden_layers=(128, 64), max_iter=300, learning_rate_init=0.001):
        self.hidden_layers       = hidden_layers
        self.max_iter            = max_iter
        self.learning_rate_init  = learning_rate_init

        # Заполняются после train_and_evaluate():
        self.pipeline     = None   # Pipeline(preprocessor + MLPClassifier/MLPRegressor)
        self.task_type    = None
        self.features     = []
        self.class_labels = None
        self.conf_matrix  = None
        self.train_history = None
        self.best_params  = {
            'hidden_layers': hidden_layers,
            'max_iter':      max_iter,
            'lr':            learning_rate_init,
        }

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str,
                           epoch_callback=None) -> dict:
        """Обучает MLP и возвращает метрики"""
        df = df.dropna(subset=[target_col])
        X  = df.drop(columns=[target_col])
        y  = df[target_col]
        self.features  = list(X.columns)
        self.task_type = detect_task_type(y)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        preprocessor = _build_preprocessor(num_cols, cat_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        if self.task_type == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes = self.hidden_layers,
                max_iter           = self.max_iter,
                learning_rate_init = self.learning_rate_init,
                early_stopping     = True,
                validation_fraction= 0.1,
                random_state       = 42,
                verbose            = False,
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes = self.hidden_layers,
                max_iter           = self.max_iter,
                learning_rate_init = self.learning_rate_init,
                early_stopping     = True,
                validation_fraction= 0.1,
                random_state       = 42,
                verbose            = False,
            )

        # Pipeline включает препроцессор — предотвращает data leakage
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model',        model),
        ])
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)

        # ── Извлекаем историю обучения ────────────────────────────────────
        inner      = self.pipeline.named_steps['model']
        train_loss = list(getattr(inner, 'loss_curve_', []))

        val_scores = list(getattr(inner, 'validation_scores_', []))
        if val_scores:
            val_loss = [round(1.0 - float(s), 5) for s in val_scores]
        else:
            val_loss = train_loss[:]  # редкий edge case

        n_iter = getattr(inner, 'n_iter_', len(train_loss))
        self.train_history = {
            'train_loss': [round(float(v), 5) for v in train_loss],
            'val_loss':   val_loss,
            'n_iter':     n_iter,
        }

        metrics, self.class_labels, self.conf_matrix = _compute_metrics_common(
            self.task_type, y_test, preds, y, self.class_labels)
        return metrics

    def generate_human_explanation(self) -> str:
        inner   = self.pipeline.named_steps['model']
        arch    = " → ".join(str(s) for s in self.hidden_layers)
        n_iter  = getattr(inner, 'n_iter_', self.max_iter)
        tl      = "классификации" if self.task_type == 'classification' else "регрессии"
        stopped = n_iter < self.max_iter
        note    = (f"Early stopping на итерации **{n_iter}**."
                   if stopped else f"Обучение завершено за **{n_iter}** итераций.")
        return (
            f"Sklearn **MLP** ({tl}). "
            f"Архитектура: вход → **{arch}** → выход. "
            f"{note} Активация: ReLU. Оптимизатор: Adam (lr={self.learning_rate_init})."
        )

    def save_model(self, path: str = "model.pkl"):
        joblib.dump({
            "model":        self.pipeline,
            "features":     self.features,
            "task_type":    self.task_type,
            "class_labels": self.class_labels,
            "nn_type":      "sklearn_mlp",
        }, path)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TabNet (pytorch-tabnet)
# ══════════════════════════════════════════════════════════════════════════════
class TabNetEngine:
    """
    TabNet — трансформер для табличных данных (Arik & Pfister, Google Brain, 2019).

    Ключевые идеи TabNet:
    1. Sequential Attention: на каждом «шаге» сеть выбирает КАКИЕ признаки использовать.
       Это похоже на то, как человек последовательно смотрит на разные части данных.
    2. Feature Importances встроены в архитектуру (из матриц внимания) — не нужен SHAP.
    3. Обычно конкурирует с Gradient Boosting на больших датасетах (>10k строк).

    Параметры:
      n_steps:    количество шагов sequential attention (2–6). Больше = сложнее модель.
      n_d, n_a:   размерности пространств решений и внимания. Одинаковы для простоты.
      max_epochs: максимум эпох. Early stopping остановит раньше при patience.
      patience:   эпох без улучшения val_loss для early stopping.

    Требует: pip install pytorch-tabnet

    """
    def __init__(self, n_steps=3, n_d=16, n_a=16, max_epochs=100, patience=15):
        self.n_steps    = n_steps
        self.n_d        = n_d
        self.n_a        = n_a
        self.max_epochs = max_epochs
        self.patience   = patience

        self.model               = None
        self.task_type           = None
        self.features            = []
        self.class_labels        = None
        self.conf_matrix         = None
        self.feature_importances_= None  # из механизма внимания — для визуализации
        self.train_history       = None
        self._preprocessor       = None  # TabNet не принимает sklearn Pipeline
        self._le                 = None  # LabelEncoder для меток классов
        self.best_params = {
            'n_steps':    n_steps,
            'n_d':        n_d,
            'n_a':        n_a,
            'max_epochs': max_epochs,
            'patience':   patience,
        }

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str,
                           epoch_callback=None) -> dict:
        """
        Обучает TabNet.

        epoch_callback(epoch, max_epochs, train_loss, val_loss, train_hist, val_hist):
            Вызывается после каждой эпохи. TabNet не предоставляет epoch-level callback
            напрямую — история читается из self.model.history после обучения.
            Поэтому epoch_callback здесь вызывается один раз в конце

        """
        try:
            import torch
            _ = torch.tensor([1.0])  # реальная загрузка c10.dll (не просто импорт)
        except Exception as e:
            raise RuntimeError(
                f"PyTorch недоступен (DLL ошибка): {e}\n"
                "Исправление: pip uninstall torch -y && "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        except ImportError:
            raise ImportError("pytorch-tabnet не установлен: pip install pytorch-tabnet")

        df = df.dropna(subset=[target_col])
        X  = df.drop(columns=[target_col])
        y  = df[target_col]
        self.features  = list(X.columns)
        self.task_type = detect_task_type(y)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # TabNet не принимает sklearn Pipeline — трансформируем данные вручную в numpy
        preprocessor    = _build_preprocessor(num_cols, cat_cols)
        self._preprocessor = preprocessor

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        # Отдельный val-split для early stopping TabNet (нужен явный eval_set)
        X_val, X_test2, y_val, y_test2 = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42)

        # Трансформируем в numpy float32 (формат требуемый TabNet)
        X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
        X_val_t   = preprocessor.transform(X_val).astype(np.float32)
        X_test_t  = preprocessor.transform(X_test2).astype(np.float32)

        if self.task_type == 'classification':
            # LabelEncoder: TabNet требует метки от 0 до N-1 (не строки / не произвольные int)
            self._le     = LabelEncoder()
            y_train_enc  = self._le.fit_transform(y_train)
            y_val_enc    = self._le.transform(y_val)
            self.class_labels = sorted(y.unique().tolist())

            self.model = TabNetClassifier(
                n_steps          = self.n_steps,
                n_d              = self.n_d,
                n_a              = self.n_a,
                optimizer_fn     = __import__('torch').optim.Adam,
                optimizer_params = {'lr': 2e-3},
                scheduler_params = {'gamma': 0.95, 'step_size': 20},
                scheduler_fn     = __import__('torch').optim.lr_scheduler.StepLR,
                verbose          = 0,                # без логов в консоль
                seed             = 42,
            )
            self.model.fit(
                X_train_t, y_train_enc,
                eval_set    = [(X_val_t, y_val_enc)],
                eval_name   = ['val'],
                eval_metric = ['accuracy'],
                max_epochs  = self.max_epochs,
                patience    = self.patience,
                batch_size  = 256,
            )
            preds      = self._le.inverse_transform(self.model.predict(X_test_t))
            y_test_orig = y_test2

        else:
            self._le      = None
            y_train_np    = y_train.values.reshape(-1, 1).astype(np.float32)
            y_val_np      = y_val.values.reshape(-1, 1).astype(np.float32)

            self.model = TabNetRegressor(
                n_steps          = self.n_steps,
                n_d              = self.n_d,
                n_a              = self.n_a,
                optimizer_fn     = __import__('torch').optim.Adam,
                optimizer_params = {'lr': 2e-3},
                verbose          = 0,
                seed             = 42,
            )
            self.model.fit(
                X_train_t, y_train_np,
                eval_set    = [(X_val_t, y_val_np)],
                eval_name   = ['val'],
                eval_metric = ['mse'],
                max_epochs  = self.max_epochs,
                patience    = self.patience,
                batch_size  = 256,
            )
            preds       = self.model.predict(X_test_t).ravel()
            y_test_orig = y_test2

        self.feature_importances_ = self.model.feature_importances_

        # Читаем историю лосса из internal history TabNet
        try:
            hist    = self.model.history
            train_l = [round(float(v), 5) for v in hist.get('loss', [])]
            val_key = 'val_accuracy' if self.task_type == 'classification' else 'val_mse'
            val_raw = hist.get(val_key, [])
            val_l = ([round(1.0 - float(v), 5) for v in val_raw]
                     if self.task_type == 'classification'
                     else [round(float(v), 5) for v in val_raw])
            self.train_history = {
                'train_loss': train_l,
                'val_loss':   val_l,
                'n_iter':     len(train_l),
            }
        except Exception:
            self.train_history = None

        # Вызываем epoch_callback ОДИН РАЗ в конце (TabNet не даёт per-epoch hook)
        if epoch_callback is not None and self.train_history:
            tl = self.train_history['train_loss']
            vl = self.train_history['val_loss']
            epoch_callback(len(tl), self.max_epochs, tl[-1] if tl else 0,
                           vl[-1] if vl else 0, tl, vl)

        metrics, self.class_labels, self.conf_matrix = _compute_metrics_common(
            self.task_type, y_test_orig, preds, y, self.class_labels)
        return metrics

    def generate_human_explanation(self) -> str:
        tl = "классификации" if self.task_type == 'classification' else "регрессии"
        top = ""
        if self.feature_importances_ is not None and len(self.features) > 0:
            fi = sorted(zip(self.features, self.feature_importances_),
                        key=lambda x: x[1], reverse=True)
            top = (f" Внимание сети сосредоточено на **{fi[0][0]}**"
                   + (f" и **{fi[1][0]}**" if len(fi) > 1 else "") + ".")
        return (
            f"**TabNet** ({tl}). "
            f"{self.n_steps} шагов sequential attention, n_d={self.n_d}, n_a={self.n_a}."
            f"{top} Интерпретируемость встроена в архитектуру."
        )

    def save_model(self, path: str = "model.pkl"):
        joblib.dump({
            "tabnet_model":  self.model,
            "preprocessor":  self._preprocessor,
            "label_encoder": self._le,
            "features":      self.features,
            "task_type":     self.task_type,
            "class_labels":  self.class_labels,
            "nn_type":       "tabnet",
        }, path)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PyTorch MLP
# ══════════════════════════════════════════════════════════════════════════════
class PyTorchMLPEngine:
    """
    Кастомная полносвязная нейросеть на PyTorch.

    Архитектура каждого скрытого слоя:
        Linear → BatchNorm1d → ReLU → Dropout

    Параметры:
      hidden_dims: список размеров скрытых слоёв: (256, 128, 64) = 3 слоя
      dropout:     вероятность обнуления нейрона при обучении (0 = выключен)
      lr:          начальный learning rate для Adam
      max_epochs:  максимум эпох (early stopping остановит раньше)
      patience:    эпох без улучшения val_loss для early stopping

    """
    # Как часто (в эпохах) вызывать epoch_callback.
    CALLBACK_EVERY = 5

    def __init__(self, hidden_dims=(256, 128, 64), dropout=0.3,
                 lr=1e-3, max_epochs=100, patience=15, batch_size=256):
        self.hidden_dims = hidden_dims
        self.dropout     = dropout
        self.lr          = lr
        self.max_epochs  = max_epochs
        self.patience    = patience
        self.batch_size  = batch_size

        self._preprocessor = None  # sklearn ColumnTransformer
        self._model        = None  # nn.Sequential
        self._le           = None  # LabelEncoder для меток классов
        self.task_type     = None
        self.features      = []
        self.class_labels  = None
        self.conf_matrix   = None
        self.train_history = None
        self.best_params = {
            'hidden_dims': hidden_dims,
            'dropout':     dropout,
            'lr':          lr,
            'max_epochs':  max_epochs,
        }

    def _build_torch_model(self, input_dim: int, output_dim: int):
        """
        Создаёт nn.Sequential из блоков [Linear → BatchNorm → ReLU → Dropout].

        Порядок слоёв внутри блока имеет значение:
          Linear → BatchNorm (перед активацией) → ReLU → Dropout (после активации)
          Этот порядок — стандарт для «Pre-activation» архитектур.

        input_dim:  число признаков после препроцессинга
        output_dim: число классов (классификация) или 1 (регрессия)

        """
        try:
            import torch.nn as nn
        except (ImportError, OSError) as e:
            raise RuntimeError(f"PyTorch недоступен: {e}")

        layers   = []
        prev_dim = input_dim
        for dim in self.hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),    # аффинное преобразование: y = Wx + b
                nn.BatchNorm1d(dim),
                nn.ReLU(),                   # нелинейность: max(0, x). Простая, не saturates
                nn.Dropout(self.dropout),
            ]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def train_and_evaluate(self, df: pd.DataFrame, target_col: str,
                           epoch_callback=None) -> dict:
        """Полный цикл обучения PyTorch MLP с early stopping"""
        try:
            import torch
            _ = torch.tensor([1.0])   # реальная загрузка c10.dll
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except OSError as e:
            raise RuntimeError(
                f"PyTorch DLL ошибка (c10.dll): {e}\n"
                "Причина: CUDA-версия torch без GPU/драйверов.\n"
                "Исправление: pip uninstall torch -y && "
                "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        except ImportError:
            raise ImportError(
                "PyTorch не установлен:\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )

        # Автоматический выбор GPU или CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        df = df.dropna(subset=[target_col])
        X  = df.drop(columns=[target_col])
        y  = df[target_col]
        self.features  = list(X.columns)
        self.task_type = detect_task_type(y)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self._preprocessor = _build_preprocessor(num_cols, cat_cols)

        # Два сплита и внутри test: val/test2, чтобы val-split не пересекался с финальным test2.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_val, X_test2, y_val, y_test2 = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42)

        # Трансформируем в numpy float32 (единственный тип принимаемый torch.tensor)
        X_train_t = self._preprocessor.fit_transform(X_train).astype(np.float32)
        X_val_t   = self._preprocessor.transform(X_val).astype(np.float32)
        X_test_t  = self._preprocessor.transform(X_test2).astype(np.float32)
        input_dim = X_train_t.shape[1]  # число признаков после препроцессинга

        # ── Подготовка тензоров по типу задачи ────────────────────────────
        if self.task_type == 'classification':
            self._le     = LabelEncoder()
            y_train_enc  = self._le.fit_transform(y_train).astype(np.int64)
            y_val_enc    = self._le.transform(y_val).astype(np.int64)
            y_test_enc   = self._le.transform(y_test2).astype(np.int64)
            self.class_labels = sorted(y.unique().tolist())
            n_classes = len(self.class_labels)

            model     = self._build_torch_model(input_dim, n_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            train_ds  = TensorDataset(
                torch.tensor(X_train_t).to(device),
                torch.tensor(y_train_enc).to(device),
            )
            val_X  = torch.tensor(X_val_t).to(device)
            val_y  = torch.tensor(y_val_enc).to(device)
            test_X = torch.tensor(X_test_t).to(device)

        else:
            self._le      = None
            y_train_np    = y_train.values.astype(np.float32).reshape(-1, 1)
            y_val_np      = y_val.values.astype(np.float32).reshape(-1, 1)

            model     = self._build_torch_model(input_dim, 1).to(device)
            criterion = nn.MSELoss()
            train_ds  = TensorDataset(
                torch.tensor(X_train_t).to(device),
                torch.tensor(y_train_np).to(device),
            )
            val_X  = torch.tensor(X_val_t).to(device)
            val_y  = torch.tensor(y_val_np).to(device)
            test_X = torch.tensor(X_test_t).to(device)

        self._model = model

        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Adam с weight_decay = L2-регуляризация: эффективно как AdamW
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-4)
        # ReduceLROnPlateau: если val_loss не улучшается 5 эпох → lr *= 0.5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        best_val_loss    = float('inf')
        patience_counter = 0
        best_state       = None  # словарь весов лучшей эпохи

        # ── ИНИЦИАЛИЗАЦИЯ СПИСКОВ ДО ЦИКЛА (критично!) ────────────────────
        history_train_loss: list = []
        history_val_loss:   list = []

        # ── Цикл обучения ─────────────────────────────────────────────────
        for epoch in range(self.max_epochs):

            # Фаза обучения: model.train() включает Dropout и BatchNorm в режим обучения
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()       # сбрасываем накопленные градиенты
                out  = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()             # вычисляем градиенты (backprop)
                optimizer.step()            # обновляем веса: w -= lr * grad

            # Фаза оценки: model.eval() выключает Dropout, BatchNorm переходит в inference-mode
            model.eval()
            with torch.no_grad():  # no_grad экономит память: не строим вычислительный граф
                val_out        = model(val_X)
                val_loss_val   = criterion(val_out, val_y).item()
                train_out_full = model(train_ds.tensors[0])
                train_loss_val = criterion(train_out_full, train_ds.tensors[1]).item()

            history_train_loss.append(round(float(train_loss_val), 5))
            history_val_loss.append(round(float(val_loss_val), 5))

            # ReduceLROnPlateau следит за val_loss и снижает lr при плато
            scheduler.step(val_loss_val)

            # ── Логика early stopping ──────────────────────────────────────
            if val_loss_val < best_val_loss:
                best_val_loss    = val_loss_val
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break  # выходим из цикла → обучение завершено досрочно

            # ── Вызов epoch_callback для live UI ──────────────────────────
            if epoch_callback is not None:
                is_last   = (patience_counter >= self.patience or
                             epoch == self.max_epochs - 1)
                should_cb = (epoch % self.CALLBACK_EVERY == 0) or is_last
                if should_cb:
                    epoch_callback(
                        epoch + 1,               # текущая эпоха (1-based)
                        self.max_epochs,
                        train_loss_val,
                        val_loss_val,
                        history_train_loss[:],   # копия (избегаем мутации в callback)
                        history_val_loss[:],
                    )

        # Восстанавливаем лучшие веса (из эпохи с минимальным val_loss)
        if best_state is not None:
            model.load_state_dict(best_state)

        self.train_history = {
            'train_loss': history_train_loss,
            'val_loss':   history_val_loss,
            'n_iter':     len(history_train_loss),
        }

        # ── Инференс на тестовой выборке ─────────────────────────────────
        model.eval()
        with torch.no_grad():
            test_out = model(test_X)

        if self.task_type == 'classification':
            pred_idx = test_out.argmax(dim=1).cpu().numpy()
            preds    = self._le.inverse_transform(pred_idx)
            y_test_orig = y_test2
        else:
            preds       = test_out.cpu().numpy().ravel()
            y_test_orig = y_test2

        metrics, self.class_labels, self.conf_matrix = _compute_metrics_common(
            self.task_type, y_test_orig, preds, y, self.class_labels)
        return metrics

    def generate_human_explanation(self) -> str:
        tl     = "классификации" if self.task_type == 'classification' else "регрессии"
        arch   = " → ".join(str(d) for d in self.hidden_dims)
        hist   = self.train_history or {}
        n_iter = hist.get('n_iter', self.max_epochs)
        try:
            import torch
            dev = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        except Exception:
            dev = "CPU"
        note = (f"Early stopping на эпохе **{n_iter}**."
                if n_iter < self.max_epochs
                else f"Обучение завершено за **{n_iter}** эпох.")
        return (
            f"**PyTorch MLP** ({tl}). "
            f"Архитектура: вход → **{arch}** → выход. "
            f"BatchNorm + Dropout({self.dropout}) на каждом слое. "
            f"Adam lr={self.lr}. {note} Устройство: {dev}."
        )

    def save_model(self, path: str = "model.pkl"):
        """
        Сохраняет веса модели + метаданные.

        Особенность PyTorch: сохраняем state_dict (только веса), а не весь модуль.
        При загрузке нужно воссоздать архитектуру и загрузить веса.
        CPU-перенос (.cpu()) гарантирует совместимость на машинах без GPU

        """
        try:
            import torch
            state = self._model.state_dict() if self._model else None
            if state:
                state = {k: v.cpu() for k, v in state.items()}
        except Exception:
            state = None

        joblib.dump({
            "model_state":  state,
            "model_config": {"hidden_dims": self.hidden_dims, "dropout": self.dropout},
            "preprocessor": self._preprocessor,
            "label_encoder": self._le,
            "features":     self.features,
            "task_type":    self.task_type,
            "class_labels": self.class_labels,
            "nn_type":      "pytorch_mlp",
        }, path)