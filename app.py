import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import datetime
from ml_engine import UniversalMLEngine
from nn_engine import SklearnMLPEngine

_PYTORCH_ERROR = None
try:
    import torch as _torch
    _torch.tensor([1.0])
    from nn_engine import PyTorchMLPEngine
    PYTORCH_AVAILABLE = True
except Exception as _e:
    PYTORCH_AVAILABLE = False
    _PYTORCH_ERROR = f"{type(_e).__name__}: {_e}"

if PYTORCH_AVAILABLE:
    try:
        from nn_engine import TabNetEngine
        TABNET_AVAILABLE = True
    except Exception:
        TABNET_AVAILABLE = False
else:
    TABNET_AVAILABLE = False


st.set_page_config(page_title="Universal ML Platform", layout="wide")

# ==========================================
# ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЙ
# ==========================================
defaults = {
    'is_trained':         False,
    'train_df':           None,
    'custom_features':    [],
    'task_type':          None,
    'metrics':            {},
    'explanation':        '',
    'trained_model_name': '',
    'features':           [],
    'conf_matrix':        None,
    'class_labels':       None,
    'experiment_history': [],
    'cleaning_log':       [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================
# ГЕНЕРАТОР PYTHON-СКРИПТА (классический ML)
# ==========================================
def generate_script(record: dict, cleaning_log: list, dataset_filename: str) -> str:
    model_name  = record.get("Модель", "Random Forest")
    target_col  = record.get("Target", "target")
    task_type   = record.get("Задача", "classification")
    cv_mode     = record.get("CV", "hold-out")
    n_trials    = record.get("Optuna trials", 20)
    best_params = record.get("best_params", {})
    metrics = {k: v for k, v in record.items()
               if k not in {"⏰ Время", "Модель", "Задача", "Target",
                            "CV", "Optuna trials", "best_params"}}

    use_cv   = cv_mode != "hold-out"
    cv_folds = int(cv_mode.replace("-fold", "")) if use_cv else 5

    lines = [
        "# =============================================================",
        f"# Автоматически сгенерированный скрипт",
        f"# Модель:  {model_name}",
        f"# Target:  {target_col}",
        f"# Задача:  {task_type}",
        f"# Метрики: {metrics}",
        f"# Дата:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "# =============================================================",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "from sklearn.model_selection import train_test_split, cross_val_score",
        "from sklearn.model_selection import StratifiedKFold, KFold",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import OrdinalEncoder, StandardScaler",
    ]

    if task_type == "classification":
        lines += ["from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix"]
    else:
        lines += ["from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error"]

    model_imports = {
        "Random Forest":       ("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor", ),
        "Gradient Boosting":   ("from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor", ),
        "Logistic Regression": ("from sklearn.linear_model import LogisticRegression, Ridge", ),
        "Ансамбль (Ensemble)": (
            "from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,",
            "    GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor)",
            "from sklearn.linear_model import LogisticRegression, Ridge",
        ),
    }
    for imp_line in model_imports.get(model_name, []):
        lines.append(imp_line)

    lines += ["import joblib", "import optuna", ""]

    lines += [
        "# =============================================================",
        "# 1. ЗАГРУЗКА ДАННЫХ",
        "# =============================================================",
        f"df = pd.read_csv('{dataset_filename}')",
        f"print(f'Датасет загружен: {{df.shape[0]}} строк, {{df.shape[1]}} столбцов')",
        "",
    ]

    lines += [
        "# =============================================================",
        "# 2. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)",
        "# =============================================================",
        "print('--- Форма датасета ---'); print(df.shape)",
        "print('--- Типы данных ---'); print(df.dtypes.to_string())",
        "print('--- Первые 5 строк ---'); print(df.head().to_string())",
        "print('--- Описательная статистика ---'); print(df.describe().round(2).to_string())",
        "",
        "print('--- Пропущенные значения ---')",
        "_miss = df.isnull().sum(); _miss = _miss[_miss > 0].sort_values(ascending=False)",
        "print(_miss.to_string() if len(_miss) > 0 else 'Пропусков нет')",
        "",
        "_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()",
        "for _col in _cat_cols:",
        "    print(f'--- {_col} (value_counts) ---'); print(df[_col].value_counts().to_string())",
        "",
        "_num_cols = df.select_dtypes(include='number').columns.tolist()",
        "if len(_num_cols) > 1:",
        "    print('--- Корреляции ---'); print(df[_num_cols].corr().round(2).to_string())",
        "",
        "print('--- Выбросы (правило 1.5 x IQR) ---')",
        "for _col in _num_cols:",
        "    _q1, _q3 = df[_col].quantile(0.25), df[_col].quantile(0.75); _iqr = _q3 - _q1",
        "    _n_out = int(((df[_col] < _q1-1.5*_iqr)|(df[_col] > _q3+1.5*_iqr)).sum())",
        "    if _n_out > 0: print(f'  {_col}: {_n_out} выбросов')",
        "",
    ]

    if cleaning_log:
        lines += [
            "# =============================================================",
            "# 3. ОЧИСТКА И ПРЕДОБРАБОТКА ДАННЫХ",
            "# =============================================================",
        ]
        for step in cleaning_log:
            op = step["op"]
            if op == "drop_columns":
                cols = step["columns"]
                lines += [f"df = df.drop(columns={cols})", ""]
            elif op == "clip_outliers":
                cols = step["columns"]; k_mult = step["iqr_mult"]
                lines += [
                    f"for col in {cols}:",
                    f"    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)",
                    f"    IQR = Q3 - Q1",
                    f"    df[col] = df[col].clip(lower=Q1 - {k_mult}*IQR, upper=Q3 + {k_mult}*IQR)",
                    "",
                ]
            elif op == "fill_missing":
                methods = step["methods"]; constants = step["constants"]
                for col, method in methods.items():
                    if method == "Медиана":
                        lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
                    elif method == "Среднее":
                        lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())")
                    elif method == "Мода":
                        lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])")
                    elif method == "Константа":
                        val = constants.get(col, 0)
                        val_repr = f"'{val}'" if isinstance(val, str) else str(val)
                        lines.append(f"df['{col}'] = df['{col}'].fillna({val_repr})")
                lines.append("")
            elif op == "feature_engineering":
                lines += [f"df['{step['name']}'] = df.eval('{step['formula']}')", ""]
            elif op == "reset":
                lines += [f"df = pd.read_csv('{dataset_filename}')", ""]
    else:
        lines += ["# Очистка не применялась", ""]

    section_offset = 1 if cleaning_log else 0

    lines += [
        "# =============================================================",
        f"# {3 + section_offset}. ПОДГОТОВКА ПРИЗНАКОВ",
        "# =============================================================",
        f"df = df.dropna(subset=['{target_col}'])",
        "",
        f"X = df.drop(columns=['{target_col}'])",
        f"y = df['{target_col}']",
        "",
        "num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()",
        "cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()",
        "",
        "numeric_transformer = Pipeline(steps=[",
        "    ('imputer', SimpleImputer(strategy='median')),",
        "    ('scaler', StandardScaler()),",
        "])",
        "categorical_transformer = Pipeline(steps=[",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),",
        "    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),",
        "])",
        "preprocessor = ColumnTransformer(transformers=[",
        "    ('num', numeric_transformer, num_cols),",
        "    ('cat', categorical_transformer, cat_cols),",
        "])",
        "",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42)",
        "",
    ]

    lines += [
        "# =============================================================",
        f"# {4 + section_offset}. ОБУЧЕНИЕ МОДЕЛИ",
        "# =============================================================",
    ]

    if model_name == "Ансамбль (Ensemble)":
        if task_type == "classification":
            lines += [
                "final_model = VotingClassifier(estimators=[",
                "    ('rf', RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)),",
                "    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),",
                "    ('lr', LogisticRegression(C=1.0, max_iter=500, random_state=42)),",
                "], voting='soft')",
            ]
        else:
            lines += [
                "final_model = VotingRegressor(estimators=[",
                "    ('rf',    RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)),",
                "    ('gb',    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),",
                "    ('ridge', Ridge(alpha=1.0)),",
                "])",
            ]
    elif best_params and "Инфо" not in best_params:
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in best_params.items())
        model_classes = {
            ("Random Forest",       "classification"): "RandomForestClassifier",
            ("Random Forest",       "regression"):     "RandomForestRegressor",
            ("Gradient Boosting",   "classification"): "GradientBoostingClassifier",
            ("Gradient Boosting",   "regression"):     "GradientBoostingRegressor",
            ("Logistic Regression", "classification"): "LogisticRegression",
            ("Logistic Regression", "regression"):     "Ridge",
        }
        cls = model_classes.get((model_name, task_type), "RandomForestClassifier")
        extra = ", max_iter=1000" if cls == "LogisticRegression" else ""
        lines += [f"final_model = {cls}({params_str}{extra}, random_state=42)"]
    else:
        cv_scoring = "accuracy" if task_type == "classification" else "r2"
        lines += [
            "optuna.logging.set_verbosity(optuna.logging.WARNING)",
            "def objective(trial):",
        ]
        if model_name == "Random Forest":
            clf_or_reg = "Classifier" if task_type == "classification" else "Regressor"
            lines += [
                f"    model = RandomForest{clf_or_reg}(",
                "        n_estimators=trial.suggest_int('n_estimators', 50, 300),",
                "        max_depth=trial.suggest_int('max_depth', 3, 15), random_state=42)",
            ]
        elif model_name == "Gradient Boosting":
            cls = "GradientBoostingClassifier" if task_type == "classification" else "GradientBoostingRegressor"
            lines += [
                f"    model = {cls}(",
                "        n_estimators=trial.suggest_int('n_estimators', 50, 300),",
                "        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),",
                "        max_depth=trial.suggest_int('max_depth', 3, 10), random_state=42)",
            ]
        cv_splitter = ("StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
                       if task_type == "classification"
                       else "KFold(n_splits=5, shuffle=True, random_state=42)")
        lines += [
            "    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])",
            f"    scores = cross_val_score(pipe, X, y, cv={cv_splitter}, scoring='{cv_scoring}', n_jobs=-1)",
            "    return scores.mean()",
            "study = optuna.create_study(direction='maximize')",
            f"study.optimize(objective, n_trials={n_trials})",
            "print('Лучшие параметры:', study.best_params)",
        ]

    lines += [""]
    lines += [
        "pipeline = Pipeline(steps=[",
        "    ('preprocessor', preprocessor),",
        "    ('model', final_model),",
        "])",
        "",
    ]

    if use_cv:
        cv_splitter_code = (
            f"StratifiedKFold(n_splits={cv_folds}, shuffle=True, random_state=42)"
            if task_type == "classification"
            else f"KFold(n_splits={cv_folds}, shuffle=True, random_state=42)"
        )
        cv_scoring = "accuracy" if task_type == "classification" else "r2"
        lines += [
            f"cv_scores = cross_val_score(pipeline, X, y, cv={cv_splitter_code},",
            f"                             scoring='{cv_scoring}', n_jobs=-1)",
            f"print(f'CV {cv_scoring}: {{cv_scores.mean():.3f}} ± {{cv_scores.std():.3f}}')",
            "pipeline.fit(X, y)",
        ]
    else:
        lines += ["pipeline.fit(X_train, y_train)", "y_pred = pipeline.predict(X_test)"]

    lines += [
        "",
        "# =============================================================",
        f"# {5 + section_offset}. ОЦЕНКА",
        "# =============================================================",
    ]
    if task_type == "classification":
        lines += [
            "y_pred = pipeline.predict(X_test)",
            "print('Accuracy: ', round(accuracy_score(y_test, y_pred), 3))",
            "print('Precision:', round(precision_score(y_test, y_pred, average='macro', zero_division=0), 3))",
            "print('Recall:   ', round(recall_score(y_test, y_pred, average='macro', zero_division=0), 3))",
            "print('Confusion matrix:'); print(confusion_matrix(y_test, y_pred))",
        ]
    else:
        lines += [
            "y_pred = pipeline.predict(X_test)",
            "print('R²:  ', round(r2_score(y_test, y_pred), 3))",
            "print('MAE: ', round(mean_absolute_error(y_test, y_pred), 3))",
            "print('RMSE:', round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3))",
        ]

    lines += [
        "",
        "# =============================================================",
        f"# {6 + section_offset}. СОХРАНЕНИЕ МОДЕЛИ",
        "# =============================================================",
        "joblib.dump({",
        "    'model':     pipeline,",
        f"    'features':  list(X.columns),",
        f"    'task_type': '{task_type}',",
        "}, 'model.pkl')",
        "print('Модель сохранена в model.pkl')",
    ]

    return "\n".join(lines)


# ==========================================
# ГЕНЕРАТОР PYTHON-СКРИПТА (нейронные сети)
# ==========================================
def generate_nn_script(record: dict, dataset_filename: str) -> str:
    """
    Генерирует воспроизводимый .py-скрипт для экспериментов с нейросетями.
    Поддерживает: sklearn MLP, PyTorch MLP, TabNet.
    """
    model_name  = record.get("Модель", "ИНС: sklearn MLP")
    target_col  = record.get("Target", "target")
    task_type   = record.get("Задача", "classification")
    best_params = record.get("best_params", {})
    metrics     = {k: v for k, v in record.items()
                   if k not in {"⏰ Время", "Модель", "Задача", "Target",
                                "CV", "Optuna trials", "best_params", "cleaning_log"}}

    # "ИНС: sklearn MLP" → "sklearn MLP"
    nn_type = model_name.replace("ИНС: ", "").strip()

    lines = [
        "# =============================================================",
        f"# Автоматически сгенерированный скрипт (Нейронная сеть)",
        f"# Архитектура: {nn_type}",
        f"# Target:      {target_col}",
        f"# Задача:      {task_type}",
        f"# Метрики:     {metrics}",
        f"# Дата:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "# =============================================================",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import OrdinalEncoder, StandardScaler",
    ]

    if task_type == "classification":
        lines += ["from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix"]
    else:
        lines += ["from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error"]

    if nn_type == "sklearn MLP":
        if task_type == "classification":
            lines += ["from sklearn.neural_network import MLPClassifier"]
        else:
            lines += ["from sklearn.neural_network import MLPRegressor"]

    elif nn_type == "PyTorch MLP":
        lines += [
            "import torch",
            "import torch.nn as nn",
            "from torch.utils.data import DataLoader, TensorDataset",
            "from sklearn.preprocessing import LabelEncoder",
        ]

    elif nn_type == "TabNet":
        if task_type == "classification":
            lines += ["from pytorch_tabnet.tab_model import TabNetClassifier"]
        else:
            lines += ["from pytorch_tabnet.tab_model import TabNetRegressor"]
        lines += ["from sklearn.preprocessing import LabelEncoder"]

    lines += ["import joblib", ""]

    # ── Загрузка данных ─────────────────────────────────────────────────
    lines += [
        "# =============================================================",
        "# 1. ЗАГРУЗКА ДАННЫХ",
        "# =============================================================",
        f"df = pd.read_csv('{dataset_filename}')",
        f"df = df.dropna(subset=['{target_col}'])",
        f"print(f'Датасет: {{df.shape[0]}} строк × {{df.shape[1]}} столбцов')",
        "",
        f"X = df.drop(columns=['{target_col}'])",
        f"y = df['{target_col}']",
        "self_features = list(X.columns)",
        "",
        "num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()",
        "cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()",
        "",
    ]

    # ── Препроцессинг ───────────────────────────────────────────────────
    lines += [
        "# =============================================================",
        "# 2. ПРЕПРОЦЕССИНГ",
        "# =============================================================",
        "numeric_tf = Pipeline(steps=[",
        "    ('imputer', SimpleImputer(strategy='median')),",
        "    ('scaler', StandardScaler()),",
        "])",
        "categorical_tf = Pipeline(steps=[",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),",
        "    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),",
        "])",
        "preprocessor = ColumnTransformer(transformers=[",
        "    ('num', numeric_tf, num_cols),",
        "    ('cat', categorical_tf, cat_cols),",
        "])",
        "",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42)",
        "",
    ]

    # ── Обучение ────────────────────────────────────────────────────────
    lines += [
        "# =============================================================",
        "# 3. ОБУЧЕНИЕ НЕЙРОСЕТИ",
        "# =============================================================",
    ]

    if nn_type == "sklearn MLP":
        hidden = best_params.get('hidden_layers', (128, 64))
        max_iter = best_params.get('max_iter', 300)
        lr_val = best_params.get('lr', 0.001)
        cls = "MLPClassifier" if task_type == "classification" else "MLPRegressor"
        lines += [
            f"model = {cls}(",
            f"    hidden_layer_sizes={hidden},",
            f"    max_iter={max_iter},",
            f"    learning_rate_init={lr_val},",
            "    early_stopping=True,",
            "    validation_fraction=0.1,",
            "    random_state=42,",
            ")",
            "pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])",
            "pipeline.fit(X_train, y_train)",
            "",
            "# Кривая лосса",
            "import matplotlib.pyplot as plt",
            "inner = pipeline.named_steps['model']",
            "plt.figure(figsize=(10, 4))",
            "plt.plot(inner.loss_curve_, label='Train Loss')",
            "if hasattr(inner, 'validation_scores_'):",
            "    plt.plot([1-s for s in inner.validation_scores_], label='Val Loss (1-score)', linestyle='--')",
            "plt.xlabel('Итерация'); plt.ylabel('Loss')",
            "plt.title(f'Кривая обучения sklearn MLP (эпох: {inner.n_iter_})')",
            "plt.legend(); plt.tight_layout(); plt.savefig('loss_curve.png', dpi=120)",
            "print('График сохранён: loss_curve.png')",
            "",
            "y_pred = pipeline.predict(X_test)",
        ]

    elif nn_type == "PyTorch MLP":
        hidden_dims  = best_params.get('hidden_dims', (256, 128, 64))
        dropout_val  = best_params.get('dropout', 0.3)
        lr_pt        = best_params.get('lr', 0.001)
        max_epochs   = best_params.get('max_epochs', 100)

        lines += [
            "# ── Архитектура PyTorch MLP ─────────────────────────────────",
            "class MLP(nn.Module):",
            "    def __init__(self, input_dim, output_dim):",
            "        super().__init__()",
            "        layers = []",
            f"        hidden_dims = {list(hidden_dims)}",
            "        prev = input_dim",
            "        for dim in hidden_dims:",
            f"            layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout({dropout_val})]",
            "            prev = dim",
            "        layers.append(nn.Linear(prev, output_dim))",
            "        self.net = nn.Sequential(*layers)",
            "    def forward(self, x): return self.net(x)",
            "",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "print(f'Устройство: {device}')",
            "",
            "# Препроцессинг → numpy",
            "X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)",
            "X_test_t  = preprocessor.transform(X_test).astype(np.float32)",
            "X_val_t, X_test2_t, y_val, y_test2 = (",
            "    X_test_t[:len(X_test_t)//2], X_test_t[len(X_test_t)//2:],",
            "    y_test.iloc[:len(X_test_t)//2],  y_test.iloc[len(X_test_t)//2:]",
            ")",
            "",
        ]

        if task_type == "classification":
            lines += [
                "le = LabelEncoder()",
                "y_train_enc = le.fit_transform(y_train).astype(np.int64)",
                "y_val_enc   = le.transform(y_val).astype(np.int64)",
                "n_classes   = len(le.classes_)",
                "",
                "model = MLP(X_train_t.shape[1], n_classes).to(device)",
                "criterion = nn.CrossEntropyLoss()",
                "",
                "train_ds = TensorDataset(torch.tensor(X_train_t).to(device), torch.tensor(y_train_enc).to(device))",
                f"loader   = DataLoader(train_ds, batch_size=256, shuffle=True)",
                f"optimizer = torch.optim.Adam(model.parameters(), lr={lr_pt}, weight_decay=1e-4)",
                "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)",
                "",
                "train_losses, val_losses = [], []",
                "best_val, patience_counter, best_state = float('inf'), 0, None",
                f"for epoch in range({max_epochs}):",
                "    model.train()",
                "    for X_b, y_b in loader:",
                "        optimizer.zero_grad()",
                "        loss = criterion(model(X_b), y_b)",
                "        loss.backward(); optimizer.step()",
                "    model.eval()",
                "    with torch.no_grad():",
                "        val_loss = criterion(model(torch.tensor(X_val_t).to(device)), torch.tensor(y_val_enc).to(device)).item()",
                "        tr_loss  = criterion(model(train_ds.tensors[0]), train_ds.tensors[1]).item()",
                "    train_losses.append(round(tr_loss, 5)); val_losses.append(round(val_loss, 5))",
                "    scheduler.step(val_loss)",
                "    if val_loss < best_val:",
                "        best_val = val_loss; patience_counter = 0",
                "        best_state = {k: v.clone() for k, v in model.state_dict().items()}",
                "    else:",
                "        patience_counter += 1",
                f"        if patience_counter >= 15: print(f'Early stopping на эпохе {{epoch+1}}'); break",
                "",
                "if best_state: model.load_state_dict(best_state)",
                "",
                "# График лосса",
                "import matplotlib.pyplot as plt",
                "plt.figure(figsize=(10, 4))",
                "plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss', linestyle='--')",
                "plt.xlabel('Эпоха'); plt.ylabel('Loss'); plt.title('PyTorch MLP: кривые обучения')",
                "plt.legend(); plt.tight_layout(); plt.savefig('loss_curve.png', dpi=120)",
                "print('График сохранён: loss_curve.png')",
                "",
                "model.eval()",
                "with torch.no_grad():",
                "    pred_idx = model(torch.tensor(X_test2_t).to(device)).argmax(dim=1).cpu().numpy()",
                "y_pred = le.inverse_transform(pred_idx)",
                "y_test = y_test2",
            ]
        else:
            lines += [
                "y_train_np = y_train.values.astype(np.float32).reshape(-1,1)",
                "",
                "model = MLP(X_train_t.shape[1], 1).to(device)",
                "criterion = nn.MSELoss()",
                "train_ds = TensorDataset(torch.tensor(X_train_t).to(device), torch.tensor(y_train_np).to(device))",
                f"loader   = DataLoader(train_ds, batch_size=256, shuffle=True)",
                f"optimizer = torch.optim.Adam(model.parameters(), lr={lr_pt}, weight_decay=1e-4)",
                "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)",
                "",
                "train_losses, val_losses = [], []",
                "best_val, patience_counter, best_state = float('inf'), 0, None",
                "y_val_np = y_val.values.astype(np.float32).reshape(-1,1)",
                f"for epoch in range({max_epochs}):",
                "    model.train()",
                "    for X_b, y_b in loader:",
                "        optimizer.zero_grad()",
                "        loss = criterion(model(X_b), y_b)",
                "        loss.backward(); optimizer.step()",
                "    model.eval()",
                "    with torch.no_grad():",
                "        val_loss = criterion(model(torch.tensor(X_val_t).to(device)), torch.tensor(y_val_np).to(device)).item()",
                "        tr_loss  = criterion(model(train_ds.tensors[0]), train_ds.tensors[1]).item()",
                "    train_losses.append(round(tr_loss, 5)); val_losses.append(round(val_loss, 5))",
                "    scheduler.step(val_loss)",
                "    if val_loss < best_val:",
                "        best_val = val_loss; patience_counter = 0",
                "        best_state = {k: v.clone() for k, v in model.state_dict().items()}",
                "    else:",
                "        patience_counter += 1",
                f"        if patience_counter >= 15: print(f'Early stopping на эпохе {{epoch+1}}'); break",
                "",
                "if best_state: model.load_state_dict(best_state)",
                "import matplotlib.pyplot as plt",
                "plt.figure(figsize=(10,4))",
                "plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss', linestyle='--')",
                "plt.xlabel('Эпоха'); plt.ylabel('MSE Loss'); plt.title('PyTorch MLP: кривые обучения')",
                "plt.legend(); plt.tight_layout(); plt.savefig('loss_curve.png', dpi=120)",
                "print('График сохранён: loss_curve.png')",
                "",
                "model.eval()",
                "with torch.no_grad():",
                "    y_pred = model(torch.tensor(X_test2_t).to(device)).cpu().numpy().ravel()",
                "y_test = y_test2",
            ]

    elif nn_type == "TabNet":
        n_steps  = best_params.get('n_steps', 3)
        n_d_val  = best_params.get('n_d', 16)
        n_a_val  = best_params.get('n_a', 16)
        max_ep   = best_params.get('max_epochs', 100)
        patience_tb = best_params.get('patience', 15)
        cls_name = "TabNetClassifier" if task_type == "classification" else "TabNetRegressor"

        lines += [
            "# TabNet не принимает sklearn Pipeline — трансформируем вручную",
            "X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)",
            "X_val_raw, X_test2_raw, y_val, y_test2 = train_test_split(",
            "    X_test, y_test, test_size=0.5, random_state=42)",
            "X_val_t   = preprocessor.transform(X_val_raw).astype(np.float32)",
            "X_test_t  = preprocessor.transform(X_test2_raw).astype(np.float32)",
            "",
        ]

        if task_type == "classification":
            lines += [
                "le = LabelEncoder()",
                "y_train_enc = le.fit_transform(y_train)",
                "y_val_enc   = le.transform(y_val)",
                "",
                f"model = TabNetClassifier(",
                f"    n_steps={n_steps}, n_d={n_d_val}, n_a={n_a_val},",
                "    optimizer_fn=__import__('torch').optim.Adam,",
                "    optimizer_params={'lr': 2e-3},",
                "    verbose=1, seed=42,",
                ")",
                "model.fit(",
                "    X_train_t, y_train_enc,",
                "    eval_set=[(X_val_t, y_val_enc)],",
                "    eval_name=['val'], eval_metric=['accuracy'],",
                f"    max_epochs={max_ep}, patience={patience_tb}, batch_size=256,",
                ")",
                "",
                "preds_enc = model.predict(X_test_t)",
                "y_pred    = le.inverse_transform(preds_enc)",
                "y_test    = y_test2",
                "",
                "# Важность признаков",
                "fi = model.feature_importances_",
                "fi_pairs = sorted(zip(self_features, fi), key=lambda x: x[1], reverse=True)",
                "print('Топ-5 признаков по attention:')",
                "for name, score in fi_pairs[:5]: print(f'  {name}: {score:.4f}')",
            ]
        else:
            lines += [
                "y_train_np = y_train.values.reshape(-1,1).astype(np.float32)",
                "y_val_np   = y_val.values.reshape(-1,1).astype(np.float32)",
                "",
                f"model = TabNetRegressor(",
                f"    n_steps={n_steps}, n_d={n_d_val}, n_a={n_a_val},",
                "    optimizer_fn=__import__('torch').optim.Adam,",
                "    optimizer_params={'lr': 2e-3},",
                "    verbose=1, seed=42,",
                ")",
                "model.fit(",
                "    X_train_t, y_train_np,",
                "    eval_set=[(X_val_t, y_val_np)],",
                "    eval_name=['val'], eval_metric=['mse'],",
                f"    max_epochs={max_ep}, patience={patience_tb}, batch_size=256,",
                ")",
                "",
                "y_pred = model.predict(X_test_t).ravel()",
                "y_test = y_test2",
            ]

    # ── Оценка ──────────────────────────────────────────────────────────
    lines += [
        "",
        "# =============================================================",
        "# 4. ОЦЕНКА",
        "# =============================================================",
    ]
    if task_type == "classification":
        lines += [
            "print('Accuracy: ', round(accuracy_score(y_test, y_pred), 3))",
            "print('Precision:', round(precision_score(y_test, y_pred, average='macro', zero_division=0), 3))",
            "print('Recall:   ', round(recall_score(y_test, y_pred, average='macro', zero_division=0), 3))",
            "print('Confusion matrix:'); print(confusion_matrix(y_test, y_pred))",
        ]
    else:
        lines += [
            "print('R²:  ', round(r2_score(y_test, y_pred), 3))",
            "print('MAE: ', round(mean_absolute_error(y_test, y_pred), 3))",
            "print('RMSE:', round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3))",
        ]

    # ── Сохранение ──────────────────────────────────────────────────────
    lines += [
        "",
        "# =============================================================",
        "# 5. СОХРАНЕНИЕ",
        "# =============================================================",
        "import joblib",
    ]

    if nn_type == "sklearn MLP":
        lines += [
            "joblib.dump({",
            "    'model':     pipeline,",
            f"    'features':  self_features,",
            f"    'task_type': '{task_type}',",
            "    'nn_type':   'sklearn_mlp',",
            "}, 'model_nn.pkl')",
            "print('Модель сохранена: model_nn.pkl')",
            "# Загрузка: data = joblib.load('model_nn.pkl'); pred = data['model'].predict(X_new)",
        ]
    else:
        lines += [
            "# Сохраняем model_state или tabnet_model отдельно через joblib",
            "# При загрузке нужно воссоздать архитектуру и загрузить веса",
            "joblib.dump({",
            "    'preprocessor': preprocessor,",
            f"    'features':     self_features,",
            f"    'task_type':    '{task_type}',",
            f"    'nn_type':      '{nn_type.lower().replace(' ', '_')}',",
            "}, 'model_nn_meta.pkl')",
            "print('Метаданные сохранены: model_nn_meta.pkl')",
        ]

    return "\n".join(lines)


# ==========================================
# БОКОВАЯ ПАНЕЛЬ
# ==========================================
with st.sidebar:
    st.header("📂 Входные данные")
    uploaded_file = st.file_uploader("Загрузи тренировочный датасет (CSV)", type="csv")

    if uploaded_file is not None:
        if ('last_uploaded' not in st.session_state
                or st.session_state.last_uploaded != uploaded_file.name):
            history = st.session_state.experiment_history
            for k, v in defaults.items():
                st.session_state[k] = v
            st.session_state.experiment_history = history
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.raw_df = pd.read_csv(uploaded_file)
            st.session_state.df = st.session_state.raw_df.copy()

        raw = st.session_state.raw_df
        st.divider()
        st.markdown("**📋 Сводка по датасету**")
        st.markdown(f"- Строк: **{raw.shape[0]:,}**")
        st.markdown(f"- Признаков: **{raw.shape[1]}**")
        mem_mb = raw.memory_usage(deep=True).sum() / 1024**2
        st.markdown(f"- Размер в памяти: **{mem_mb:.1f} МБ**")
        missing_total = raw.isnull().sum().sum()
        missing_pct   = missing_total / raw.size * 100
        st.markdown(f"- Пропусков: **{missing_total:,}** ({missing_pct:.1f}%)")
        num_count = len(raw.select_dtypes(include='number').columns)
        cat_count = len(raw.select_dtypes(exclude='number').columns)
        st.markdown(f"- Числовых / категориальных: **{num_count} / {cat_count}**")

        if st.session_state.is_trained:
            st.divider()
            st.markdown("**🤖 Активная модель**")
            st.markdown(f"- {st.session_state.trained_model_name}")
            task_icon = "🔵" if st.session_state.task_type == "classification" else "📈"
            st.markdown(f"- {task_icon} {st.session_state.task_type}")
            if st.session_state.metrics:
                first_k, first_v = list(st.session_state.metrics.items())[0]
                st.markdown(f"- {first_k}: **{first_v}**")

if uploaded_file is None:
    st.title("🚀 Universal ML Experimentation Platform")
    st.info("👈 Загрузи датасет в боковом меню слева, чтобы начать работу.")
    st.stop()

df = st.session_state.df
dataset_filename = st.session_state.get('last_uploaded', 'dataset.csv')
st.title("🚀 Universal ML Platform")

tab_eda, tab_train, tab_nn, tab_history = st.tabs([
    "📊 Анализ и Очистка",
    "⚙️ Обучение и Тестер",
    "🧠 Нейросети (ИНС)",
    "📜 История экспериментов",
])

# ==========================================
# ВКЛАДКА 1: EDA, ОЧИСТКА, FEATURE ENGINEERING
# ==========================================
with tab_eda:
    delta_rows = len(df) - len(st.session_state.raw_df)
    delta_cols = len(df.columns) - len(st.session_state.raw_df.columns)
    info_parts = [f"**{len(df):,}** строк × **{len(df.columns)}** столбцов"]
    if delta_rows != 0 or delta_cols != 0:
        info_parts.append(f"(Δ строк: {delta_rows:+d}, Δ столбцов: {delta_cols:+d})")
    st.caption(" ".join(info_parts))

    st.subheader("1. Качество данных")

    quality_report = []
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    for col in df.columns:
        missing     = df[col].isnull().sum()
        missing_pct = missing / len(df) * 100
        outliers, outliers_pct = None, None
        if col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            outliers     = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
            outliers_pct = round(outliers / len(df) * 100, 1)
        quality_report.append({
            "Признак":       col,
            "Тип":           str(df[col].dtype),
            "Пропуски (шт)": int(missing),
            "Пропуски (%)":  round(missing_pct, 1),
            "Выбросы (шт)":  outliers,
            "Выбросы (%)":   outliers_pct,
        })

    st.dataframe(pd.DataFrame(quality_report), use_container_width=True)

    with st.expander("🛠 Инструменты очистки", expanded=False):
        st.info("💡 **Рекомендуемый порядок:** удали ненужные столбцы → сгладь выбросы → заполни пропуски.")

        st.markdown("#### 1. Удаление столбцов по порогу пропусков")
        drop_thresh = st.slider("Удалить столбцы, где пропусков больше (%)", 10, 100, 50, key="drop_thresh")
        qdf = pd.DataFrame(quality_report)
        cols_would_drop = qdf[qdf['Пропуски (%)'] > drop_thresh]['Признак'].tolist()
        st.caption(f"Будут удалены: `{', '.join(cols_would_drop)}`" if cols_would_drop else "Нет столбцов, превышающих порог.")
        if st.button("🗑️ Удалить"):
            if cols_would_drop:
                st.session_state.df = df.drop(columns=cols_would_drop)
                st.session_state.custom_features = [
                    f for f in st.session_state.custom_features if f not in cols_would_drop]
                st.session_state.cleaning_log.append({"op": "drop_columns", "columns": cols_would_drop})
                st.success(f"Удалено: {', '.join(cols_would_drop)}")
                st.rerun()

        st.divider()

        st.markdown("#### 2. Сглаживание выбросов")
        st.info(
            "💡 Множитель **1.5 × IQR** — стандартный порог (правило Тьюки). "
            "Увеличь до **2.0–3.0** для данных с широким распределением."
        )
        iqr_mult = st.slider("Множитель IQR", min_value=1.0, max_value=4.0, value=1.5, step=0.5, key="iqr_mult")
        outlier_preview = []
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            n_out  = int(((df[col] < Q1 - iqr_mult*IQR) | (df[col] > Q3 + iqr_mult*IQR)).sum())
            if n_out > 0:
                outlier_preview.append(f"`{col}`: {n_out} шт.")
        st.caption(("Будут сглажены: " + " | ".join(outlier_preview)) if outlier_preview else "Выбросов не обнаружено.")
        cols_to_clip = st.multiselect("Применить только к столбцам (пусто = все числовые):",
                                      options=num_cols, default=[], key="cols_to_clip")
        if st.button("✂️ Сгладить выбросы"):
            new_df = df.copy()
            apply_cols = cols_to_clip if cols_to_clip else num_cols
            for col in apply_cols:
                Q1, Q3 = new_df[col].quantile(0.25), new_df[col].quantile(0.75)
                IQR    = Q3 - Q1
                new_df[col] = new_df[col].clip(lower=Q1 - iqr_mult*IQR, upper=Q3 + iqr_mult*IQR)
            st.session_state.df = new_df
            st.session_state.cleaning_log.append({"op": "clip_outliers", "columns": apply_cols, "iqr_mult": iqr_mult})
            st.success(f"Выбросы сглажены (k={iqr_mult}) в {len(apply_cols)} столбцах.")
            st.rerun()

        st.divider()

        st.markdown("#### 3. Заполнение пропусков")
        st.info(
            "💡 Для **числовых** признаков предпочтительна **Медиана** — устойчива к выбросам. "
            "Для **категориальных** используй **Моду**."
        )
        cols_with_missing = [c for c in df.columns if df[c].isnull().sum() > 0]
        if not cols_with_missing:
            st.success("✅ Пропусков нет — датасет чистый!")
        else:
            st.caption(f"Признаки с пропусками: {len(cols_with_missing)} шт.")
            fill_methods = {}
            fill_const   = {}
            missing_num  = [c for c in cols_with_missing if c in num_cols]
            missing_cat  = [c for c in cols_with_missing if c not in num_cols]

            if missing_num:
                st.markdown("**Числовые признаки:**")
                for col in missing_num:
                    n_miss = df[col].isnull().sum()
                    rc1, rc2, rc3 = st.columns([2, 2, 1])
                    rc1.markdown(f"`{col}` — **{n_miss}** пропусков")
                    method = rc2.selectbox("Метод", ["Медиана", "Среднее", "Мода", "Константа"],
                                           index=0, key=f"fill_method_{col}", label_visibility="collapsed")
                    fill_methods[col] = method
                    if method == "Константа":
                        fill_const[col] = rc3.number_input("Значение", value=0.0,
                                                            key=f"fill_const_{col}", label_visibility="collapsed")
            if missing_cat:
                st.markdown("**Категориальные признаки:**")
                for col in missing_cat:
                    n_miss = df[col].isnull().sum()
                    rc1, rc2, rc3 = st.columns([2, 2, 1])
                    rc1.markdown(f"`{col}` — **{n_miss}** пропусков")
                    method = rc2.selectbox("Метод", ["Мода", "Константа"],
                                           index=0, key=f"fill_method_{col}", label_visibility="collapsed")
                    fill_methods[col] = method
                    if method == "Константа":
                        fill_const[col] = rc3.text_input("Значение", value="unknown",
                                                          key=f"fill_const_{col}", label_visibility="collapsed")

            if st.button("✨ Заполнить пропуски"):
                new_df = df.copy()
                for col, method in fill_methods.items():
                    if method == "Медиана":
                        new_df[col] = new_df[col].fillna(new_df[col].median())
                    elif method == "Среднее":
                        new_df[col] = new_df[col].fillna(new_df[col].mean())
                    elif method == "Мода":
                        new_df[col] = new_df[col].fillna(new_df[col].mode()[0])
                    elif method == "Константа":
                        new_df[col] = new_df[col].fillna(fill_const.get(col, 0))
                st.session_state.df = new_df
                st.session_state.cleaning_log.append({
                    "op": "fill_missing",
                    "methods":   dict(fill_methods),
                    "constants": {k: v for k, v in fill_const.items()},
                })
                st.success("Пропуски заполнены!")
                st.rerun()

        st.divider()
        col_r, col_d = st.columns(2)
        with col_r:
            if st.button("🔄 Сброс к исходным данным"):
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.custom_features = []
                st.session_state.cleaning_log.append({"op": "reset"})
                st.rerun()
        with col_d:
            st.download_button("💾 Скачать очищенный CSV",
                               df.to_csv(index=False).encode('utf-8'),
                               file_name="cleaned_dataset.csv", mime="text/csv")

    st.subheader("2. Feature Engineering")
    with st.expander("🏗️ Создать и управлять признаками", expanded=True):
        st.write("Операторы: `+`, `-`, `*`, `/`, `**`. Например: `(SibSp + Parch) * Fare`")
        cf1, cf2, cf3 = st.columns([3, 1, 1])
        with cf1:
            formula = st.text_input("Формула:", placeholder="SibSp + Parch + 1")
        with cf2:
            new_feat_name = st.text_input("Название:", placeholder="FamilySize")
        with cf3:
            st.write(""); st.write("")
            if st.button("➕ Создать"):
                if formula and new_feat_name:
                    try:
                        new_df = df.copy()
                        new_df[new_feat_name] = new_df.eval(formula)
                        st.session_state.df = new_df
                        if new_feat_name not in st.session_state.custom_features:
                            st.session_state.custom_features.append(new_feat_name)
                        st.session_state.cleaning_log.append({
                            "op": "feature_engineering",
                            "name": new_feat_name, "formula": formula})
                        st.success(f"Признак '{new_feat_name}' создан!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                else:
                    st.warning("Заполни формулу и название.")

        if st.session_state.custom_features:
            st.divider()
            st.write("**Созданные признаки:**")
            for feat in st.session_state.custom_features:
                fc1, fc2 = st.columns([4, 1])
                fc1.code(feat)
                if fc2.button("🗑️", key=f"del_{feat}"):
                    if feat in df.columns:
                        st.session_state.df = df.drop(columns=[feat])
                    st.session_state.custom_features.remove(feat)
                    st.rerun()

    st.divider()
    st.subheader("3. Интерактивные графики")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        chart_type = st.selectbox("Тип:", ["Матрица корреляций", "Гистограмма",
                                           "Диаграмма рассеяния", "Ящик с усами"])
    if chart_type != "Матрица корреляций":
        with gc2:
            x_axis = st.selectbox("Ось X:", df.columns)
        with gc3:
            y_axis = st.selectbox("Ось Y / Цвет:", ["Нет"] + list(df.columns))

    if chart_type == "Гистограмма":
        fig = px.histogram(df, x=x_axis, color=(y_axis if y_axis != "Нет" else None),
                           marginal="box", title=f"Распределение {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Диаграмма рассеяния":
        if y_axis == "Нет":
            st.warning("Выбери Ось Y!")
        else:
            hue = st.selectbox("Цвет:", ["Нет"] + list(df.columns))
            fig = px.scatter(df, x=x_axis, y=y_axis,
                             color=(hue if hue != "Нет" else None),
                             title=f"{y_axis} vs {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Ящик с усами":
        fig = px.box(df, x=x_axis, color=(y_axis if y_axis != "Нет" else None),
                     title=f"Boxplot: {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Матрица корреляций":
        num_df = df.select_dtypes(include=['number'])
        if not num_df.empty:
            fig = px.imshow(num_df.corr(), text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# ВКЛАДКА 2: ОБУЧЕНИЕ И ТЕСТЕР
# ==========================================
with tab_train:
    st.write("Настрой параметры и запусти обучение.")

    tc1, tc2, tc3 = st.columns([1, 1, 1])
    with tc1:
        target_col = st.selectbox("Целевая колонка (Target):", df.columns)
    with tc2:
        selected_model = st.selectbox("Алгоритм:",
                                      ["Random Forest", "Gradient Boosting",
                                       "Logistic Regression", "Ансамбль (Ensemble)"])
    with tc3:
        if selected_model != "Ансамбль (Ensemble)":
            n_trials = st.slider("Итерации Optuna", min_value=5, max_value=150, value=20, step=5)
        else:
            st.info("Ансамбль использует фиксированные параметры.")
            n_trials = 1

    cv_c1, cv_c2 = st.columns([2, 1])
    with cv_c1:
        use_cv = st.toggle("Использовать Cross-Validation (K-Fold)",
                           help="Честнее оценивает модель, но в K раз дольше.")
    with cv_c2:
        cv_folds = st.slider("Фолды", 3, 10, 5, disabled=not use_cv)

    if use_cv:
        st.caption(f"⏱️ Всего обучений: {n_trials} × {cv_folds} = {n_trials * cv_folds}")
    if selected_model == "Logistic Regression":
        st.caption("ℹ️ При задаче регрессии автоматически заменится на Ridge.")

    if st.button("▶ Запустить ML пайплайн", use_container_width=True):
        with st.spinner(f"Обучаю {selected_model}..."):
            engine  = UniversalMLEngine(model_type=selected_model)
            metrics = engine.train_and_evaluate(
                df, target_col, n_trials=n_trials, use_cv=use_cv, cv_folds=cv_folds)
            explanation = engine.generate_human_explanation()
            engine.save_model("model.pkl")

            st.session_state.metrics            = metrics
            st.session_state.explanation        = explanation
            st.session_state.trained_model_name = selected_model
            st.session_state.task_type          = engine.task_type
            st.session_state.is_trained         = True
            st.session_state.train_df           = df.drop(columns=[target_col])
            st.session_state.features           = engine.features
            st.session_state.conf_matrix        = engine.conf_matrix
            st.session_state.class_labels       = engine.class_labels
            st.session_state['lc_data']         = engine.learning_curve_data

            record = {
                "⏰ Время":       datetime.datetime.now().strftime("%H:%M:%S"),
                "Модель":        selected_model,
                "Задача":        engine.task_type,
                "Target":        target_col,
                "CV":            f"{cv_folds}-fold" if use_cv else "hold-out",
                "Optuna trials": n_trials,
                "best_params":   dict(engine.best_params),
                "cleaning_log":  list(st.session_state.cleaning_log),
                **metrics,
            }
            st.session_state.experiment_history.append(record)

    if st.session_state.is_trained:
        task_label = "классификация" if st.session_state.task_type == "classification" else "регрессия"
        task_icon  = "🔵" if st.session_state.task_type == "classification" else "📈"
        st.success(
            f"✅ **{st.session_state.trained_model_name}** обучена! "
            f"{task_icon} Тип задачи: **{task_label}**"
        )

        m1, m2 = st.columns(2)
        with m1:
            st.subheader("📊 Метрики")
            mcols = st.columns(min(len(st.session_state.metrics), 3))
            for i, (name, value) in enumerate(st.session_state.metrics.items()):
                mcols[i % 3].metric(name, value)
        with m2:
            st.subheader("🧠 Интерпретация")
            st.info(st.session_state.explanation)

        # ── Важность признаков ──────────────────────────────────────────
        try:
            data_pkl     = joblib.load("model.pkl")
            pipe_inner   = data_pkl["model"]
            inner_model  = pipe_inner.named_steps['model']
            preprocessor = pipe_inner.named_steps['preprocessor']

            if hasattr(inner_model, 'feature_importances_'):
                importances = inner_model.feature_importances_
            elif hasattr(inner_model, 'coef_'):
                coef = inner_model.coef_
                importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
            else:
                importances = None

            if importances is not None:
                feat_names_fi = (list(preprocessor.transformers_[0][2])
                                 + list(preprocessor.transformers_[1][2]))
                fi_df = (pd.DataFrame({"Признак": feat_names_fi, "Важность": importances})
                           .sort_values("Важность", ascending=True).tail(15))
                st.divider()
                st.subheader("📌 Важность признаков")
                fig_fi = px.bar(fi_df, x="Важность", y="Признак", orientation="h",
                                title="Топ-15 признаков", color="Важность",
                                color_continuous_scale="Blues", text_auto=".3f")
                fig_fi.update_layout(showlegend=False, height=max(300, 30 * len(fi_df)))
                st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            pass

        # ── Confusion Matrix ────────────────────────────────────────────
        if (st.session_state.task_type == "classification"
                and st.session_state.conf_matrix is not None):
            st.divider()
            st.subheader("📉 Матрица ошибок")
            cm     = np.array(st.session_state.conf_matrix)
            labels = [str(l) for l in st.session_state.class_labels]
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm  = np.where(row_sums > 0, cm / row_sums * 100, 0).round(1)

            fig_cm = px.imshow(cm_norm, x=labels, y=labels,
                               color_continuous_scale="Blues",
                               labels=dict(x="Предсказано", y="Факт", color="%"),
                               title="Confusion Matrix (% от строки)", aspect="auto")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    fig_cm.add_annotation(x=labels[j], y=labels[i],
                                          text=f"{cm[i][j]}<br>({cm_norm[i][j]}%)",
                                          showarrow=False, font=dict(size=13, color="black"))
            fig_cm.update_layout(height=max(300, 100 * len(labels)))
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Learning Curve — диагностика переобучения ──────────────────
        lc_data = st.session_state.get('lc_data')
        if lc_data:
            st.divider()
            st.subheader("📈 Кривая обучения (Learning Curve)")
            st.caption(
                "Показывает как меняется качество модели при увеличении размера обучающей выборки. "
                "**Синяя** линия — качество на обучении, **красная** — на валидации (CV). "
                "Широкая полоса = высокая дисперсия между фолдами."
            )
            sizes   = lc_data['train_sizes']
            tr_mean = lc_data['train_mean']
            tr_std  = lc_data['train_std']
            vl_mean = lc_data['val_mean']
            vl_std  = lc_data['val_std']
            sc_name = lc_data['scoring'].upper()

            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=sizes + sizes[::-1],
                y=[m+s for m, s in zip(tr_mean, tr_std)] +
                  [m-s for m, s in zip(tr_mean[::-1], tr_std[::-1])],
                fill='toself', fillcolor='rgba(55,138,221,0.15)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False,
            ))
            fig_lc.add_trace(go.Scatter(
                x=sizes + sizes[::-1],
                y=[m+s for m, s in zip(vl_mean, vl_std)] +
                  [m-s for m, s in zip(vl_mean[::-1], vl_std[::-1])],
                fill='toself', fillcolor='rgba(231,76,60,0.12)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False,
            ))
            fig_lc.add_trace(go.Scatter(
                x=sizes, y=tr_mean, mode='lines+markers',
                name='Обучение', line=dict(color='#378add', width=2), marker=dict(size=7),
            ))
            fig_lc.add_trace(go.Scatter(
                x=sizes, y=vl_mean, mode='lines+markers',
                name='Валидация (CV)', line=dict(color='#e74c3c', width=2), marker=dict(size=7),
            ))
            fig_lc.update_layout(
                title=f'Learning Curve — {sc_name}',
                xaxis_title='Размер обучающей выборки (строк)',
                yaxis_title=sc_name,
                legend=dict(orientation='h', y=-0.2),
                height=380,
            )
            st.plotly_chart(fig_lc, use_container_width=True)

            gap = round(tr_mean[-1] - vl_mean[-1], 3)
            val_final = round(vl_mean[-1], 3)
            if gap > 0.15:
                st.error(
                    f"🔴 **Переобучение** — разрыв Train–Val = **{gap:.3f}**. "
                    "Модель хорошо запомнила обучающие данные, но плохо обобщает. "
                    "Попробуй: уменьшить сложность модели, добавить данных, включить CV."
                )
            elif val_final < 0.6 and lc_data['scoring'] == 'accuracy':
                st.warning(
                    f"🟡 **Недообучение** — Val {sc_name} = **{val_final:.3f}**. "
                    "Модель слишком простая. "
                    "Попробуй: увеличить n_estimators, глубину дерева, добавить признаки."
                )
            elif gap <= 0.05:
                st.success(
                    f"🟢 **Хороший баланс** — разрыв Train–Val = **{gap:.3f}**. "
                    "Модель хорошо обобщает и не переобучена."
                )
            else:
                st.info(
                    f"🔵 **Умеренный разрыв** Train–Val = **{gap:.3f}**. "
                    "Небольшое переобучение — добавление данных или CV может помочь."
                )

        # ── SHAP Waterfall (загрузка CSV) ───────────────────────────────
        st.divider()
        st.subheader("🔍 SHAP Waterfall")
        st.write("Загрузи CSV с одной строкой или объясни первую строку датасета.")
        sh1, sh2 = st.columns([2, 1])
        with sh1:
            shap_file = st.file_uploader("CSV для объяснения", type="csv", key="shap_file")
        with sh2:
            st.write("")
            use_first_row = st.button("🎲 Объяснить первую строку")

        shap_row_df = None
        if shap_file is not None:
            shap_row_df = pd.read_csv(shap_file).iloc[[0]]
        elif use_first_row and st.session_state.train_df is not None:
            shap_row_df = st.session_state.train_df.iloc[[0]]

        if shap_row_df is not None:
            with st.spinner("Вычисляю SHAP..."):
                data = joblib.load("model.pkl")
                se   = UniversalMLEngine(model_type=st.session_state.trained_model_name)
                se.pipeline  = data["model"]
                se.features  = data["features"]
                se.task_type = data.get("task_type", "classification")
                result = se.compute_shap_values(shap_row_df)

            if result is not None:
                sv, bv, fn = result
                top_n = min(12, len(sv))
                idx   = np.argsort(np.abs(sv))[::-1][:top_n]
                sv_t  = sv[idx]
                fn_t  = [fn[i] for i in idx]

                running = bv
                measures = ["absolute"]; texts = [f"{bv:.3f}"]
                for v in sv_t:
                    measures.append("relative"); texts.append(f"{v:+.3f}"); running += v
                measures.append("total"); texts.append(f"{running:.3f}")

                fig_shap = go.Figure(go.Waterfall(
                    orientation="h", measure=measures,
                    x=[bv] + list(sv_t) + [running],
                    y=["Базовое значение"] + fn_t + ["Итоговое предсказание"],
                    text=texts, textposition="outside",
                    connector={"line": {"color": "rgba(63,63,63,0.4)"}},
                    decreasing={"marker": {"color": "#3498db"}},
                    increasing={"marker": {"color": "#e74c3c"}},
                    totals={"marker":    {"color": "#2ecc71"}},
                ))
                fig_shap.update_layout(title=f"SHAP: топ-{top_n} признаков",
                                       xaxis_title="Значение",
                                       height=max(400, 40*(top_n+2)),
                                       margin=dict(l=20, r=100, t=60, b=40))
                st.plotly_chart(fig_shap, use_container_width=True)

                push_up   = [(n, v) for n, v in zip(fn_t[:3], sv_t[:3]) if v > 0]
                push_down = [(n, v) for n, v in zip(fn_t[:3], sv_t[:3]) if v < 0]
                parts = []
                if push_up:
                    parts.append("**повышают**: " + ", ".join(f"**{n}** (+{v:.3f})" for n, v in push_up))
                if push_down:
                    parts.append("**снижают**: " + ", ".join(f"**{n}** ({v:.3f})" for n, v in push_down))
                if parts:
                    st.info("Для этого наблюдения: " + "; ".join(parts) + ".")
            else:
                st.warning("⚠️ SHAP недоступен для данной комбинации модели и задачи.")

        # ── Встроенный тестер ─────────────────────────────────────────
        st.divider()
        st.subheader("🧪 Встроенный тестер модели")
        st.write("Введи значения признаков — получи предсказание прямо здесь.")
        train_df_snap = st.session_state.train_df

        def _make_widget(feat, col_widget, key_prefix):
            if train_df_snap is not None and feat in train_df_snap.columns:
                col_data = train_df_snap[feat].dropna()
                dtype    = col_data.dtype
                n_unique = col_data.nunique()
                if dtype in ['object', 'bool'] or str(dtype) == 'category':
                    return col_widget.selectbox(feat, sorted(col_data.unique().tolist(), key=str),
                                                key=f"{key_prefix}_{feat}")
                if n_unique <= 15:
                    return col_widget.selectbox(feat, sorted(col_data.unique().tolist()),
                                                key=f"{key_prefix}_{feat}")
                median_val = col_data.median()
                if dtype in ['int32', 'int64']:
                    return col_widget.number_input(feat, value=int(median_val), step=1,
                                                   key=f"{key_prefix}_{feat}")
                return col_widget.number_input(feat, value=float(median_val),
                                               step=None, format="%.4f",
                                               key=f"{key_prefix}_{feat}")
            return col_widget.text_input(feat, value="0", key=f"{key_prefix}_{feat}")

        with st.form("api_tester_form"):
            n_cols = min(len(st.session_state.features), 4)
            t_cols = st.columns(n_cols)
            input_values = {}
            for i, feat in enumerate(st.session_state.features):
                input_values[feat] = _make_widget(feat, t_cols[i % n_cols], "ti")
            predict_btn = st.form_submit_button("🔮 Предсказать", use_container_width=True)

        if predict_btn:
            try:
                data  = joblib.load("model.pkl")
                model = data["model"]
                feats = data["features"]
                row   = pd.DataFrame([input_values]).reindex(columns=feats)
                pred  = model.predict(row)[0]
                pred_py = pred.item() if hasattr(pred, 'item') else pred

                if data.get("task_type") == "classification" and hasattr(model, "predict_proba"):
                    proba  = model.predict_proba(row)[0]
                    labels = data.get("class_labels") or list(range(len(proba)))
                    st.success(f"**Предсказание: {pred_py}**")
                    proba_df = pd.DataFrame({"Класс": [str(l) for l in labels],
                                             "Вероятность": [round(float(p), 4) for p in proba]})
                    fig_p = px.bar(proba_df, x="Класс", y="Вероятность",
                                   color="Вероятность", color_continuous_scale="Blues",
                                   range_y=[0, 1], title="Вероятности по классам", text_auto=".3f")
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    val_str = f"{pred_py:.4f}" if isinstance(pred_py, float) else str(pred_py)
                    st.success(f"**Предсказание: {val_str}**")
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")

        if predict_btn and st.session_state.is_trained:
            try:
                data_shap = joblib.load("model.pkl")
                se_t = UniversalMLEngine(model_type=st.session_state.trained_model_name)
                se_t.pipeline  = data_shap["model"]
                se_t.features  = data_shap["features"]
                se_t.task_type = data_shap.get("task_type", "classification")
                row_shap = pd.DataFrame([input_values]).reindex(columns=se_t.features)
                with st.spinner("Считаю SHAP для введённых данных..."):
                    res_t = se_t.compute_shap_values(row_shap)

                if res_t is not None:
                    sv_t2, bv_t2, fn_t2 = res_t
                    top_n2 = min(12, len(sv_t2))
                    idx_t2 = np.argsort(np.abs(sv_t2))[::-1][:top_n2]
                    sv_show = sv_t2[idx_t2]
                    fn_show = [fn_t2[i] for i in idx_t2]

                    running2 = bv_t2
                    measures2 = ["absolute"]; texts2 = [f"{bv_t2:.3f}"]
                    for v in sv_show:
                        measures2.append("relative"); texts2.append(f"{v:+.3f}"); running2 += v
                    measures2.append("total"); texts2.append(f"{running2:.3f}")

                    fig_st = go.Figure(go.Waterfall(
                        orientation="h", measure=measures2,
                        x=[bv_t2] + list(sv_show) + [running2],
                        y=["Базовое значение"] + fn_show + ["Итоговое предсказание"],
                        text=texts2, textposition="outside",
                        connector={"line": {"color": "rgba(63,63,63,0.4)"}},
                        decreasing={"marker": {"color": "#3498db"}},
                        increasing={"marker": {"color": "#e74c3c"}},
                        totals={"marker":    {"color": "#2ecc71"}},
                    ))
                    fig_st.update_layout(
                        title=f"SHAP: почему именно такое предсказание (топ-{top_n2} признаков)",
                        xaxis_title="Значение",
                        height=max(400, 40 * (top_n2 + 2)),
                        margin=dict(l=20, r=100, t=60, b=40),
                    )
                    st.plotly_chart(fig_st, use_container_width=True)
            except Exception as e_shap:
                st.caption(f"SHAP недоступен: {e_shap}")

        st.divider()
        try:
            with open("model.pkl", "rb") as f_pkl:
                model_bytes = f_pkl.read()
            st.download_button(
                label="💾 Скачать модель (model.pkl)",
                data=model_bytes,
                file_name=f"model_{st.session_state.trained_model_name.replace(' ','_')}.pkl",
                mime="application/octet-stream",
            )
            st.caption("`data = joblib.load('model.pkl'); pred = data['model'].predict(X)`")
        except FileNotFoundError:
            pass


# ==========================================
# ВКЛАДКА 3: НЕЙРОСЕТИ
# ==========================================
with tab_nn:
    st.write(
        "Обучи нейронную сеть на тех же данных и сравни с классическими алгоритмами."
    )

    nn_col1, nn_col2 = st.columns([1, 2])
    with nn_col1:
        available_nn = ["sklearn MLP"]
        if PYTORCH_AVAILABLE:
            available_nn.append("PyTorch MLP")
        if TABNET_AVAILABLE:
            available_nn.append("TabNet")

        if not PYTORCH_AVAILABLE:
            err_detail = f"\n\n**Детали:** `{_PYTORCH_ERROR}`" if _PYTORCH_ERROR else ""
            st.warning(
                "⚠️ **PyTorch недоступен** — PyTorch MLP и TabNet отключены.  \n"
                "Исправление:\n```\npip uninstall torch -y\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cpu\n```"
                + err_detail
            )
        elif not TABNET_AVAILABLE:
            st.caption("ℹ️ TabNet: `pip install pytorch-tabnet`")

        nn_model_type = st.selectbox(
            "Архитектура нейросети:",
            options=available_nn,
            help=(
                "sklearn MLP — полносвязная сеть без новых зависимостей\n"
                "PyTorch MLP — кастомная сеть с BatchNorm и Dropout\n"
                "TabNet — трансформер для таблиц (требует pytorch-tabnet)"
            ),
        )
    with nn_col2:
        nn_target_col = st.selectbox("Целевая колонка:", df.columns, key="nn_target")

    st.divider()
    if nn_model_type == "sklearn MLP":
        st.markdown("**Параметры sklearn MLP**")
        sc1, sc2, sc3 = st.columns(3)
        layer1   = sc1.number_input("Нейроны в слое 1", 16, 512, 128, step=16)
        layer2   = sc2.number_input("Нейроны в слое 2 (0 = один слой)", 0, 512, 64, step=16)
        max_iter = sc3.number_input("Макс. итераций", 50, 1000, 300, step=50)
        hidden   = (layer1,) if layer2 == 0 else (layer1, layer2)
        st.caption(
            f"Архитектура: вход → {layer1}"
            + (f" → {layer2}" if layer2 > 0 else "") +
            " → выход | Early stopping включён"
        )

    elif nn_model_type == "PyTorch MLP":
        st.markdown("**Параметры PyTorch MLP**")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pt_l1      = pc1.number_input("Слой 1", 32, 512, 256, step=32)
        pt_l2      = pc2.number_input("Слой 2", 0, 512, 128, step=32)
        pt_l3      = pc3.number_input("Слой 3 (0 = пропустить)", 0, 512, 64, step=32)
        pt_dropout = pc4.slider("Dropout", 0.0, 0.7, 0.3, step=0.1)
        pt_epochs  = st.slider("Макс. эпох", 20, 300, 100, step=10)
        hidden     = tuple(d for d in [pt_l1, pt_l2, pt_l3] if d > 0)
        st.caption(
            f"Архитектура: вход → {' → '.join(str(d) for d in hidden)} → выход | "
            f"BatchNorm + Dropout({pt_dropout}) | Early stopping"
        )

    elif nn_model_type == "TabNet":
        st.markdown("**Параметры TabNet**")
        tb1, tb2, tb3 = st.columns(3)
        tb_steps  = tb1.slider("Шаги внимания (n_steps)", 2, 6, 3)
        tb_nd     = tb2.slider("Размер n_d / n_a", 8, 64, 16, step=8)
        tb_epochs = tb3.slider("Макс. эпох", 20, 300, 100, step=10)
        st.caption(f"TabNet: {tb_steps} шагов × n_d={tb_nd} | Интерпретируемость через механизм внимания")

    if st.button("🚀 Обучить нейросеть", use_container_width=True):
        with st.spinner(f"Обучаю {nn_model_type}..."):
            try:
                if nn_model_type == "sklearn MLP":
                    engine_nn = SklearnMLPEngine(hidden_layers=hidden, max_iter=int(max_iter))
                elif nn_model_type == "PyTorch MLP":
                    if not PYTORCH_AVAILABLE:
                        st.error("PyTorch недоступен.")
                        st.stop()
                    engine_nn = PyTorchMLPEngine(
                        hidden_dims=hidden, dropout=pt_dropout, max_epochs=int(pt_epochs))
                elif nn_model_type == "TabNet":
                    engine_nn = TabNetEngine(
                        n_steps=int(tb_steps), n_d=int(tb_nd), n_a=int(tb_nd),
                        max_epochs=int(tb_epochs))

                nn_metrics = engine_nn.train_and_evaluate(df, nn_target_col)
                engine_nn.save_model("model_nn.pkl")

                st.session_state["nn_metrics"]      = nn_metrics
                st.session_state["nn_explanation"]  = engine_nn.generate_human_explanation()
                st.session_state["nn_model_type"]   = nn_model_type
                st.session_state["nn_task_type"]    = engine_nn.task_type
                st.session_state["nn_conf_matrix"]  = engine_nn.conf_matrix
                st.session_state["nn_class_labels"] = engine_nn.class_labels
                st.session_state["nn_fi"] = (
                    engine_nn.feature_importances_
                    if hasattr(engine_nn, "feature_importances_") else None
                )
                st.session_state["nn_history"] = getattr(engine_nn, "train_history", None)

                if nn_model_type == "sklearn MLP":
                    st.session_state["nn_max_epochs"] = int(max_iter)
                elif nn_model_type == "PyTorch MLP":
                    st.session_state["nn_max_epochs"] = int(pt_epochs)
                elif nn_model_type == "TabNet":
                    st.session_state["nn_max_epochs"] = int(tb_epochs)

                nn_record = {
                    "⏰ Время":       datetime.datetime.now().strftime("%H:%M:%S"),
                    "Модель":        f"ИНС: {nn_model_type}",
                    "Задача":        engine_nn.task_type,
                    "Target":        nn_target_col,
                    "CV":            "hold-out",
                    "Optuna trials": 0,
                    "best_params":   dict(engine_nn.best_params),
                    "cleaning_log":  [],
                    **nn_metrics,
                }
                st.session_state.experiment_history.append(nn_record)

            except Exception as e:
                st.error(f"Ошибка обучения: {e}")

    if st.session_state.get("nn_metrics"):
        nn_task  = st.session_state["nn_task_type"]
        nn_icon  = "🔵" if nn_task == "classification" else "📈"
        nn_label = "классификация" if nn_task == "classification" else "регрессия"
        st.success(
            f"✅ **{st.session_state['nn_model_type']}** обучена! "
            f"{nn_icon} {nn_label}"
        )

        nn1, nn2 = st.columns(2)
        with nn1:
            st.subheader("📊 Метрики ИНС")
            nm_cols = st.columns(min(len(st.session_state["nn_metrics"]), 3))
            for i, (k, v) in enumerate(st.session_state["nn_metrics"].items()):
                nm_cols[i % 3].metric(k, v)
        with nn2:
            st.subheader("🧠 Архитектура")
            st.info(st.session_state["nn_explanation"])

        # ── Loss Curve нейросети ────────────────────────────────────────
        nn_history = st.session_state.get("nn_history")
        if nn_history and nn_history.get('train_loss'):
            st.divider()
            st.subheader("📉 Кривые обучения (Train vs Validation Loss)")
            st.caption(
                "**Синяя** линия — лосс на обучении, **красная пунктирная** — на валидации. "
                "Зелёная вертикаль — лучшая эпоха (минимум val loss). "
                "Большой разрыв train << val → переобучение."
            )
            train_l = nn_history['train_loss']
            val_l   = nn_history['val_loss']
            epochs  = list(range(1, len(train_l) + 1))

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=train_l, mode='lines',
                name='Train Loss', line=dict(color='#378add', width=2)
            ))
            if val_l:
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=val_l, mode='lines',
                    name='Val Loss', line=dict(color='#e74c3c', width=2, dash='dash')
                ))
                best_epoch = int(np.argmin(val_l)) + 1
                fig_loss.add_vline(
                    x=best_epoch, line_dash='dot', line_color='#2ecc71',
                    annotation_text=f'Лучшая эпоха: {best_epoch}',
                    annotation_position='top right',
                )

            fig_loss.update_layout(
                title='Динамика лосса по эпохам',
                xaxis_title='Эпоха / Итерация',
                yaxis_title='Loss',
                legend=dict(orientation='h', y=-0.2),
                height=360,
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            # Интерпретация
            final_train = train_l[-1]
            final_val   = val_l[-1] if val_l else final_train
            final_gap   = round(final_train - final_val, 4)
            n_iter      = nn_history['n_iter']
            max_ep_cfg  = st.session_state.get('nn_max_epochs', n_iter + 1)
            stopped_early = n_iter < max_ep_cfg

            if stopped_early:
                st.success(
                    f"🟢 **Early stopping** сработал на эпохе/итерации **{n_iter}** "
                    f"из {max_ep_cfg} — переобучение предотвращено."
                )
            if abs(final_gap) > 0.3:
                st.error(
                    f"🔴 **Переобучение** — Train Loss ({final_train:.4f}) "
                    f"намного ниже Val Loss ({final_val:.4f}). "
                    "Попробуй: увеличить Dropout, уменьшить число нейронов или эпох."
                )
            elif final_train > 0.5 and final_val > 0.5:
                st.warning(
                    "🟡 **Недообучение** — оба лосса высокие. "
                    "Попробуй увеличить число слоёв, нейронов или эпох."
                )
            else:
                st.info(
                    f"🔵 Train Loss: **{final_train:.4f}** | "
                    f"Val Loss: **{final_val:.4f}** | "
                    f"Разрыв: {abs(final_gap):.4f}"
                )

        # ── Сравнение с классическим ML ─────────────────────────────────
        if st.session_state.is_trained:
            st.divider()
            st.subheader("⚖️ Сравнение: ИНС vs Классический ML")
            classic_m = st.session_state.metrics
            nn_m      = st.session_state["nn_metrics"]
            common_keys = [k for k in classic_m if k in nn_m]
            if common_keys:
                cmp_df = pd.DataFrame({
                    "Метрика":        common_keys,
                    st.session_state.trained_model_name: [classic_m[k] for k in common_keys],
                    st.session_state["nn_model_type"]:   [nn_m[k]      for k in common_keys],
                })
                cmp_melted = cmp_df.melt(id_vars="Метрика", var_name="Модель", value_name="Значение")
                fig_cmp = px.bar(
                    cmp_melted, x="Метрика", y="Значение", color="Модель",
                    barmode="group",
                    title=f"{st.session_state.trained_model_name} vs {st.session_state['nn_model_type']}",
                    text_auto=".3f",
                    color_discrete_map={
                        st.session_state.trained_model_name: "#378add",
                        st.session_state["nn_model_type"]:   "#e74c3c",
                    },
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                first_key = common_keys[0]
                classic_v = classic_m[first_key]
                nn_v      = nn_m[first_key]
                margin    = abs(nn_v - classic_v)
                if margin < 0.005:
                    st.info(f"🤝 Результаты практически одинаковы (Δ={margin:.3f}).")
                elif nn_v > classic_v:
                    st.success(f"🏆 Нейросеть выигрывает по {first_key}: **{nn_v}** vs {classic_v}")
                else:
                    st.warning(f"⚡ Классический ML выигрывает по {first_key}: **{classic_v}** vs {nn_v}")
        else:
            st.info("💡 Обучи классическую модель во вкладке 'Обучение и Тестер' чтобы сравнить результаты.")

        # ── Confusion Matrix ИНС ────────────────────────────────────────
        if (nn_task == "classification" and st.session_state.get("nn_conf_matrix") is not None):
            st.divider()
            st.subheader("📉 Confusion Matrix (нейросеть)")
            cm     = np.array(st.session_state["nn_conf_matrix"])
            labels = [str(l) for l in st.session_state["nn_class_labels"]]
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm  = np.where(row_sums > 0, cm / row_sums * 100, 0).round(1)
            fig_nn_cm = px.imshow(cm_norm, x=labels, y=labels,
                                   color_continuous_scale="Reds",
                                   labels=dict(x="Предсказано", y="Факт", color="%"),
                                   title="Confusion Matrix ИНС", aspect="auto")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    fig_nn_cm.add_annotation(
                        x=labels[j], y=labels[i],
                        text=f"{cm[i][j]}<br>({cm_norm[i][j]}%)",
                        showarrow=False, font=dict(size=13, color="black"))
            fig_nn_cm.update_layout(height=max(300, 100 * len(labels)))
            st.plotly_chart(fig_nn_cm, use_container_width=True)

        # ── TabNet feature importances ──────────────────────────────────
        nn_fi = st.session_state.get("nn_fi")
        if nn_fi is not None and len(nn_fi) > 0:
            st.divider()
            st.subheader("📌 Важность признаков (TabNet attention)")
            fi_names = df.drop(columns=[nn_target_col]).columns.tolist()
            if len(fi_names) == len(nn_fi):
                fi_df = (pd.DataFrame({"Признак": fi_names, "Важность": nn_fi})
                           .sort_values("Важность", ascending=True).tail(15))
                fig_fi_nn = px.bar(fi_df, x="Важность", y="Признак", orientation="h",
                                   title="Важность признаков по механизму внимания TabNet",
                                   color="Важность", color_continuous_scale="Reds",
                                   text_auto=".3f")
                fig_fi_nn.update_layout(showlegend=False, height=max(300, 30*len(fi_df)))
                st.plotly_chart(fig_fi_nn, use_container_width=True)
                st.caption("Важность = среднее внимание по всем шагам sequential attention.")


# ==========================================
# ВКЛАДКА 4: ИСТОРИЯ ЭКСПЕРИМЕНТОВ
# ==========================================
with tab_history:
    st.subheader("📜 История экспериментов")
    history = st.session_state.experiment_history

    if not history:
        st.info("Пока нет запущенных экспериментов. Обучи хотя бы одну модель.")
    else:
        display_cols = [k for k in history[0].keys()
                        if k not in {"best_params", "cleaning_log"}]
        hist_df = pd.DataFrame(history)[display_cols]
        st.dataframe(hist_df, use_container_width=True)

        hc1, hc2 = st.columns([1, 1])
        with hc1:
            if st.button("🗑️ Очистить историю"):
                st.session_state.experiment_history = []
                st.rerun()
        with hc2:
            st.download_button(
                "💾 Скачать историю (CSV)",
                hist_df.to_csv(index=False).encode("utf-8"),
                file_name="experiment_history.csv", mime="text/csv")

        st.divider()

        numeric_cols = hist_df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            metric_to_plot = st.selectbox("Метрика для сравнения:", numeric_cols)
            hist_df["Эксперимент"] = (hist_df["⏰ Время"] + " | "
                                      + hist_df["Модель"] + " | "
                                      + hist_df["Target"])
            fig_h = px.bar(hist_df, x="Эксперимент", y=metric_to_plot, color="Модель",
                           title=f"Сравнение по «{metric_to_plot}»", text_auto=".3f")
            fig_h.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_h, use_container_width=True)

        st.divider()

        # ── Генератор Python-скрипта ────────────────────────────────────
        st.subheader("🐍 Скачать Python-скрипт для выбранного эксперимента")
        st.write(
            "Выбери эксперимент — получишь полностью воспроизводимый `.py` файл. "
            "Для **классического ML**: EDA + очистка + обучение с параметрами Optuna + оценка + сохранение. "
            "Для **нейросетей**: препроцессинг + архитектура + обучение с early stopping + график лосса."
        )

        exp_labels = [
            f"{i+1}. {r['⏰ Время']} | {r['Модель']} | target={r['Target']} | {r['CV']}"
            for i, r in enumerate(history)
        ]
        selected_idx = st.selectbox("Выбери эксперимент:", range(len(exp_labels)),
                                    format_func=lambda i: exp_labels[i])
        selected_record = history[selected_idx]

        # Определяем тип: ИНС или классика
        is_nn_record = selected_record.get("Модель", "").startswith("ИНС:")

        with st.expander("ℹ️ Параметры выбранного эксперимента", expanded=False):
            info_rows = {k: v for k, v in selected_record.items()
                         if k not in {"best_params", "cleaning_log"}}
            st.json(info_rows)
            bp = selected_record.get("best_params", {})
            if bp and "Инфо" not in bp:
                st.markdown("**Параметры модели:**")
                st.json(bp)
            cl = selected_record.get("cleaning_log", [])
            if cl:
                st.markdown(f"**Шагов очистки:** {len(cl)}")
                for step in cl:
                    st.write(f"- `{step['op']}`:", {k: v for k, v in step.items() if k != 'op'})

        # Индикатор типа скрипта
        if is_nn_record:
            nn_badge = selected_record["Модель"].replace("ИНС: ", "")
            st.info(f"🧠 Будет сгенерирован скрипт для нейросети: **{nn_badge}**")
        else:
            st.info(f"⚙️ Будет сгенерирован скрипт для классического ML: **{selected_record['Модель']}**")

        if st.button("⬇️ Сгенерировать и скачать .py скрипт", use_container_width=True):
            if is_nn_record:
                # ── ИНС: генерируем скрипт нейросети ───────────────────
                script_code = generate_nn_script(
                    record=selected_record,
                    dataset_filename=dataset_filename,
                )
                nn_type_safe = selected_record["Модель"].replace("ИНС: ", "").replace(" ", "_")
                fname = f"nn_solution_{nn_type_safe}_{selected_record['Target']}.py"
            else:
                # ── Классика: генерируем стандартный скрипт ─────────────
                script_code = generate_script(
                    record=selected_record,
                    cleaning_log=selected_record.get("cleaning_log", []),
                    dataset_filename=dataset_filename,
                )
                model_safe = (selected_record["Модель"]
                              .replace(" ", "_").replace("(", "").replace(")", ""))
                fname = f"ml_solution_{model_safe}_{selected_record['Target']}.py"

            st.download_button(
                label="📥 Скачать .py файл",
                data=script_code.encode("utf-8"),
                file_name=fname,
                mime="text/x-python",
                use_container_width=True,
            )
            st.divider()
            st.markdown("**Превью сгенерированного скрипта:**")
            st.code(script_code, language="python")