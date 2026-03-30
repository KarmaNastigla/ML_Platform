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

# ══════════════════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЙ SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
# Streamlit перезапускает скрипт при каждом взаимодействии.
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


# ══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР PYTHON-СКРИПТА (классический ML)
# ══════════════════════════════════════════════════════════════════════════════
def generate_script(record: dict, cleaning_log: list, dataset_filename: str) -> str:
    model_name  = record.get("Модель", "Random Forest")
    target_col  = record.get("Target", "target")
    task_type   = record.get("Задача", "classification")
    cv_mode     = record.get("CV", "hold-out")
    n_trials    = record.get("Optuna trials", 20)
    best_params = record.get("best_params", {})
    metrics     = {k: v for k, v in record.items()
                   if k not in {"⏰ Время","Модель","Задача","Target",
                                "CV","Optuna trials","best_params"}}
    use_cv   = cv_mode != "hold-out"
    cv_folds = int(cv_mode.replace("-fold","")) if use_cv else 5
    so = 1 if cleaning_log else 0  # section offset

    lines = [
        "# ================================================================",
        f"# Автоматически сгенерированный скрипт",
        f"# Модель:  {model_name}",
        f"# Target:  {target_col}",
        f"# Задача:  {task_type}",
        f"# Метрики: {metrics}",
        f"# Дата:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "# ================================================================",
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
        "Random Forest":
            ("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",),
        "Gradient Boosting":
            ("from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor",),
        "Logistic Regression":
            ("from sklearn.linear_model import LogisticRegression, Ridge",),
        "Ансамбль (Ensemble)": (
            "from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,",
            "    GradientBoostingClassifier, GradientBoostingRegressor,",
            "    VotingClassifier, VotingRegressor)",
            "from sklearn.linear_model import LogisticRegression, Ridge",
        ),
    }
    for imp in model_imports.get(model_name, []):
        lines.append(imp)
    lines += ["import joblib", "import optuna", ""]

    lines += [
        "# ── 1. ЗАГРУЗКА ДАННЫХ ──────────────────────────────────────────",
        f"df = pd.read_csv('{dataset_filename}')",
        f"print(f'Датасет: {{df.shape[0]}} строк × {{df.shape[1]}} столбцов')", "",
        "# ── 2. EDA ──────────────────────────────────────────────────────",
        "print(df.head().to_string())",
        "print(df.describe().round(2).to_string())",
        "_miss = df.isnull().sum(); _miss = _miss[_miss>0]",
        "if len(_miss): print('Пропуски:', _miss.to_string())",
        "_num = df.select_dtypes(include='number').columns",
        "if len(_num) > 1: print(df[_num].corr().round(2).to_string())", "",
    ]

    if cleaning_log:
        lines += ["# ── 3. ОЧИСТКА ──────────────────────────────────────────────"]
        for step in cleaning_log:
            op = step["op"]
            if op == "drop_columns":
                lines += [f"df = df.drop(columns={step['columns']})", ""]
            elif op == "clip_outliers":
                k = step["iqr_mult"]
                lines += [
                    f"for col in {step['columns']}:",
                    f"    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)",
                    f"    df[col] = df[col].clip(Q1-{k}*(Q3-Q1), Q3+{k}*(Q3-Q1))", ""]
            elif op == "fill_missing":
                for col, method in step["methods"].items():
                    if method == "Медиана":
                        lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
                    elif method == "Среднее":
                        lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())")
                    elif method == "Мода":
                        lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])")
                    elif method == "Константа":
                        v = step["constants"].get(col, 0)
                        lines.append(f"df['{col}'] = df['{col}'].fillna({repr(v)})")
                lines.append("")
            elif op == "feature_engineering":
                lines += [f"df['{step['name']}'] = df.eval('{step['formula']}')", ""]
            elif op == "reset":
                lines += [f"df = pd.read_csv('{dataset_filename}')", ""]

    lines += [
        f"# ── {3+so}. ПРИЗНАКИ ────────────────────────────────────────────",
        f"df = df.dropna(subset=['{target_col}'])",
        f"X = df.drop(columns=['{target_col}']); y = df['{target_col}']",
        "num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()",
        "cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()",
        "preprocessor = ColumnTransformer(transformers=[",
        "    ('num', Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]), num_cols),",
        "    ('cat', Pipeline([('i',SimpleImputer(strategy='most_frequent')),",
        "                      ('e',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]), cat_cols),",
        "])",
        "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)", "",
        f"# ── {4+so}. МОДЕЛЬ ─────────────────────────────────────────────",
    ]

    if model_name == "Ансамбль (Ensemble)":
        if task_type == "classification":
            lines += [
                "final_model = VotingClassifier(estimators=[",
                "    ('rf', RandomForestClassifier(n_estimators=100,max_depth=7,random_state=42)),",
                "    ('gb', GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=42)),",
                "    ('lr', LogisticRegression(C=1.0,max_iter=500,random_state=42)),",
                "], voting='soft')",
            ]
        else:
            lines += [
                "final_model = VotingRegressor(estimators=[",
                "    ('rf', RandomForestRegressor(n_estimators=100,max_depth=7,random_state=42)),",
                "    ('gb', GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=42)),",
                "    ('ridge', Ridge(alpha=1.0)),",
                "])",
            ]
    elif best_params and "Инфо" not in best_params:
        ps = ", ".join(f"{k}={repr(v)}" for k, v in best_params.items())
        mc = {
            ("Random Forest","classification"): "RandomForestClassifier",
            ("Random Forest","regression"):     "RandomForestRegressor",
            ("Gradient Boosting","classification"): "GradientBoostingClassifier",
            ("Gradient Boosting","regression"):     "GradientBoostingRegressor",
            ("Logistic Regression","classification"): "LogisticRegression",
            ("Logistic Regression","regression"):     "Ridge",
        }
        cls   = mc.get((model_name, task_type), "RandomForestClassifier")
        extra = ", max_iter=1000" if cls == "LogisticRegression" else ""
        lines += [f"final_model = {cls}({ps}{extra}, random_state=42)"]
    else:
        cv_sc = "accuracy" if task_type == "classification" else "r2"
        lines += [
            "optuna.logging.set_verbosity(optuna.logging.WARNING)",
            "def objective(trial): ...",
            f"study = optuna.create_study(direction='maximize')",
            f"study.optimize(objective, n_trials={n_trials})",
        ]

    lines += [
        "pipeline = Pipeline([('preprocessor',preprocessor),('model',final_model)])",
        "pipeline.fit(X_train, y_train)",
        "y_pred = pipeline.predict(X_test)", "",
        f"# ── {5+so}. ОЦЕНКА ─────────────────────────────────────────────",
    ]
    if task_type == "classification":
        lines += [
            "print('Accuracy: ', round(accuracy_score(y_test,y_pred),3))",
            "print('Precision:', round(precision_score(y_test,y_pred,average='macro',zero_division=0),3))",
            "print('Recall:   ', round(recall_score(y_test,y_pred,average='macro',zero_division=0),3))",
            "print(confusion_matrix(y_test,y_pred))",
        ]
    else:
        lines += [
            "print('R²:  ', round(r2_score(y_test,y_pred),3))",
            "print('MAE: ', round(mean_absolute_error(y_test,y_pred),3))",
            "print('RMSE:', round(float(np.sqrt(mean_squared_error(y_test,y_pred))),3))",
        ]
    lines += [
        "",
        f"# ── {6+so}. СОХРАНЕНИЕ ──────────────────────────────────────────",
        "joblib.dump({'model':pipeline,'features':list(X.columns),'task_type':'"
        + task_type + "'}, 'model.pkl')",
        "print('Модель сохранена в model.pkl')",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР PYTHON-СКРИПТА (нейронные сети)
# ══════════════════════════════════════════════════════════════════════════════
def generate_nn_script(record: dict, dataset_filename: str) -> str:
    model_name  = record.get("Модель", "ИНС: sklearn MLP")
    target_col  = record.get("Target", "target")
    task_type   = record.get("Задача", "classification")
    best_params = record.get("best_params", {})
    nn_type     = model_name.replace("ИНС: ", "").strip()
    metrics     = {k: v for k, v in record.items()
                   if k not in {"⏰ Время","Модель","Задача","Target",
                                "CV","Optuna trials","best_params","cleaning_log"}}
    lines = [
        "# ================================================================",
        f"# Нейросеть: {nn_type}",
        f"# Target:    {target_col}",
        f"# Задача:    {task_type}",
        f"# Метрики:   {metrics}",
        f"# Дата:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "# ================================================================",
        "",
        "import pandas as pd",
        "import numpy as np",
        "import warnings; warnings.filterwarnings('ignore')",
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
        cls = "MLPClassifier" if task_type == "classification" else "MLPRegressor"
        lines += [f"from sklearn.neural_network import {cls}"]
    elif nn_type in ("PyTorch MLP", "TabNet"):
        lines += ["import torch, torch.nn as nn",
                  "from torch.utils.data import DataLoader, TensorDataset",
                  "from sklearn.preprocessing import LabelEncoder"]
    if nn_type == "TabNet":
        tc = "TabNetClassifier" if task_type == "classification" else "TabNetRegressor"
        lines += [f"from pytorch_tabnet.tab_model import {tc}"]
    lines += ["import joblib", ""]

    lines += [
        "# ── 1. ЗАГРУЗКА ДАННЫХ ──────────────────────────────────────────",
        f"df = pd.read_csv('{dataset_filename}').dropna(subset=['{target_col}'])",
        f"X = df.drop(columns=['{target_col}']); y = df['{target_col}']",
        "num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()",
        "cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()",
        "",
        "# ── 2. ПРЕПРОЦЕССИНГ ────────────────────────────────────────────",
        "preprocessor = ColumnTransformer(transformers=[",
        "    ('num', Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]), num_cols),",
        "    ('cat', Pipeline([('i',SimpleImputer(strategy='most_frequent')),",
        "                      ('e',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]), cat_cols),",
        "])",
        "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)", "",
        "# ── 3. ОБУЧЕНИЕ ─────────────────────────────────────────────────",
    ]

    if nn_type == "sklearn MLP":
        hl  = best_params.get('hidden_layers', (128, 64))
        mi  = best_params.get('max_iter', 300)
        lr  = best_params.get('lr', 0.001)
        cls = "MLPClassifier" if task_type == "classification" else "MLPRegressor"
        lines += [
            f"model = {cls}(hidden_layer_sizes={hl}, max_iter={mi},",
            f"    learning_rate_init={lr}, early_stopping=True,",
            f"    validation_fraction=0.1, random_state=42)",
            "pipeline = Pipeline([('preprocessor',preprocessor),('model',model)])",
            "pipeline.fit(X_train, y_train)",
            "y_pred = pipeline.predict(X_test)",
            "",
            "# Кривая лосса",
            "import matplotlib.pyplot as plt",
            "inner = pipeline.named_steps['model']",
            "plt.figure(figsize=(10,4))",
            "plt.plot(inner.loss_curve_, label='Train Loss')",
            "if hasattr(inner,'validation_scores_'):",
            "    plt.plot([1-s for s in inner.validation_scores_], '--', label='Val Loss')",
            "plt.xlabel('Итерация'); plt.ylabel('Loss')",
            f"plt.title('sklearn MLP Loss Curve'); plt.legend()",
            "plt.tight_layout(); plt.savefig('loss_curve.png',dpi=120)",
        ]
    elif nn_type == "PyTorch MLP":
        hd  = best_params.get('hidden_dims', (256, 128, 64))
        dr  = best_params.get('dropout', 0.3)
        lr  = best_params.get('lr', 0.001)
        ep  = best_params.get('max_epochs', 100)
        out = "n_classes" if task_type == "classification" else "1"
        lines += [
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "X_tr = preprocessor.fit_transform(X_train).astype(np.float32)",
            "X_val_raw, X_te, y_val, y_te = train_test_split(X_test,y_test,test_size=0.5,random_state=42)",
            "X_va = preprocessor.transform(X_val_raw).astype(np.float32)",
            "X_te_t = preprocessor.transform(X_te).astype(np.float32)",
            "",
            "# Архитектура MLP",
            "def make_mlp(inp, out):",
            "    layers, prev = [], inp",
            f"    for d in {list(hd)}:",
            f"        layers += [nn.Linear(prev,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout({dr})]",
            "        prev = d",
            "    layers.append(nn.Linear(prev, out))",
            "    return nn.Sequential(*layers)",
        ]
        if task_type == "classification":
            lines += [
                "le = LabelEncoder()",
                "y_tr = le.fit_transform(y_train).astype(np.int64)",
                "y_va = le.transform(y_val).astype(np.int64)",
                "n_classes = len(le.classes_)",
                "model = make_mlp(X_tr.shape[1], n_classes).to(device)",
                "crit  = nn.CrossEntropyLoss()",
                "ds    = TensorDataset(torch.tensor(X_tr).to(device), torch.tensor(y_tr).to(device))",
            ]
        else:
            lines += [
                "y_tr_np = y_train.values.astype(np.float32).reshape(-1,1)",
                "y_va_np = y_val.values.astype(np.float32).reshape(-1,1)",
                "model   = make_mlp(X_tr.shape[1], 1).to(device)",
                "crit    = nn.MSELoss()",
                "ds      = TensorDataset(torch.tensor(X_tr).to(device), torch.tensor(y_tr_np).to(device))",
            ]
        lines += [
            f"opt  = torch.optim.Adam(model.parameters(), lr={lr}, weight_decay=1e-4)",
            "sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)",
            f"loader = DataLoader(ds, batch_size=256, shuffle=True)",
            "tl, vl, best_v, pat, best_s = [], [], 1e9, 0, None",
            f"for ep in range({ep}):",
            "    model.train()",
            "    for xb, yb in loader:",
            "        opt.zero_grad(); loss = crit(model(xb),yb); loss.backward(); opt.step()",
            "    model.eval()",
            "    with torch.no_grad():",
        ]
        if task_type == "classification":
            lines += [
                "        vl_loss = crit(model(torch.tensor(X_va).to(device)), torch.tensor(y_va).to(device)).item()",
            ]
        else:
            lines += [
                "        vl_loss = crit(model(torch.tensor(X_va).to(device)), torch.tensor(y_va_np).to(device)).item()",
            ]
        lines += [
            "        tr_loss = crit(model(ds.tensors[0]), ds.tensors[1]).item()",
            "    tl.append(round(tr_loss,5)); vl.append(round(vl_loss,5)); sch.step(vl_loss)",
            "    if vl_loss < best_v: best_v,pat,best_s = vl_loss,0,{k:v.clone() for k,v in model.state_dict().items()}",
            "    else:",
            "        pat += 1",
            "        if pat >= 15: print(f'Early stopping эпоха {ep+1}'); break",
            "if best_s: model.load_state_dict(best_s)",
            "",
            "import matplotlib.pyplot as plt",
            "plt.figure(figsize=(10,4)); plt.plot(tl,label='Train'); plt.plot(vl,'--',label='Val')",
            "plt.xlabel('Эпоха'); plt.ylabel('Loss'); plt.legend(); plt.savefig('loss_curve.png',dpi=120)",
            "",
            "model.eval()",
            "with torch.no_grad(): out = model(torch.tensor(X_te_t).to(device))",
        ]
        if task_type == "classification":
            lines += ["y_pred = le.inverse_transform(out.argmax(1).cpu().numpy())"]
        else:
            lines += ["y_pred = out.cpu().numpy().ravel()"]
        lines += ["y_test = y_te"]

    lines += [
        "",
        "# ── 4. ОЦЕНКА ───────────────────────────────────────────────────",
    ]
    if task_type == "classification":
        lines += [
            "print('Accuracy:', round(accuracy_score(y_test,y_pred),3))",
            "print('Precision:', round(precision_score(y_test,y_pred,average='macro',zero_division=0),3))",
            "print('Recall:', round(recall_score(y_test,y_pred,average='macro',zero_division=0),3))",
        ]
    else:
        lines += [
            "print('R²:  ', round(r2_score(y_test,y_pred),3))",
            "print('MAE: ', round(mean_absolute_error(y_test,y_pred),3))",
            "print('RMSE:', round(float(np.sqrt(mean_squared_error(y_test,y_pred))),3))",
        ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# БОКОВАЯ ПАНЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════
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
            st.session_state.last_uploaded      = uploaded_file.name
            st.session_state.raw_df             = pd.read_csv(uploaded_file)
            st.session_state.df                 = st.session_state.raw_df.copy()

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

df               = st.session_state.df
dataset_filename = st.session_state.get('last_uploaded', 'dataset.csv')
st.title("🚀 Universal ML Platform")

tab_eda, tab_train, tab_nn, tab_history = st.tabs([
    "📊 Анализ и Очистка",
    "⚙️ Обучение и Тестер",
    "🧠 Нейросети (ИНС)",
    "📜 История экспериментов",
])


# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 1: EDA, ОЧИСТКА, FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    delta_rows = len(df) - len(st.session_state.raw_df)
    delta_cols = len(df.columns) - len(st.session_state.raw_df.columns)
    info_parts = [f"**{len(df):,}** строк × **{len(df.columns)}** столбцов"]
    if delta_rows != 0 or delta_cols != 0:
        info_parts.append(f"(Δ строк: {delta_rows:+d}, Δ столбцов: {delta_cols:+d})")
    st.caption(" ".join(info_parts))

    st.subheader("1. Качество данных")
    quality_report = []
    num_cols_eda = df.select_dtypes(include=['number']).columns.tolist()
    for col in df.columns:
        missing     = df[col].isnull().sum()
        missing_pct = missing / len(df) * 100
        outliers, outliers_pct = None, None
        if col in num_cols_eda:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            outliers     = int(((df[col] < Q1-1.5*IQR)|(df[col] > Q3+1.5*IQR)).sum())
            outliers_pct = round(outliers/len(df)*100, 1)
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
        drop_thresh    = st.slider("Удалить столбцы, где пропусков больше (%)", 10, 100, 50, key="drop_thresh")
        qdf            = pd.DataFrame(quality_report)
        cols_would_drop= qdf[qdf['Пропуски (%)'] > drop_thresh]['Признак'].tolist()
        st.caption(f"Будут удалены: `{', '.join(cols_would_drop)}`" if cols_would_drop else "Нет столбцов, превышающих порог.")
        if st.button("🗑️ Удалить"):
            if cols_would_drop:
                st.session_state.df = df.drop(columns=cols_would_drop)
                st.session_state.custom_features = [
                    f for f in st.session_state.custom_features if f not in cols_would_drop]
                st.session_state.cleaning_log.append({"op":"drop_columns","columns":cols_would_drop})
                st.success(f"Удалено: {', '.join(cols_would_drop)}"); st.rerun()

        st.divider()
        st.markdown("#### 2. Сглаживание выбросов")
        st.info("💡 Множитель **1.5×IQR** — стандарт. Увеличь до 2.0–3.0 для данных с широким распределением.")
        iqr_mult = st.slider("Множитель IQR", 1.0, 4.0, 1.5, 0.5, key="iqr_mult")
        outlier_preview = []
        for col in num_cols_eda:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            n_out  = int(((df[col] < Q1-iqr_mult*IQR)|(df[col] > Q3+iqr_mult*IQR)).sum())
            if n_out > 0:
                outlier_preview.append(f"`{col}`: {n_out} шт.")
        st.caption(("Будут сглажены: " + " | ".join(outlier_preview)) if outlier_preview else "Выбросов не обнаружено.")
        cols_to_clip = st.multiselect("Применить только к (пусто = все числовые):", num_cols_eda, [], key="cols_to_clip")
        if st.button("✂️ Сгладить выбросы"):
            new_df = df.copy()
            apply_cols = cols_to_clip if cols_to_clip else num_cols_eda
            for col in apply_cols:
                Q1, Q3 = new_df[col].quantile(0.25), new_df[col].quantile(0.75)
                IQR = Q3 - Q1
                new_df[col] = new_df[col].clip(Q1-iqr_mult*IQR, Q3+iqr_mult*IQR)
            st.session_state.df = new_df
            st.session_state.cleaning_log.append({"op":"clip_outliers","columns":apply_cols,"iqr_mult":iqr_mult})
            st.success(f"Выбросы сглажены (k={iqr_mult}) в {len(apply_cols)} столбцах."); st.rerun()

        st.divider()
        st.markdown("#### 3. Заполнение пропусков")
        cols_with_missing = [c for c in df.columns if df[c].isnull().sum() > 0]
        if not cols_with_missing:
            st.success("✅ Пропусков нет!")
        else:
            fill_methods, fill_const = {}, {}
            missing_num = [c for c in cols_with_missing if c in num_cols_eda]
            missing_cat = [c for c in cols_with_missing if c not in num_cols_eda]
            if missing_num:
                st.markdown("**Числовые:**")
                for col in missing_num:
                    r1, r2, r3 = st.columns([2,2,1])
                    r1.markdown(f"`{col}` — **{df[col].isnull().sum()}** пропусков")
                    m = r2.selectbox("Метод", ["Медиана","Среднее","Мода","Константа"],
                                     key=f"fill_method_{col}", label_visibility="collapsed")
                    fill_methods[col] = m
                    if m == "Константа":
                        fill_const[col] = r3.number_input("Значение", value=0.0,
                            key=f"fill_const_{col}", label_visibility="collapsed")
            if missing_cat:
                st.markdown("**Категориальные:**")
                for col in missing_cat:
                    r1, r2, r3 = st.columns([2,2,1])
                    r1.markdown(f"`{col}` — **{df[col].isnull().sum()}** пропусков")
                    m = r2.selectbox("Метод", ["Мода","Константа"],
                                     key=f"fill_method_{col}", label_visibility="collapsed")
                    fill_methods[col] = m
                    if m == "Константа":
                        fill_const[col] = r3.text_input("Значение", value="unknown",
                            key=f"fill_const_{col}", label_visibility="collapsed")
            if st.button("✨ Заполнить пропуски"):
                new_df = df.copy()
                for col, method in fill_methods.items():
                    if method == "Медиана":  new_df[col] = new_df[col].fillna(new_df[col].median())
                    elif method == "Среднее": new_df[col] = new_df[col].fillna(new_df[col].mean())
                    elif method == "Мода":    new_df[col] = new_df[col].fillna(new_df[col].mode()[0])
                    elif method == "Константа": new_df[col] = new_df[col].fillna(fill_const.get(col,0))
                st.session_state.df = new_df
                st.session_state.cleaning_log.append({
                    "op":"fill_missing","methods":dict(fill_methods),
                    "constants":{k:v for k,v in fill_const.items()}})
                st.success("Пропуски заполнены!"); st.rerun()

        st.divider()
        col_r, col_d = st.columns(2)
        with col_r:
            if st.button("🔄 Сброс к исходным данным"):
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.custom_features = []
                st.session_state.cleaning_log.append({"op":"reset"}); st.rerun()
        with col_d:
            st.download_button("💾 Скачать очищенный CSV",
                               df.to_csv(index=False).encode('utf-8'),
                               file_name="cleaned_dataset.csv", mime="text/csv")

    st.subheader("2. Feature Engineering")
    with st.expander("🏗️ Создать и управлять признаками", expanded=True):
        st.write("Операторы: `+`, `-`, `*`, `/`, `**`. Например: `(SibSp + Parch) * Fare`")
        cf1, cf2, cf3 = st.columns([3,1,1])
        formula       = cf1.text_input("Формула:", placeholder="SibSp + Parch + 1")
        new_feat_name = cf2.text_input("Название:", placeholder="FamilySize")
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
                            "op":"feature_engineering","name":new_feat_name,"formula":formula})
                        st.success(f"Признак '{new_feat_name}' создан!"); st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                else:
                    st.warning("Заполни формулу и название.")
        if st.session_state.custom_features:
            st.divider(); st.write("**Созданные признаки:**")
            for feat in st.session_state.custom_features:
                fc1, fc2 = st.columns([4,1])
                fc1.code(feat)
                if fc2.button("🗑️", key=f"del_{feat}"):
                    if feat in df.columns:
                        st.session_state.df = df.drop(columns=[feat])
                    st.session_state.custom_features.remove(feat); st.rerun()

    st.divider()
    st.subheader("3. Интерактивные графики")
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        chart_type = st.selectbox("Тип:", ["Матрица корреляций","Гистограмма",
                                            "Диаграмма рассеяния","Ящик с усами"])
    if chart_type != "Матрица корреляций":
        with gc2: x_axis = st.selectbox("Ось X:", df.columns)
        with gc3: y_axis = st.selectbox("Ось Y / Цвет:", ["Нет"] + list(df.columns))

    if chart_type == "Гистограмма":
        fig = px.histogram(df, x=x_axis, color=(y_axis if y_axis!="Нет" else None),
                           marginal="box", title=f"Распределение {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Диаграмма рассеяния":
        if y_axis == "Нет": st.warning("Выбери Ось Y!")
        else:
            hue = st.selectbox("Цвет:", ["Нет"] + list(df.columns))
            fig = px.scatter(df, x=x_axis, y=y_axis,
                             color=(hue if hue!="Нет" else None), title=f"{y_axis} vs {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Ящик с усами":
        fig = px.box(df, x=x_axis, color=(y_axis if y_axis!="Нет" else None),
                     title=f"Boxplot: {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Матрица корреляций":
        num_df = df.select_dtypes(include=['number'])
        if not num_df.empty:
            fig = px.imshow(num_df.corr(), text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 2: ОБУЧЕНИЕ И ТЕСТЕР
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.write("Настрой параметры и запусти обучение.")

    # ── Основные параметры ────────────────────────────────────────────────────
    tc1, tc2, tc3 = st.columns([1,1,1])
    with tc1:
        target_col = st.selectbox("Целевая колонка (Target):", df.columns)
    with tc2:
        selected_model = st.selectbox("Алгоритм:",
            ["Random Forest","Gradient Boosting","Logistic Regression","Ансамбль (Ensemble)"])
    with tc3:
        if selected_model != "Ансамбль (Ensemble)":
            n_trials = st.slider("Итерации Optuna", 5, 150, 20, 5)
        else:
            st.info("Ансамбль: фиксированные параметры.")
            n_trials = 1

    cv_c1, cv_c2 = st.columns([2,1])
    with cv_c1:
        use_cv = st.toggle("Использовать Cross-Validation (K-Fold)",
                           help="Честнее оценивает модель, но медленнее в K раз.")
    with cv_c2:
        cv_folds = st.slider("Фолды", 3, 10, 5, disabled=not use_cv)
    if use_cv:
        st.caption(f"⏱️ Всего обучений: {n_trials} × {cv_folds} = {n_trials*cv_folds}")
    if selected_model == "Logistic Regression":
        st.caption("ℹ️ При задаче регрессии автоматически заменится на Ridge.")

    # ── Расширенные гиперпараметры (диапазоны поиска Optuna) ─────────────────
    with st.expander("🔧 Пространство поиска гиперпараметров (Optuna)", expanded=False):
        st.info(
            "Задай диапазоны поиска для Optuna. "
            "**Зауженные диапазоны** = более точный поиск в нужной области. "
            "Используй для борьбы с переобучением: уменьши **max_depth**, "
            "увеличь **min_samples_leaf**, уменьши **subsample**."
        )

        hp_ranges_user = {}   # будет передан в engine.train_and_evaluate()

        if selected_model == "Random Forest":
            c1, c2 = st.columns(2)
            n_est_range = c1.slider(
                "n_estimators (кол-во деревьев)", 10, 1000, (50, 300),
                help="Больше деревьев = стабильнее, но медленнее. >300 даёт минимальный прирост.")
            max_depth_range = c2.slider(
                "max_depth (глубина дерева)", 1, 30, (3, 15),
                help="⭐ Ключевой параметр! Уменьши верхнюю границу (до 5–8) чтобы снизить переобучение.")
            c3, c4 = st.columns(2)
            leaf_range = c3.slider(
                "min_samples_leaf (мин. объектов в листе)", 1, 50, (1, 5),
                help="Увеличь нижнюю границу (до 10–20) — более гладкие листья = меньше переобучения.")
            split_range = c4.slider(
                "min_samples_split (мин. объектов для разбиения)", 2, 50, (2, 10),
                help="Больше = реже разбиваем узлы = менее сложная модель.")
            max_feat = st.selectbox(
                "max_features (признаков при разбиении)", ['sqrt', 'log2', 1.0, 0.5],
                help="'sqrt' = √p — стандарт RF. 'log2' = log2(p). 1.0 = все признаки (медленнее).")
            hp_ranges_user = {
                'n_estimators':      n_est_range,
                'max_depth':         max_depth_range,
                'min_samples_leaf':  leaf_range,
                'min_samples_split': split_range,
                'max_features':      max_feat,
            }
            # Подсказки по борьбе с переобучением
            if max_depth_range[1] > 10:
                st.warning("⚠️ max_depth > 10 может привести к переобучению. "
                           "Попробуй ограничить до (3, 7).")
            if leaf_range[0] < 2 and leaf_range[1] < 3:
                st.info("💡 min_samples_leaf=(1,2) → маленькие листья → риск переобучения. "
                        "Попробуй (5, 20).")

        elif selected_model == "Gradient Boosting":
            c1, c2 = st.columns(2)
            n_est_range = c1.slider("n_estimators", 10, 1000, (50, 300))
            depth_range  = c2.slider(
                "max_depth (глубина дерева)", 1, 15, (3, 10),
                help="GB деревья намеренно мелкие (слабые ученики). 3–5 = оптимально.")
            c3, c4 = st.columns(2)
            lr_range = c3.slider(
                "learning_rate (шаг обучения)", 0.001, 0.5, (0.01, 0.3),
                help="Малый lr + больше деревьев = лучше. Уменьши lr и увеличь n_estimators.")
            sub_range = c4.slider(
                "subsample (доля выборки на дерево)", 0.3, 1.0, (0.7, 1.0),
                help="⭐ Ключевой параметр против переобучения! < 1.0 = стохастический GB."
                     " Попробуй (0.5, 0.8) при переобучении.")
            leaf_range = st.slider(
                "min_samples_leaf", 1, 50, (1, 5),
                help="Больше = менее детальные листья = меньше переобучения.")
            hp_ranges_user = {
                'n_estimators':     n_est_range,
                'max_depth':        depth_range,
                'learning_rate':    lr_range,
                'subsample':        sub_range,
                'min_samples_leaf': leaf_range,
            }
            if sub_range[1] == 1.0 and depth_range[1] >= 8:
                st.warning("⚠️ subsample=1.0 + max_depth≥8 → высокий риск переобучения. "
                           "Установи subsample=(0.5, 0.8).")

        elif selected_model == "Logistic Regression":
            c1, c2 = st.columns(2)
            c_range = c1.slider(
                "C (обратная регуляризация, классификация)", 0.001, 50.0, (0.01, 20.0),
                help="Малый C → сильная L2-регуляризация → простая модель. "
                     "При переобучении уменьши верхнюю границу до 1–5.")
            alpha_range = c2.slider(
                "alpha (Ridge, регрессия)", 0.001, 100.0, (0.01, 50.0),
                help="Большой alpha → сильная L2-регуляризация. "
                     "При переобучении увеличь нижнюю границу.")
            hp_ranges_user = {
                'C':     c_range,
                'alpha': alpha_range,
            }

        elif selected_model == "Ансамбль (Ensemble)":
            st.info("Ансамбль использует фиксированные параметры — "
                    "разнородность алгоритмов обеспечивает стабильность без Optuna.")

    # ── Кнопка запуска ────────────────────────────────────────────────────────
    if st.button("▶ Запустить ML пайплайн", use_container_width=True):

        # ── КОНТЕЙНЕРЫ ПРОГРЕССА ─────────────────────────────────────────────
        # st.empty() — «слот» в UI, который можно перезаписывать из callback.
        # Это позволяет обновлять прогресс-бар без перезапуска всего скрипта.
        prog_container  = st.container()
        prog_bar        = prog_container.progress(0, text="Инициализация Optuna...")
        stat_cols       = prog_container.columns(4)
        cur_trial_ph    = stat_cols[0].empty()   # «Trial X / N»
        cur_val_ph      = stat_cols[1].empty()   # «Текущий: 0.832»
        best_val_ph     = stat_cols[2].empty()   # «Лучший: 0.847»
        pct_ph          = stat_cols[3].empty()   # «Прогресс: 75%»
        status_ph       = prog_container.empty() # статусная строка

        # Коллбэк вызывается Optuna после каждого trial
        def _progress_cb(trial_num: int, total: int,
                         trial_val: float, best_val: float):
            pct = trial_num / total
            # text= отображается рядом с прогресс-баром
            prog_bar.progress(pct, text=f"🔍 Optuna: Trial {trial_num}/{total}")
            cur_trial_ph.metric("Trial", f"{trial_num}/{total}")
            cur_val_ph.metric("Текущий", f"{trial_val:.4f}")
            best_val_ph.metric("Лучший 🏆", f"{best_val:.4f}")
            pct_ph.metric("Прогресс", f"{pct*100:.0f}%")

        with st.spinner(f"Обучаю {selected_model}..."):
            engine  = UniversalMLEngine(model_type=selected_model)
            metrics = engine.train_and_evaluate(
                df, target_col,
                n_trials=n_trials,
                use_cv=use_cv,
                cv_folds=cv_folds,
                hp_ranges=hp_ranges_user,
                progress_callback=_progress_cb,
            )
            explanation = engine.generate_human_explanation()
            engine.save_model("model.pkl")

        # Завершаем прогресс-бар
        prog_bar.progress(1.0, text="✅ Обучение завершено!")
        status_ph.success(f"Все {n_trials} trial(s) Optuna завершены. "
                          f"Лучшие параметры: {engine.best_params}")

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

    # ── Результаты ───────────────────────────────────────────────────────────
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

        # ── Важность признаков ────────────────────────────────────────────
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
                feat_names_fi = (list(preprocessor.transformers_[0][2]) +
                                 list(preprocessor.transformers_[1][2]))
                fi_df = (pd.DataFrame({"Признак": feat_names_fi, "Важность": importances})
                           .sort_values("Важность", ascending=True).tail(15))
                st.divider(); st.subheader("📌 Важность признаков")
                fig_fi = px.bar(fi_df, x="Важность", y="Признак", orientation="h",
                                title="Топ-15 признаков", color="Важность",
                                color_continuous_scale="Blues", text_auto=".3f")
                fig_fi.update_layout(showlegend=False, height=max(300, 30*len(fi_df)))
                st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            pass

        # ── Confusion Matrix ─────────────────────────────────────────────
        if st.session_state.task_type == "classification" and st.session_state.conf_matrix is not None:
            st.divider(); st.subheader("📉 Матрица ошибок")
            cm     = np.array(st.session_state.conf_matrix)
            labels = [str(l) for l in st.session_state.class_labels]
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm  = np.where(row_sums > 0, cm/row_sums*100, 0).round(1)
            fig_cm = px.imshow(cm_norm, x=labels, y=labels,
                               color_continuous_scale="Blues",
                               labels=dict(x="Предсказано", y="Факт", color="%"),
                               title="Confusion Matrix (% от строки)", aspect="auto")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    fig_cm.add_annotation(x=labels[j], y=labels[i],
                                          text=f"{cm[i][j]}<br>({cm_norm[i][j]}%)",
                                          showarrow=False, font=dict(size=13, color="black"))
            fig_cm.update_layout(height=max(300, 100*len(labels)))
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Learning Curve ───────────────────────────────────────────────
        lc_data = st.session_state.get('lc_data')
        if lc_data:
            st.divider(); st.subheader("📈 Кривая обучения (Learning Curve)")
            st.caption(
                "**Синяя** — качество на обучении, **красная** — на валидации (CV). "
                "Широкая полоса = высокая дисперсия между фолдами."
            )
            sizes  = lc_data['train_sizes']
            tr_m   = lc_data['train_mean']; tr_s = lc_data['train_std']
            vl_m   = lc_data['val_mean'];   vl_s = lc_data['val_std']
            sc_name= lc_data['scoring'].upper()
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=sizes+sizes[::-1],
                y=[m+s for m,s in zip(tr_m,tr_s)]+[m-s for m,s in zip(tr_m[::-1],tr_s[::-1])],
                fill='toself', fillcolor='rgba(55,138,221,0.15)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig_lc.add_trace(go.Scatter(
                x=sizes+sizes[::-1],
                y=[m+s for m,s in zip(vl_m,vl_s)]+[m-s for m,s in zip(vl_m[::-1],vl_s[::-1])],
                fill='toself', fillcolor='rgba(231,76,60,0.12)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig_lc.add_trace(go.Scatter(x=sizes, y=tr_m, mode='lines+markers',
                name='Обучение', line=dict(color='#378add', width=2), marker=dict(size=7)))
            fig_lc.add_trace(go.Scatter(x=sizes, y=vl_m, mode='lines+markers',
                name='Валидация (CV)', line=dict(color='#e74c3c', width=2), marker=dict(size=7)))
            fig_lc.update_layout(title=f'Learning Curve — {sc_name}',
                xaxis_title='Размер обучающей выборки (строк)',
                yaxis_title=sc_name, legend=dict(orientation='h', y=-0.2), height=380)
            st.plotly_chart(fig_lc, use_container_width=True)

            gap = round(tr_m[-1] - vl_m[-1], 3)
            val_final = round(vl_m[-1], 3)
            bp = st.session_state.metrics

            if gap > 0.15:
                # Даём конкретные советы по борьбе с переобучением на основе типа модели
                tips = {
                    "Random Forest":
                        "Уменьши `max_depth` (до 5–7), увеличь `min_samples_leaf` (до 10–20).",
                    "Gradient Boosting":
                        "Уменьши `subsample` (до 0.5–0.7), снизь `max_depth` (до 3–5), "
                        "уменьши `learning_rate`.",
                    "Logistic Regression":
                        "Уменьши `C` (до 0.01–0.1) или увеличь `alpha` (до 10–50).",
                    "Ансамбль (Ensemble)": "Попробуй классическую модель с регуляризацией.",
                }
                tip = tips.get(st.session_state.trained_model_name, "")
                st.error(
                    f"🔴 **Переобучение** — разрыв Train–Val = **{gap:.3f}**. "
                    f"Модель запомнила обучающие данные, но плохо обобщает.\n\n"
                    f"**Как исправить:** {tip} Или увеличь данные / включи CV."
                )
            elif val_final < 0.6 and lc_data['scoring'] == 'accuracy':
                st.warning(
                    f"🟡 **Недообучение** — Val {sc_name} = **{val_final:.3f}**. "
                    "Попробуй: увеличить n_estimators, depth, добавить признаки через Feature Engineering."
                )
            elif gap <= 0.05:
                st.success(
                    f"🟢 **Хороший баланс** — разрыв Train–Val = **{gap:.3f}**. "
                    "Модель хорошо обобщает и не переобучена."
                )
            else:
                st.info(
                    f"🔵 **Умеренный разрыв** Train–Val = **{gap:.3f}**. "
                    "Небольшое переобучение. Попробуй зауженные диапазоны гиперпараметров выше."
                )

        # ── SHAP ─────────────────────────────────────────────────────────
        st.divider(); st.subheader("🔍 SHAP Waterfall")
        sh1, sh2 = st.columns([2,1])
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
                se.task_type = data.get("task_type","classification")
                result = se.compute_shap_values(shap_row_df)
            if result is not None:
                sv, bv, fn = result
                top_n = min(12, len(sv))
                idx   = np.argsort(np.abs(sv))[::-1][:top_n]
                sv_t  = sv[idx]; fn_t = [fn[i] for i in idx]
                running = bv
                measures = ["absolute"]; texts = [f"{bv:.3f}"]
                for v in sv_t:
                    measures.append("relative"); texts.append(f"{v:+.3f}"); running += v
                measures.append("total"); texts.append(f"{running:.3f}")
                fig_shap = go.Figure(go.Waterfall(
                    orientation="h", measure=measures,
                    x=[bv]+list(sv_t)+[running],
                    y=["Базовое значение"]+fn_t+["Итоговое предсказание"],
                    text=texts, textposition="outside",
                    connector={"line":{"color":"rgba(63,63,63,0.4)"}},
                    decreasing={"marker":{"color":"#3498db"}},
                    increasing={"marker":{"color":"#e74c3c"}},
                    totals={"marker":{"color":"#2ecc71"}},
                ))
                fig_shap.update_layout(title=f"SHAP: топ-{top_n} признаков",
                    xaxis_title="Значение", height=max(400, 40*(top_n+2)),
                    margin=dict(l=20, r=100, t=60, b=40))
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.warning("⚠️ SHAP недоступен для данной комбинации модели и задачи.")

        # ── Тестер ───────────────────────────────────────────────────────
        st.divider(); st.subheader("🧪 Встроенный тестер модели")
        train_df_snap = st.session_state.train_df

        def _make_widget(feat, col_widget, key_prefix):
            if train_df_snap is not None and feat in train_df_snap.columns:
                col_data = train_df_snap[feat].dropna()
                dtype    = col_data.dtype
                n_unique = col_data.nunique()
                if dtype in ['object','bool'] or str(dtype)=='category':
                    return col_widget.selectbox(feat,
                        sorted(col_data.unique().tolist(), key=str), key=f"{key_prefix}_{feat}")
                if n_unique <= 15:
                    return col_widget.selectbox(feat,
                        sorted(col_data.unique().tolist()), key=f"{key_prefix}_{feat}")
                median_val = col_data.median()
                if dtype in ['int32','int64']:
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
                                             "Вероятность": [round(float(p),4) for p in proba]})
                    fig_p = px.bar(proba_df, x="Класс", y="Вероятность",
                                   color="Вероятность", color_continuous_scale="Blues",
                                   range_y=[0,1], title="Вероятности по классам", text_auto=".3f")
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    val_str = f"{pred_py:.4f}" if isinstance(pred_py, float) else str(pred_py)
                    st.success(f"**Предсказание: {val_str}**")
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")

        st.divider()
        try:
            with open("model.pkl","rb") as f_pkl:
                model_bytes = f_pkl.read()
            st.download_button(
                label="💾 Скачать модель (model.pkl)", data=model_bytes,
                file_name=f"model_{st.session_state.trained_model_name.replace(' ','_')}.pkl",
                mime="application/octet-stream")
            st.caption("`data = joblib.load('model.pkl'); pred = data['model'].predict(X)`")
        except FileNotFoundError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 3: НЕЙРОСЕТИ (ИНС)
# ══════════════════════════════════════════════════════════════════════════════
with tab_nn:
    st.write("Обучи нейронную сеть на тех же данных и сравни с классическими алгоритмами.")

    nn_col1, nn_col2 = st.columns([1, 2])
    with nn_col1:
        available_nn = ["sklearn MLP"]
        if PYTORCH_AVAILABLE:
            available_nn.append("PyTorch MLP")
        if TABNET_AVAILABLE:
            available_nn.append("TabNet")

        if not PYTORCH_AVAILABLE:
            err_detail = f"\n\n`{_PYTORCH_ERROR}`" if _PYTORCH_ERROR else ""
            st.warning(
                "⚠️ **PyTorch недоступен.**\n```\n"
                "pip uninstall torch -y\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cpu\n```"
                + err_detail)
        elif not TABNET_AVAILABLE:
            st.caption("ℹ️ TabNet: `pip install pytorch-tabnet`")

        nn_model_type = st.selectbox("Архитектура:", options=available_nn)
    with nn_col2:
        nn_target_col = st.selectbox("Целевая колонка:", df.columns, key="nn_target")

    st.divider()

    # ── Параметры каждой архитектуры ─────────────────────────────────────────
    if nn_model_type == "sklearn MLP":
        st.markdown("**Параметры sklearn MLP**")
        c1, c2, c3, c4 = st.columns(4)
        layer1   = c1.number_input("Нейроны слой 1", 16, 512, 128, step=16)
        layer2   = c2.number_input("Нейроны слой 2 (0=один слой)", 0, 512, 64, step=16)
        max_iter = c3.number_input("Макс. итераций", 50, 1000, 300, step=50)
        lr_sk    = c4.number_input("Learning rate", 0.0001, 0.1, 0.001, step=0.0001,
                                   format="%.4f")
        hidden   = (layer1,) if layer2 == 0 else (layer1, layer2)
        st.caption(f"Архитектура: вход → {' → '.join(str(s) for s in hidden)} → выход | "
                   f"Early stopping · lr={lr_sk}")

    elif nn_model_type == "PyTorch MLP":
        st.markdown("**Параметры PyTorch MLP**")
        c1, c2, c3, c4 = st.columns(4)
        pt_l1      = c1.number_input("Слой 1", 32, 512, 256, step=32)
        pt_l2      = c2.number_input("Слой 2", 0, 512, 128, step=32)
        pt_l3      = c3.number_input("Слой 3 (0=пропустить)", 0, 512, 64, step=32)
        pt_dropout = c4.slider("Dropout", 0.0, 0.7, 0.3, step=0.1,
                               help="0.3 = 30% нейронов случайно отключаются при обучении → регуляризация")
        c5, c6, c7 = st.columns(3)
        pt_lr      = c5.number_input("Learning rate", 0.0001, 0.05, 0.001, step=0.0001,
                                     format="%.4f")
        pt_epochs  = c6.slider("Макс. эпох", 20, 500, 100, step=10)
        pt_patience= c7.slider("Early stopping patience", 5, 50, 15,
                               help="Эпох без улучшения val_loss до остановки")
        hidden     = tuple(d for d in [pt_l1, pt_l2, pt_l3] if d > 0)
        st.caption(
            f"Архитектура: вход → {' → '.join(str(d) for d in hidden)} → выход · "
            f"BatchNorm + Dropout({pt_dropout}) · Adam lr={pt_lr} · patience={pt_patience}"
        )

    elif nn_model_type == "TabNet":
        st.markdown("**Параметры TabNet**")
        tb1, tb2, tb3, tb4 = st.columns(4)
        tb_steps   = tb1.slider("n_steps (шаги внимания)", 2, 6, 3,
                                help="Сколько раз сеть последовательно выбирает признаки")
        tb_nd      = tb2.slider("n_d / n_a (размерности)", 8, 64, 16, step=8)
        tb_epochs  = tb3.slider("Макс. эпох", 20, 300, 100, step=10)
        tb_patience= tb4.slider("Patience", 5, 50, 15)
        st.caption(f"TabNet: {tb_steps} шагов × n_d={tb_nd} · patience={tb_patience}")

    # ── Кнопка запуска ────────────────────────────────────────────────────────
    if st.button("🚀 Обучить нейросеть", use_container_width=True):

        # ── КОНТЕЙНЕРЫ ПРОГРЕССА ─────────────────────────────────────────────
        nn_prog_box   = st.container()
        nn_prog_bar   = nn_prog_box.progress(0, text="Инициализация...")
        nn_stat_cols  = nn_prog_box.columns(4)
        nn_epoch_ph   = nn_stat_cols[0].empty()   # «Эпоха X / N»
        nn_tl_ph      = nn_stat_cols[1].empty()   # Train Loss
        nn_vl_ph      = nn_stat_cols[2].empty()   # Val Loss
        nn_eta_ph     = nn_stat_cols[3].empty()   # % прогресса
        nn_chart_ph   = nn_prog_box.empty()        # Live Loss Curve (обновляется каждые N эпох)

        # Хранилище для live-графика
        _live_tl: list = []
        _live_vl: list = []

        def _nn_epoch_cb(epoch: int, max_ep: int,
                         train_loss: float, val_loss: float,
                         train_hist: list, val_hist: list):
            """
            Коллбэк вызывается из nn_engine каждые CALLBACK_EVERY эпох.
            Обновляет прогресс-бар и live-график прямо во время обучения.
            """
            pct = min(epoch / max(max_ep, 1), 1.0)
            nn_prog_bar.progress(pct, text=f"⚡ Эпоха {epoch}/{max_ep}")
            nn_epoch_ph.metric("Эпоха", f"{epoch}/{max_ep}")
            nn_tl_ph.metric("Train Loss", f"{train_loss:.4f}")
            nn_vl_ph.metric("Val Loss", f"{val_loss:.4f}")
            nn_eta_ph.metric("Прогресс", f"{pct*100:.0f}%")

            # Обновляем live-график Loss Curve
            if len(train_hist) >= 2:
                ep_x = list(range(1, len(train_hist)+1))
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(
                    x=ep_x, y=train_hist, mode='lines', name='Train Loss',
                    line=dict(color='#378add', width=2)))
                if val_hist:
                    fig_live.add_trace(go.Scatter(
                        x=ep_x, y=val_hist, mode='lines', name='Val Loss',
                        line=dict(color='#e74c3c', width=2, dash='dash')))
                fig_live.update_layout(
                    title=f"Loss Curve (live) — эпоха {epoch}/{max_ep}",
                    xaxis_title="Эпоха", yaxis_title="Loss",
                    legend=dict(orientation='h', y=-0.3),
                    height=280,
                    margin=dict(l=40, r=20, t=40, b=60),
                )
                nn_chart_ph.plotly_chart(fig_live, use_container_width=True)

        with st.spinner(f"Обучаю {nn_model_type}..."):
            try:
                if nn_model_type == "sklearn MLP":
                    engine_nn = SklearnMLPEngine(
                        hidden_layers=hidden,
                        max_iter=int(max_iter),
                        learning_rate_init=float(lr_sk),
                    )
                elif nn_model_type == "PyTorch MLP":
                    if not PYTORCH_AVAILABLE:
                        st.error("PyTorch недоступен."); st.stop()
                    engine_nn = PyTorchMLPEngine(
                        hidden_dims=hidden,
                        dropout=pt_dropout,
                        lr=float(pt_lr),
                        max_epochs=int(pt_epochs),
                        patience=int(pt_patience),
                    )
                elif nn_model_type == "TabNet":
                    engine_nn = TabNetEngine(
                        n_steps=int(tb_steps),
                        n_d=int(tb_nd),
                        n_a=int(tb_nd),
                        max_epochs=int(tb_epochs),
                        patience=int(tb_patience),
                    )

                # sklearn MLP не поддерживает per-epoch callback — передаём None,
                # для PyTorch MLP и TabNet передаём _nn_epoch_cb
                cb = None if nn_model_type == "sklearn MLP" else _nn_epoch_cb

                # Для sklearn MLP — анимированный спиннер с шагами
                if nn_model_type == "sklearn MLP":
                    nn_prog_bar.progress(0.1, text="Препроцессинг данных...")
                    nn_epoch_ph.metric("Статус", "Обучение...")
                    nn_tl_ph.metric("Архитектура", str(hidden))
                    nn_vl_ph.metric("Early stopping", "включён")
                    nn_eta_ph.metric("Итераций макс.", int(max_iter))

                nn_metrics = engine_nn.train_and_evaluate(
                    df, nn_target_col, epoch_callback=cb)

                # Для sklearn MLP достраиваем Loss Curve из history после обучения
                if nn_model_type == "sklearn MLP":
                    nn_prog_bar.progress(1.0, text="✅ Обучение завершено!")
                    th = engine_nn.train_history or {}
                    tl = th.get('train_loss', [])
                    vl = th.get('val_loss', [])
                    n_it = th.get('n_iter', 0)
                    if tl:
                        ep_x = list(range(1, len(tl)+1))
                        fig_sk = go.Figure()
                        fig_sk.add_trace(go.Scatter(x=ep_x, y=tl, mode='lines',
                            name='Train Loss', line=dict(color='#378add', width=2)))
                        if vl:
                            fig_sk.add_trace(go.Scatter(x=ep_x, y=vl, mode='lines',
                                name='Val Loss (1−score)', line=dict(color='#e74c3c', width=2, dash='dash')))
                        fig_sk.update_layout(
                            title=f"Loss Curve — sklearn MLP (итераций: {n_it})",
                            xaxis_title="Итерация", yaxis_title="Loss",
                            legend=dict(orientation='h', y=-0.3),
                            height=280, margin=dict(l=40, r=20, t=40, b=60))
                        nn_chart_ph.plotly_chart(fig_sk, use_container_width=True)
                    nn_epoch_ph.metric("Итераций", n_it)

                engine_nn.save_model("model_nn.pkl")

                st.session_state["nn_metrics"]      = nn_metrics
                st.session_state["nn_explanation"]  = engine_nn.generate_human_explanation()
                st.session_state["nn_model_type"]   = nn_model_type
                st.session_state["nn_task_type"]    = engine_nn.task_type
                st.session_state["nn_conf_matrix"]  = engine_nn.conf_matrix
                st.session_state["nn_class_labels"] = engine_nn.class_labels
                st.session_state["nn_fi"] = (
                    engine_nn.feature_importances_
                    if hasattr(engine_nn, "feature_importances_") else None)
                st.session_state["nn_history"]     = getattr(engine_nn, "train_history", None)
                st.session_state["nn_max_epochs"]  = (
                    int(max_iter) if nn_model_type == "sklearn MLP"
                    else int(pt_epochs) if nn_model_type == "PyTorch MLP"
                    else int(tb_epochs))

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
                nn_prog_bar.progress(1.0, text="❌ Ошибка")
                st.error(f"Ошибка обучения: {e}")

    # ── Результаты ───────────────────────────────────────────────────────────
    if st.session_state.get("nn_metrics"):
        nn_task  = st.session_state["nn_task_type"]
        nn_icon  = "🔵" if nn_task == "classification" else "📈"
        nn_label = "классификация" if nn_task == "classification" else "регрессия"
        st.success(f"✅ **{st.session_state['nn_model_type']}** обучена! {nn_icon} {nn_label}")

        nn1, nn2 = st.columns(2)
        with nn1:
            st.subheader("📊 Метрики ИНС")
            nm_cols = st.columns(min(len(st.session_state["nn_metrics"]), 3))
            for i, (k, v) in enumerate(st.session_state["nn_metrics"].items()):
                nm_cols[i % 3].metric(k, v)
        with nn2:
            st.subheader("🧠 Архитектура")
            st.info(st.session_state["nn_explanation"])

        # ── Финальный Loss Curve (после обучения) ─────────────────────────
        nn_history = st.session_state.get("nn_history")
        if nn_history and nn_history.get('train_loss'):
            st.divider(); st.subheader("📉 Loss Curve (финальный)")
            st.caption(
                "**Синяя** — лосс на обучении, **красная пунктирная** — на валидации. "
                "Зелёная вертикаль — лучшая эпоха. Широкий разрыв train << val → переобучение."
            )
            tl_fin = nn_history['train_loss']
            vl_fin = nn_history['val_loss']
            ep_fin = list(range(1, len(tl_fin)+1))

            fig_final = go.Figure()
            fig_final.add_trace(go.Scatter(x=ep_fin, y=tl_fin, mode='lines',
                name='Train Loss', line=dict(color='#378add', width=2)))
            if vl_fin:
                fig_final.add_trace(go.Scatter(x=ep_fin, y=vl_fin, mode='lines',
                    name='Val Loss', line=dict(color='#e74c3c', width=2, dash='dash')))
                best_ep = int(np.argmin(vl_fin)) + 1
                fig_final.add_vline(x=best_ep, line_dash='dot', line_color='#2ecc71',
                    annotation_text=f'Лучшая эпоха: {best_ep}',
                    annotation_position='top right')
            fig_final.update_layout(
                title='Динамика лосса по эпохам (финальный)',
                xaxis_title='Эпоха / Итерация', yaxis_title='Loss',
                legend=dict(orientation='h', y=-0.2), height=360)
            st.plotly_chart(fig_final, use_container_width=True)

            # Интерпретация
            ft = tl_fin[-1]; fv = vl_fin[-1] if vl_fin else ft
            gap = round(ft - fv, 4)
            n_it = nn_history['n_iter']
            max_ep_cfg = st.session_state.get('nn_max_epochs', n_it+1)

            if n_it < max_ep_cfg:
                st.success(f"🟢 **Early stopping** сработал на эпохе **{n_it}** из {max_ep_cfg}.")
            if abs(gap) > 0.3:
                st.error(
                    f"🔴 **Переобучение** — Train Loss {ft:.4f} << Val Loss {fv:.4f} (разрыв {abs(gap):.4f}). "
                    "Попробуй: увеличить **Dropout**, уменьшить размер слоёв, снизить число эпох.")
            elif ft > 0.5 and fv > 0.5:
                st.warning(
                    "🟡 **Недообучение** — оба лосса высокие. "
                    "Попробуй увеличить число нейронов, слоёв или эпох.")
            else:
                st.info(f"🔵 Train Loss: **{ft:.4f}** | Val Loss: **{fv:.4f}** | Разрыв: {abs(gap):.4f}")

        # ── Сравнение с классическим ML ───────────────────────────────────
        if st.session_state.is_trained:
            st.divider(); st.subheader("⚖️ Сравнение: ИНС vs Классический ML")
            classic_m   = st.session_state.metrics
            nn_m        = st.session_state["nn_metrics"]
            common_keys = [k for k in classic_m if k in nn_m]
            if common_keys:
                cmp_df = pd.DataFrame({
                    "Метрика": common_keys,
                    st.session_state.trained_model_name: [classic_m[k] for k in common_keys],
                    st.session_state["nn_model_type"]:   [nn_m[k]      for k in common_keys],
                })
                cmp_melt = cmp_df.melt(id_vars="Метрика", var_name="Модель", value_name="Значение")
                fig_cmp = px.bar(cmp_melt, x="Метрика", y="Значение", color="Модель",
                    barmode="group",
                    title=f"{st.session_state.trained_model_name} vs {st.session_state['nn_model_type']}",
                    text_auto=".3f",
                    color_discrete_map={
                        st.session_state.trained_model_name: "#378add",
                        st.session_state["nn_model_type"]:   "#e74c3c",
                    })
                st.plotly_chart(fig_cmp, use_container_width=True)
                fk = common_keys[0]
                cv, nv = classic_m[fk], nn_m[fk]
                margin = abs(nv - cv)
                if margin < 0.005: st.info(f"🤝 Результаты практически одинаковы (Δ={margin:.3f}).")
                elif nv > cv:      st.success(f"🏆 Нейросеть выигрывает по {fk}: **{nv}** vs {cv}")
                else:              st.warning(f"⚡ Классический ML выигрывает по {fk}: **{cv}** vs {nv}")
        else:
            st.info("💡 Обучи классическую модель во вкладке '⚙️ Обучение' чтобы сравнить.")

        # ── Confusion Matrix ИНС ──────────────────────────────────────────
        if nn_task == "classification" and st.session_state.get("nn_conf_matrix") is not None:
            st.divider(); st.subheader("📉 Confusion Matrix (нейросеть)")
            cm     = np.array(st.session_state["nn_conf_matrix"])
            labels = [str(l) for l in st.session_state["nn_class_labels"]]
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm  = np.where(row_sums > 0, cm/row_sums*100, 0).round(1)
            fig_nn_cm = px.imshow(cm_norm, x=labels, y=labels,
                color_continuous_scale="Reds",
                labels=dict(x="Предсказано", y="Факт", color="%"),
                title="Confusion Matrix ИНС", aspect="auto")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    fig_nn_cm.add_annotation(x=labels[j], y=labels[i],
                        text=f"{cm[i][j]}<br>({cm_norm[i][j]}%)",
                        showarrow=False, font=dict(size=13, color="black"))
            fig_nn_cm.update_layout(height=max(300, 100*len(labels)))
            st.plotly_chart(fig_nn_cm, use_container_width=True)

        # ── TabNet feature importances ────────────────────────────────────
        nn_fi = st.session_state.get("nn_fi")
        if nn_fi is not None and len(nn_fi) > 0:
            st.divider(); st.subheader("📌 Важность признаков (TabNet attention)")
            fi_names = df.drop(columns=[nn_target_col]).columns.tolist()
            if len(fi_names) == len(nn_fi):
                fi_df = (pd.DataFrame({"Признак": fi_names, "Важность": nn_fi})
                           .sort_values("Важность", ascending=True).tail(15))
                fig_fi_nn = px.bar(fi_df, x="Важность", y="Признак", orientation="h",
                    title="Важность признаков по механизму внимания TabNet",
                    color="Важность", color_continuous_scale="Reds", text_auto=".3f")
                fig_fi_nn.update_layout(showlegend=False, height=max(300, 30*len(fi_df)))
                st.plotly_chart(fig_fi_nn, use_container_width=True)
                st.caption("Важность = среднее внимание по всем шагам sequential attention.")


# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 4: ИСТОРИЯ ЭКСПЕРИМЕНТОВ
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("📜 История экспериментов")
    history = st.session_state.experiment_history

    if not history:
        st.info("Пока нет запущенных экспериментов. Обучи хотя бы одну модель.")
    else:
        display_cols = [k for k in history[0].keys()
                        if k not in {"best_params","cleaning_log"}]
        hist_df = pd.DataFrame(history)[display_cols]
        st.dataframe(hist_df, use_container_width=True)

        hc1, hc2 = st.columns([1,1])
        with hc1:
            if st.button("🗑️ Очистить историю"):
                st.session_state.experiment_history = []; st.rerun()
        with hc2:
            st.download_button("💾 Скачать историю (CSV)",
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
        st.subheader("🐍 Скачать Python-скрипт для выбранного эксперимента")
        st.write(
            "Выбери эксперимент — получишь воспроизводимый `.py` файл. "
            "**Классический ML**: EDA + очистка + Optuna-параметры + оценка. "
            "**Нейросети**: архитектура + цикл обучения + loss curve."
        )

        exp_labels = [
            f"{i+1}. {r['⏰ Время']} | {r['Модель']} | target={r['Target']} | {r['CV']}"
            for i, r in enumerate(history)
        ]
        selected_idx    = st.selectbox("Выбери эксперимент:", range(len(exp_labels)),
                                       format_func=lambda i: exp_labels[i])
        selected_record = history[selected_idx]
        is_nn_record    = selected_record.get("Модель","").startswith("ИНС:")

        with st.expander("ℹ️ Параметры эксперимента", expanded=False):
            st.json({k:v for k,v in selected_record.items()
                     if k not in {"best_params","cleaning_log"}})
            bp = selected_record.get("best_params",{})
            if bp and "Инфо" not in bp:
                st.markdown("**Параметры модели:**"); st.json(bp)
            cl = selected_record.get("cleaning_log",[])
            if cl:
                st.markdown(f"**Шагов очистки:** {len(cl)}")
                for step in cl:
                    st.write(f"- `{step['op']}`:", {k:v for k,v in step.items() if k!='op'})

        badge = "🧠 Нейросеть: **" + selected_record["Модель"].replace("ИНС: ","") + "**" \
            if is_nn_record else "⚙️ Классический ML: **" + selected_record["Модель"] + "**"
        st.info(badge)

        if st.button("⬇️ Сгенерировать и скачать .py скрипт", use_container_width=True):
            if is_nn_record:
                script_code = generate_nn_script(selected_record, dataset_filename)
                nn_safe = selected_record["Модель"].replace("ИНС: ","").replace(" ","_")
                fname = f"nn_solution_{nn_safe}_{selected_record['Target']}.py"
            else:
                script_code = generate_script(
                    selected_record,
                    selected_record.get("cleaning_log",[]),
                    dataset_filename)
                ms = (selected_record["Модель"]
                      .replace(" ","_").replace("(","").replace(")",""))
                fname = f"ml_solution_{ms}_{selected_record['Target']}.py"

            st.download_button("📥 Скачать .py файл", data=script_code.encode("utf-8"),
                               file_name=fname, mime="text/x-python", use_container_width=True)
            st.divider()
            st.markdown("**Превью скрипта:**")
            st.code(script_code, language="python")