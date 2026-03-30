"""
ml_engine.py — Ядро классического машинного обучения для Universal ML Platform.

Полный жизненный цикл эксперимента:
    1. Автоопределение типа задачи (классификация / регрессия)
    2. Препроцессинг: медианная импутация → OrdinalEncoder → StandardScaler
    3. Байесовская оптимизация гиперпараметров через Optuna (TPE Sampler)
    4. K-Fold Cross-Validation — опционально (StratifiedKFold / KFold)
    5. Оценка метрик + confusion matrix
    6. Learning Curve — диагностика переобучения / недообучения
    7. SHAP-объяснения (TreeExplainer / LinearExplainer)
    8. Сохранение в pkl через joblib

"""
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,   # hold-out разбивка
    cross_val_score,    # K независимых оценок качества
    StratifiedKFold,    # K-Fold с сохранением пропорций классов в каждом фолде
    KFold,              # обычный K-Fold (для регрессии, где стратификация неприменима)
    learning_curve,     # метрики при разных размерах обучающей выборки (20%→100%)
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,        # ансамбль: усредняем вероятности нескольких классификаторов
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,         # ансамбль: усредняем числовые предсказания регрессоров
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error)
from sklearn.compose import ColumnTransformer    # разные трансформеры на разные колонки
from sklearn.pipeline import Pipeline            # цепочка: препроцессор → модель
from sklearn.impute import SimpleImputer         # заполнение NaN: медиана / мода
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import optuna
import joblib
import warnings

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# Дефолтные диапазоны поиска для Optuna
# ══════════════════════════════════════════════════════════════════════════════
def _default_hp_ranges(model_type: str) -> dict:
    """
    Дефолтные диапазоны гиперпараметров для байесового поиска Optuna.

    Пользовательские диапазоны из UI перекрывают эти значения через dict.update().
    Это позволяет передавать только изменённые параметры, не перечисляя все.

    Ключи с tuple (lo, hi)  — Optuna будет искать в этом диапазоне.
    Ключи с scalar value   — фиксированное значение (не ищется Optuna).

    """
    defaults = {
        "Random Forest": {
            'n_estimators':       (50, 300),
            'max_depth':          (3, 15),
            'min_samples_leaf':   (1, 5),    # больше = меньше переобучения (сглаживание)
            'min_samples_split':  (2, 10),
            'max_features':       'sqrt',    # стандарт RF; можно 'log2' или 1.0
        },
        "Gradient Boosting": {
            'n_estimators':       (50, 300),
            'learning_rate':      (0.01, 0.3),
            'max_depth':          (3, 10),   # GB деревья мелкие намеренно (слабые ученики)
            'subsample':          (0.7, 1.0),# < 1.0 → стохастический GB → регуляризация
            'min_samples_leaf':   (1, 5),
        },
        "Logistic Regression": {
            'C':     (0.01, 20.0),   # для классификации: C = 1/λ
            'alpha': (0.01, 50.0),   # для регрессии (Ridge): λ
        },
    }
    return defaults.get(model_type, {})


# ══════════════════════════════════════════════════════════════════════════════
# Основной класс
# ══════════════════════════════════════════════════════════════════════════════
class UniversalMLEngine:
    """
    Оркестратор одного ML-эксперимента. Один экземпляр = один запуск обучения.

    После вызова train_and_evaluate() доступны:
        .pipeline             — sklearn Pipeline (препроцессор + модель), идёт в pkl
        .features             — список признаков в порядке обучения
        .task_type            — 'classification' | 'regression'
        .best_params          — оптимальные гиперпараметры Optuna
        .conf_matrix          — numpy N×N (только классификация)
        .class_labels         — sorted(y.unique())
        .learning_curve_data  — dict для графика Learning Curve

    """
    def __init__(self, model_type: str = "Random Forest"):
        self.model_type  = model_type    # ключ для выбора алгоритма

        # Заполняются в train_and_evaluate():
        self.pipeline      = None
        self.features      = []          # для reindex при инференсе
        self.best_params   = {}
        self.task_type     = None

        # Артефакты для UI-визуализации:
        self.y_test              = None
        self.y_pred              = None
        self.class_labels        = None
        self.conf_matrix         = None
        self.cv_scores           = None  # numpy-массив length K
        self.cv_mean             = None  # float: среднее по фолдам
        self.cv_std              = None  # float: std (низкий = стабильная модель)
        self.learning_curve_data = None

    # ── Автоопределение типа задачи ───────────────────────────────────────────
    def detect_task_type(self, y: pd.Series) -> str:
        if y.dtype in ['float64', 'float32']:
            return 'regression'
        if y.nunique() > 20:
            return 'regression'
        return 'classification'

    # ── Главный метод ─────────────────────────────────────────────────────────
    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_trials: int = 10,
        use_cv: bool = False,
        cv_folds: int = 5,
        hp_ranges: dict = None,
        progress_callback=None,
    ) -> dict:
        """
        Полный пайплайн обучения.

        hp_ranges:
            Словарь диапазонов поиска Optuna. Примеры:
              {'max_depth': (2, 6)}               — ограничить глубину для борьбы с overfitting
              {'subsample': (0.5, 0.8)}           — усилить регуляризацию GB
              {'min_samples_leaf': (10, 50)}      — крупные листья для шумных данных
              {'max_features': 'log2'}            — фиксированный параметр RF

        progress_callback:
            Вызывается после каждого trial Optuna — используется для обновления
            прогресс-бара в Streamlit (st.progress / st.empty)

        """
        # ── 1. Слияние диапазонов ────────────────────────────────────────────
        merged_ranges = {**_default_hp_ranges(self.model_type), **(hp_ranges or {})}

        # ── 2. Очистка от NaN в целевой переменной ──────────────────────────
        n_before  = len(df)
        df        = df.dropna(subset=[target_col])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"[ml_engine] Удалено {n_dropped} строк с NaN в '{target_col}'.")

        # ── 3. X / y и тип задачи ───────────────────────────────────────────
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.features  = list(X.columns)  # порядок сохраняется для reindex при инференсе
        self.task_type = self.detect_task_type(y)

        # ── 4. Разбиение по типу данных ─────────────────────────────────────
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # ── 5. Препроцессор ──────────────────────────────────────────────────
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1)),
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ])

        # ── 6. CV-сплиттер ──────────────────────────────────────────────────
        if self.task_type == 'classification':
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scoring  = 'accuracy'
        else:
            cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scoring  = 'r2'

        # ── 7. Hold-out split ────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ── 8. Целевая функция Optuna ────────────────────────────────────────
        def objective(trial):
            model = (
                self._build_classifier(trial, merged_ranges)
                if self.task_type == 'classification'
                else self._build_regressor(trial, merged_ranges)
            )
            # Pipeline в objective: препроцессор обучается на train-части каждого фолда исключая data leakage
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

            if use_cv:
                scores = cross_val_score(
                    pipe, X, y, cv=cv_splitter, scoring=cv_scoring, n_jobs=1)
                return scores.mean()
            else:
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                return (accuracy_score(y_test, preds)
                        if self.task_type == 'classification'
                        else r2_score(y_test, preds))

        # ── 9. Запуск Optuna / построение ансамбля ──────────────────────────
        if self.model_type == "Ансамбль (Ensemble)":
            final_model  = self._build_ensemble()
            self.best_params = {"Инфо": "VotingEnsemble (RF + GradientBoosting + Ridge/LogReg)"}
        else:
            study = optuna.create_study(direction='maximize')
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            def _optuna_ui_callback(study: optuna.Study, trial: optuna.Trial):
                if progress_callback is not None:
                    best_val = study.best_value if study.best_trial is not None else 0.0
                    progress_callback(
                        trial.number + 1,    # текущий trial (1-based)
                        n_trials,            # итого trial'ов
                        trial.value or 0.0,  # метрика текущего trial
                        best_val,            # лучшая метрика за всё время
                    )

            study.optimize(objective, n_trials=n_trials, callbacks=[_optuna_ui_callback])
            self.best_params = dict(study.best_params)

            if self.model_type == "Random Forest":
                self.best_params['max_features'] = merged_ranges.get('max_features', 'sqrt')

            final_model = (
                self._build_classifier_from_params(self.best_params)
                if self.task_type == 'classification'
                else self._build_regressor_from_params(self.best_params)
            )

        # ── 10. Финальный Pipeline ───────────────────────────────────────────
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', final_model),
        ])

        # ── 11. Финальное обучение ───────────────────────────────────────────
        if use_cv:
            cv_sc = cross_val_score(
                self.pipeline, X, y, cv=cv_splitter, scoring=cv_scoring, n_jobs=1)
            self.cv_scores = cv_sc
            self.cv_mean   = float(cv_sc.mean())
            self.cv_std    = float(cv_sc.std())  # std < 0.02 = стабильная модель

            # Финальная модель обучается на ВСЕХ данных
            self.pipeline.fit(X, y)

            # Отдельный hold-out для confusion matrix:
            _m = (self._build_classifier_from_params(self.best_params)
                  if self.model_type != "Ансамбль (Ensemble)" and self.task_type == 'classification'
                  else (self._build_regressor_from_params(self.best_params)
                        if self.model_type != "Ансамбль (Ensemble)"
                        else self._build_ensemble()))
            pipe_holdout = Pipeline(steps=[('preprocessor', preprocessor), ('model', _m)])
            pipe_holdout.fit(X_train, y_train)
            preds = pipe_holdout.predict(X_test)
        else:
            self.cv_scores = self.cv_mean = self.cv_std = None
            self.pipeline.fit(X_train, y_train)
            preds = self.pipeline.predict(X_test)

        self.y_test = y_test
        self.y_pred = preds

        # ── 12. Метрики ──────────────────────────────────────────────────────
        if self.task_type == 'classification':
            self.class_labels = sorted(y.unique().tolist())
            self.conf_matrix = confusion_matrix(y_test, preds, labels=self.class_labels)

            metrics = {
                "Accuracy":  round(accuracy_score(y_test, preds), 3),
                "Precision": round(precision_score(
                    y_test, preds, average='macro', zero_division=0), 3),
                "Recall":    round(recall_score(
                    y_test, preds, average='macro', zero_division=0), 3),
            }
            if use_cv:
                metrics["CV Accuracy (mean)"] = round(self.cv_mean, 3)
                metrics["CV Accuracy (±std)"] = round(self.cv_std, 3)
        else:
            self.class_labels = None
            self.conf_matrix  = None
            metrics = {
                "R²":   round(r2_score(y_test, preds), 3),
                "MAE":  round(mean_absolute_error(y_test, preds), 3),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            }
            if use_cv:
                metrics["CV R² (mean)"] = round(self.cv_mean, 3)
                metrics["CV R² (±std)"] = round(self.cv_std, 3)

        # ── 13. Learning Curve ───────────────────────────────────────────────
        try:
            lc_cv = (
                StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                if self.task_type == 'classification'
                else KFold(n_splits=3, shuffle=True, random_state=42)
            )
            lc_scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
            lc_sizes, lc_train, lc_val = learning_curve(
                self.pipeline, X, y,
                train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
                cv=lc_cv, scoring=lc_scoring, n_jobs=1,
            )
            self.learning_curve_data = {
                'train_sizes': lc_sizes.tolist(),
                'train_mean':  lc_train.mean(axis=1).tolist(),
                'train_std':   lc_train.std(axis=1).tolist(),
                'val_mean':    lc_val.mean(axis=1).tolist(),
                'val_std':     lc_val.std(axis=1).tolist(),
                'scoring':     lc_scoring,
            }
        except Exception:
            self.learning_curve_data = None    # тихо: edge case с малым датасетом

        return metrics

    # ══════════════════════════════════════════════════════════════════════════
    # SHAP
    # ══════════════════════════════════════════════════════════════════════════
    def compute_shap_values(self, row_df: pd.DataFrame):
        """
        Вычисляет SHAP-значения (SHapley Additive exPlanations) для одного наблюдения.

        Математическая гарантия: sum(shap_values) + base_value = prediction модели.

        Выбор explainer'а автоматически:
          TreeExplainer   для деревьев (RF, GB): точный, O(TLD²) — намного быстрее KernelExplainer
          LinearExplainer для Ridge/LogReg:       аналитически SHAP_i = w_i × (x_i − E[x_i])

        """
        try:
            import shap
        except ImportError:
            print("[SHAP] shap не установлен. pip install shap")
            return None

        preprocessor  = self.pipeline.named_steps['preprocessor']
        model         = self.pipeline.named_steps['model']

        # Имена признаков из ColumnTransformer в порядке: числовые + категориальные
        feature_names = (
            list(preprocessor.transformers_[0][2]) +
            list(preprocessor.transformers_[1][2])
        )

        row_aligned   = row_df.reindex(columns=self.features)
        X_transformed = preprocessor.transform(row_aligned)

        actual_model = model
        if hasattr(model, 'voting') or (
            hasattr(model, 'estimators_') and isinstance(model.estimators_, list)
        ):
            actual_model = model.estimators_[0]

        try:
            if hasattr(actual_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(actual_model)
                shap_raw  = explainer.shap_values(X_transformed)
                base_val  = explainer.expected_value
            else:
                explainer = shap.LinearExplainer(actual_model, X_transformed)
                shap_raw  = explainer.shap_values(X_transformed)
                base_val  = explainer.expected_value

            sv_arr = np.array(shap_raw)
            ev_arr = np.atleast_1d(np.array(base_val).ravel())

            if isinstance(shap_raw, list):
                sv_arr = np.array(shap_raw[-1]); ev_scalar = float(ev_arr[-1])
            elif sv_arr.ndim == 3:
                sv_arr = sv_arr[0, :, -1];       ev_scalar = float(ev_arr[-1])
            elif sv_arr.ndim == 2:
                sv_arr = sv_arr[0]
                ev_scalar = float(ev_arr[-1] if len(ev_arr) > 1 else ev_arr[0])
            else:
                ev_scalar = float(ev_arr[0])

            return np.array(sv_arr), ev_scalar, feature_names

        except Exception as e:
            print(f"[SHAP] Ошибка: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # Построение моделей (для Optuna и финального обучения)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_classifier(self, trial: optuna.Trial, hp_ranges: dict):
        """
        Строит классификатор с гиперпараметрами предложенными текущим trial.

        Optuna (TPE Sampler) не случайный — анализирует прошлые trial'ы и
        предлагает значения в областях с высокой вероятностью улучшения

        """
        if self.model_type == "Random Forest":
            n1, n2   = hp_ranges.get('n_estimators',      (50, 300))
            d1, d2   = hp_ranges.get('max_depth',         (3, 15))
            l1, l2   = hp_ranges.get('min_samples_leaf',  (1, 5))
            s1, s2   = hp_ranges.get('min_samples_split', (2, 10))
            mf       = hp_ranges.get('max_features',      'sqrt')
            return RandomForestClassifier(
                n_estimators      = trial.suggest_int('n_estimators',      n1, n2),
                max_depth         = trial.suggest_int('max_depth',         d1, d2),
                # min_samples_leaf: больше → более гладкие листья → меньше переобучения
                min_samples_leaf  = trial.suggest_int('min_samples_leaf',  l1, l2),
                # min_samples_split: больше → реже разбиваем узлы → менее сложная модель
                min_samples_split = trial.suggest_int('min_samples_split', s1, s2),
                # max_features='sqrt': случайное подпространство декоррелирует деревья RF
                max_features      = mf,
                random_state      = 42,
            )

        elif self.model_type == "Gradient Boosting":
            n1, n2   = hp_ranges.get('n_estimators',     (50, 300))
            d1, d2   = hp_ranges.get('max_depth',        (3, 10))
            lr1, lr2 = hp_ranges.get('learning_rate',   (0.01, 0.3))
            sub1, sub2 = hp_ranges.get('subsample',     (0.7, 1.0))
            l1, l2   = hp_ranges.get('min_samples_leaf', (1, 5))
            return GradientBoostingClassifier(
                n_estimators     = trial.suggest_int('n_estimators',   n1,   n2),
                max_depth        = trial.suggest_int('max_depth',      d1,   d2),
                # learning_rate: малый lr + много деревьев = лучшее качество (медленнее)
                learning_rate    = trial.suggest_float('learning_rate', lr1,  lr2),
                # subsample аналог Dropout в нейросетях — снижает переобучение
                subsample        = trial.suggest_float('subsample',    sub1, sub2),
                min_samples_leaf = trial.suggest_int('min_samples_leaf', l1,  l2),
                random_state     = 42,
            )

        elif self.model_type == "Logistic Regression":
            c1, c2 = hp_ranges.get('C', (0.01, 20.0))
            return LogisticRegression(
                C=trial.suggest_float('C', c1, c2), # Большой C - риск переобучения
                max_iter=1000, random_state=42,
            )

    def _build_regressor(self, trial: optuna.Trial, hp_ranges: dict):
        """Аналог _build_classifier для регрессии. Ridge заменяет LogReg."""
        if self.model_type == "Random Forest":
            n1, n2 = hp_ranges.get('n_estimators',      (50, 300))
            d1, d2 = hp_ranges.get('max_depth',         (3, 15))
            l1, l2 = hp_ranges.get('min_samples_leaf',  (1, 5))
            s1, s2 = hp_ranges.get('min_samples_split', (2, 10))
            mf     = hp_ranges.get('max_features',      'sqrt')
            return RandomForestRegressor(
                n_estimators      = trial.suggest_int('n_estimators',      n1, n2),
                max_depth         = trial.suggest_int('max_depth',         d1, d2),
                min_samples_leaf  = trial.suggest_int('min_samples_leaf',  l1, l2),
                min_samples_split = trial.suggest_int('min_samples_split', s1, s2),
                max_features      = mf, random_state=42,
            )

        elif self.model_type == "Gradient Boosting":
            n1, n2     = hp_ranges.get('n_estimators',     (50, 300))
            d1, d2     = hp_ranges.get('max_depth',        (3, 10))
            lr1, lr2   = hp_ranges.get('learning_rate',    (0.01, 0.3))
            sub1, sub2 = hp_ranges.get('subsample',        (0.7, 1.0))
            l1, l2     = hp_ranges.get('min_samples_leaf', (1, 5))
            return GradientBoostingRegressor(
                n_estimators      = trial.suggest_int('n_estimators',    n1,   n2),
                max_depth         = trial.suggest_int('max_depth',       d1,   d2),
                learning_rate     = trial.suggest_float('learning_rate', lr1,  lr2),
                subsample         = trial.suggest_float('subsample',     sub1, sub2),
                min_samples_leaf  = trial.suggest_int('min_samples_leaf', l1,  l2),
                random_state      = 42,
            )

        elif self.model_type == "Logistic Regression":
            a1, a2 = hp_ranges.get('alpha', (0.01, 50.0))
            return Ridge(alpha=trial.suggest_float('alpha', a1, a2))

    def _build_classifier_from_params(self, params: dict):
        """Финальная модель с уже известными best_params (после Optuna)"""
        if self.model_type == "Random Forest":
            return RandomForestClassifier(**params, random_state=42)
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingClassifier(**params, random_state=42)
        elif self.model_type == "Logistic Regression":
            return LogisticRegression(**params, max_iter=1000, random_state=42)

    def _build_regressor_from_params(self, params: dict):
        """Финальная модель регрессии с уже известными best_params"""
        if self.model_type == "Random Forest":
            return RandomForestRegressor(**params, random_state=42)
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingRegressor(**params, random_state=42)
        elif self.model_type == "Logistic Regression":
            return Ridge(**params)  # params содержит 'alpha'

    def _build_ensemble(self):
        """
        Voting Ensemble из трёх принципиально разных алгоритмов.

        Философия «мудрости толпы»:
          RF       — устойчив к выбросам, хорош на нелинейных паттернах, высокий разброс
          GB       — сильнее RF на сложных зависимостях, чувствителен к шуму
          LogReg   — ловит линейные закономерности, хорошо откалиброван
        Усредняя их предсказания, компенсируем слабости каждого.

        """
        if self.task_type == 'classification':
            return VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(
                    n_estimators=100, max_depth=7, random_state=42)),
                ('gb', GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
                ('lr', LogisticRegression(C=1.0, max_iter=500, random_state=42)),
            ], voting='soft')
        else:
            return VotingRegressor(estimators=[
                ('rf',    RandomForestRegressor(
                    n_estimators=100, max_depth=7, random_state=42)),
                ('gb',    GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
                ('ridge', Ridge(alpha=1.0)),
            ])

    # ══════════════════════════════════════════════════════════════════════════
    # Объяснение и сохранение
    # ══════════════════════════════════════════════════════════════════════════
    def generate_human_explanation(self) -> str:
        """Текстовое объяснение: топ-2 признака + параметры Optuna"""
        model        = self.pipeline.named_steps['model']
        preprocessor = self.pipeline.named_steps['preprocessor']

        if self.model_type == "Ансамбль (Ensemble)":
            t = "классификации" if self.task_type == 'classification' else "регрессии"
            return f"🌟 **Ансамбль** ({t}): консенсус RF + GradientBoosting + Ridge/LogReg."

        ordered_features = (
            list(preprocessor.transformers_[0][2]) +  # числовые
            list(preprocessor.transformers_[1][2])    # категориальные
        )

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return "Интерпретация недоступна для данного типа модели."

        fi   = sorted(zip(ordered_features, importances), key=lambda x: x[1], reverse=True)
        top1 = fi[0]
        top2 = fi[1] if len(fi) > 1 else None

        tl = "классификации" if self.task_type == 'classification' else "регрессии"
        md = ("Ridge Regression (авто-замена)"
              if self.model_type == "Logistic Regression" and self.task_type == 'regression'
              else self.model_type)

        text = f"Модель **{md}** ({tl}) в первую очередь опирается на **{top1[0]}**."
        if top2:
            text += f" Вторая по значимости — **{top2[0]}**."

        ps = ", ".join(
            f"{k}={round(v, 4) if isinstance(v, float) else v}"
            for k, v in self.best_params.items()
        )
        text += f"\n\n🤖 **Optuna:** `{ps}`."
        return text

    def save_model(self, path: str = "model.pkl"):
        """
        Сохраняет Pipeline + метаданные для инференса

        Содержимое pkl-файла:
          'model'        → Pipeline(preprocessor + model) — predict() применяет трансформации автоматически
          'features'     → список признаков для df.reindex(columns=features)
          'task_type'    → 'classification' | 'regression' — для корректного отображения в API
          'class_labels' → для predict_proba и confusion matrix при загрузке

        """
        joblib.dump({
            "model":        self.pipeline,
            "features":     self.features,
            "task_type":    self.task_type,
            "class_labels": self.class_labels,
        }, path)