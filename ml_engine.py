import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.compose import ColumnTransformer  # применяет разные трансформеры к разным колонкам, результаты склеивает горизонтально
from sklearn.pipeline import Pipeline          # цепочка шагов: каждый передаёт результат следующему
from sklearn.impute import SimpleImputer       # заполняет пропуски (NaN): медианой, средним или модой
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import optuna       # фреймворк байесовской оптимизации гиперпараметров
import joblib       # сериализация Python-объектов
import warnings

warnings.filterwarnings('ignore')


class UniversalMLEngine:
    """
    Ядро платформы. Инкапсулирует полный жизненный цикл одного ML-эксперимента:
    препроцессинг -> оптимизация гиперпараметров -> обучение -> оценка -> SHAP -> сохранение.

    Каждый вызов кнопки "Запустить ML пайплайн" создаёт новый экземпляр этого класса.

    """
    def __init__(self, model_type="Random Forest"):
        self.pipeline    = None         # Обученный Pipeline (препроцессор + модель)
        self.features    = []           # Список признаков в том порядке, в котором модель их ожидает
        self.best_params = {}           # Словарь лучших гиперпараметров после оптимизации Optuna
        self.model_type  = model_type
        self.task_type   = None         # 'classification' | 'regression'

        # Артефакты, сохраняемые после обучения — используются для визуализации в UI
        self.y_test      = None         # Истинные значения тестовой выборки
        self.y_pred      = None         # Предсказания модели на тестовой выборке
        self.class_labels = None        # Отсортированный список уникальных классов
        self.conf_matrix  = None        # Матрица ошибок N x N (None при регрессии)

        # Результаты K-Fold CV. None если CV не использовался (use_cv=False)
        self.cv_scores    = None        # numpy-массив длиной K с метрикой каждого фолда
        self.cv_mean      = None        # Среднее по фолдам — более честная оценка качества
        self.cv_std       = None        # Стандартное отклонение — показывает стабильность модели

    # ------------------------------------------------------------------
    # Автоматическое определение типа задачи
    # ------------------------------------------------------------------
    def detect_task_type(self, y: pd.Series) -> str:
        """
        Определяет тип задачи по целевой переменной.

        1. float64/float32 dtype -> регрессия.
            Так как непрерывные величины (цена, температура, площадь) всегда float.
            Edge case: если в int-колонке есть NaN, pandas автоматически конвертирует
        2. Более 20 уникальных значений -> регрессия.
        3. Иначе -> классификация.

        """
        if y.dtype in ['float64', 'float32']:
            return 'regression'
        if y.nunique() > 20:
            return 'regression'
        return 'classification'

    # ------------------------------------------------------------------
    # Главный метод: обучение и оценка
    # ------------------------------------------------------------------
    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_trials: int = 10,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        """
        Оркестрирует весь процесс обучения.

        Параметры:
            df         — датафрейм с признаками и целевой колонкой
            target_col — название целевой колонки
            n_trials   — количество итераций Optuna
            use_cv     — если True, Optuna оценивает через K-Fold
            cv_folds   — количество фолдов при use_cv=True

        Возвращает:
            dict с метриками: {'Accuracy': 0.844, 'Precision': 0.84, 'Recall': 0.849}
            или {'R2': 0.72, 'MAE': 8.3, 'RMSE': 11.2} для регрессии

        """
        # Шаг 1: Удаление строк с NaN в y, sklearn обрабатывает Х (SimpleImputer), но не y
        n_before = len(df)
        df = df.dropna(subset=[target_col])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"[ml_engine] Удалено {n_dropped} строк с NaN в '{target_col}'.")

        # Шаг 2: Разделение на признаки в таргет
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.features  = list(X.columns)            # сохраняем список признаков (для инференса)
        self.task_type = self.detect_task_type(y)   # определение типа задачи (зависит от дальнейшего решения)

        # Шаг 3: Разделение на числовые и категориальные признаки
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Шаг 4: построение препроцессора
        numeric_transformer = Pipeline(steps=[          # два последовательный шага для числовых признаков
            ('imputer', SimpleImputer(strategy='median')),          # медиана при выбросах
            ('scaler', StandardScaler())                            # нормальизация
        ])
        categorical_transformer = Pipeline(steps=[      # два последовательный шага для категориальных признаков
            ('imputer', SimpleImputer(strategy='most_frequent')),   # мода при выбросах
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        preprocessor = ColumnTransformer(transformers=[ # горизонтальная склейка каждого трансформера
            ('num', numeric_transformer, num_cols),     # сначала числа потом категории
            ('cat', categorical_transformer, cat_cols)
        ])

        # Шаг 5: Выбор CV-сплиттера
        if self.task_type == 'classification':
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scoring  = 'accuracy'
        else:
            cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scoring  = 'r2'

        # Шаг 6: Обычный hold-out сплит нужен в любом случае — для confusion matrix и y_pred
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Шаг 7: Функция цели для Optuna
        def objective(trial):
            # Строим модель с гиперпараметрами, предложенными Optuna
            model = (self._build_classifier(trial)
                     if self.task_type == 'classification'
                     else self._build_regressor(trial))
            # Pipline, для обучения препроцессора заново на каждом trial
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

            if use_cv:
                scores = cross_val_score(pipe, X, y, cv=cv_splitter,     # честная оценка через K фолдов
                                         scoring=cv_scoring, n_jobs=-1)
                return scores.mean()                         # Optuna максимизирует данное значение
            else:
                # Быстый режим: через один split, обучение и проверка
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                return (accuracy_score(y_test, preds)
                        if self.task_type == 'classification'
                        else r2_score(y_test, preds))

        # Шаг 8: Подбор гиперпараметров или создание ансамбля
        if self.model_type == "Ансамбль (Ensemble)":             # Ансамбль не требует Optuna
            final_model = self._build_ensemble()
            self.best_params = {"Инфо": "VotingEnsemble (RF + GradientBoosting + Ridge/LogReg)"}
        else:
            study = optuna.create_study(direction='maximize')    # Создаем иследование Optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING) # Подавление логов Optuna (печатает строку)
            study.optimize(objective, n_trials=n_trials)           # Запуск оптимизатора
            self.best_params = study.best_params                   # словарь с лучшими найденными гиперпараметрами
            # Строим финальную модель с лучшими гиперпараметрами
            final_model = (self._build_classifier_from_params(self.best_params)
                           if self.task_type == 'classification'
                           else self._build_regressor_from_params(self.best_params))

        # Шаг 9: Финальное обучение на всём X_train и оценка
        self.pipeline = Pipeline(steps=[    # Финальный Pipline, который идет в model.pkl
            ('preprocessor', preprocessor),
            ('model', final_model)
        ])

        if use_cv:
            # Оцениваем финальную модель через CV — это итоговые метрики в UI
            cv_scores = cross_val_score(
                self.pipeline, X, y,
                cv=cv_splitter, scoring=cv_scoring, n_jobs=-1
            )
            self.cv_scores = cv_scores
            self.cv_mean   = float(cv_scores.mean())
            # std показывает стабильность (малый std = модель одинакова на всех фолдах)
            self.cv_std    = float(cv_scores.std())
            # Обучаем на всех данных для деплоя и SHAP
            self.pipeline.fit(X, y)

            # Для confusion matrix используем hold-out (CV не даёт отдельного y_pred)
            pipe_holdout = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', (self._build_classifier_from_params(self.best_params)
                           if self.model_type != "Ансамбль (Ensemble)"
                              and self.task_type == 'classification'
                           else (self._build_regressor_from_params(self.best_params)
                                 if self.model_type != "Ансамбль (Ensemble)"
                                 else self._build_ensemble())))
            ])
            pipe_holdout.fit(X_train, y_train)
            preds = pipe_holdout.predict(X_test)
        else:
            # Быстрый режим: обучение на X_train, оценка на X_test
            self.cv_scores = None
            self.cv_mean   = None
            self.cv_std    = None
            self.pipeline.fit(X_train, y_train)
            preds = self.pipeline.predict(X_test)

        # Сохраняем для confusion matrix и внешнего доступа (SHAP из тестера)
        self.y_test = y_test
        self.y_pred = preds

        # # Шаг 10: Вычисление метрик
        if self.task_type == 'classification':
            # Сортируем классы в предсказуемом порядке: [0, 1] или ['cat', 'dog']
            self.class_labels = sorted(y.unique().tolist())
            # Матрица N x N: строка = реальный класс, столбец = предсказанный
            self.conf_matrix  = confusion_matrix(y_test, preds, labels=self.class_labels)
            metrics = {
                "Accuracy":  round(accuracy_score(y_test, preds), 3),
                "Precision": round(precision_score(y_test, preds, average='macro', zero_division=0), 3),
                "Recall":    round(recall_score(y_test, preds, average='macro', zero_division=0), 3),
            }
            if use_cv:
                metrics["CV Accuracy (mean)"] = round(self.cv_mean, 3) # CV-метрики добавляются только при включённом CV
                metrics["CV Accuracy (±std)"] = round(self.cv_std, 3)  # std ~0.02 -> модель стабильна; std ~0.15 -> нестабильна (переобучение)
        else:
            self.class_labels = None
            self.conf_matrix  = None
            metrics = {
                "R²":   round(r2_score(y_test, preds), 3),                           # R2 коэффициент детерминации
                "MAE":  round(mean_absolute_error(y_test, preds), 3),                # MAE устойчива к выбросам
                "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3), # Штрафует крупные ошибки сильнее
            }
            if use_cv:
                metrics["CV R² (mean)"] = round(self.cv_mean, 3)
                metrics["CV R² (±std)"] = round(self.cv_std, 3)

        return metrics

    # ------------------------------------------------------------------
    # SHAP: вычисление значений для одной строки
    # ------------------------------------------------------------------
    def compute_shap_values(self, row_df: pd.DataFrame):
        """
        Вычисляет SHAP-значения для одной строки данных.

        SHAP (SHapley Additive exPlanations) — метод из теории кооперативных игр.
        Каждый признак — "игрок", предсказание — "выигрыш команды".
        Значение Шепли показывает: на сколько этот признак сдвинул предсказание
        от среднего по тренировочным данным (base_value).

        Математическая гарантия: sum(shap_values) + base_value = prediction модели.

        Возвращает:
            (shap_values_1d, base_value, feature_names)
            - shap_values_1d: numpy-массив длиной N_features
            - base_value: float — среднее предсказание по тренировочным данным
            - feature_names: список названий признаков в том же порядке
            None — при любой ошибке (импорт, вычисление)

        """
        try:
            import shap
        except ImportError:
            print("[SHAP] shap не установлен.")
            return None

        # Извлекаем препроцессор и модель из Pipline по именам шагов
        preprocessor = self.pipeline.named_steps['preprocessor']
        model        = self.pipeline.named_steps['model']

        # Востанавливаем имена признаков из ColumnTransformer
        num_cols = list(preprocessor.transformers_[0][2]) # колонки, переданные в числовой тр-формер
        cat_cols = list(preprocessor.transformers_[1][2]) # колонки, переданные в категориальный тр-р
        feature_names = num_cols + cat_cols

        row_aligned   = row_df.reindex(columns=self.features) # Выравниваем входную строку по обучающим призакам
        X_transformed = preprocessor.transform(row_aligned)   # трансформируем строку, отдельно от модели

        # Для VotingClassifier/VotingRegressor берём первый estimator (RF)
        actual_model = model
        if hasattr(model, 'voting') or (
            hasattr(model, 'estimators_') and isinstance(model.estimators_, list)
        ):
            actual_model = model.estimators_[0]         # первый estimator = RF

        try:
            if hasattr(actual_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(actual_model)     # Деревья (RF, GB) -> TreeExplainer
                shap_raw  = explainer.shap_values(X_transformed)
                base_val  = explainer.expected_value
            else:                                # Линейные модели (Ridge, LogisticRegression) -> LinearExplainer.
                explainer = shap.LinearExplainer(actual_model, X_transformed)
                shap_raw  = explainer.shap_values(X_transformed)
                base_val  = explainer.expected_value

            # Нормализация форматов возвращаемых значений
            sv_arr = np.array(shap_raw)                  # Разные версии SHAP возвращаем массивы в разных форматах
            ev_arr = np.atleast_1d(np.array(base_val).ravel())

            if isinstance(shap_raw, list):
                sv_arr    = np.array(shap_raw[-1]) # Устаревший API для multiclass: list[array_class0, array_class1, ...]
                ev_scalar = float(ev_arr[-1])
            elif sv_arr.ndim == 3:          # Новый API RandomForest: (1 sample, N features, N classes)
                sv_arr    = sv_arr[0, :, -1]
                ev_scalar = float(ev_arr[-1])
            elif sv_arr.ndim == 2:          # Стандартный формат для GB и регрессии: (1 sample, N features)
                sv_arr    = sv_arr[0]
                ev_scalar = float(ev_arr[-1] if len(ev_arr) > 1 else ev_arr[0])
            else:                           # Уже одномерный массив (1D)
                ev_scalar = float(ev_arr[0])

            return np.array(sv_arr), ev_scalar, feature_names

        except Exception as e:
            print(f"[SHAP] Ошибка: {e}")
            return None

    # ------------------------------------------------------------------
    # Вспомогательные методы: построение моделей
    # ------------------------------------------------------------------
    def _build_classifier(self, trial):
        """
        Строит классификатор с гиперпараметрами, предложенными Optuna.
        trial.suggest_int/float() не просто случайные числа — Optuna анализирует
        прошлые trial'ы и предлагает значения в перспективных областях пространства

        """
        if self.model_type == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                random_state=42)
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                random_state=42)
        elif self.model_type == "Logistic Regression":
            return LogisticRegression(
                C=trial.suggest_float('C', 0.01, 20.0),
                max_iter=1000, random_state=42)

    def _build_regressor(self, trial):
        """Аналог _build_classifier() для задачи регрессии."""
        if self.model_type == "Random Forest":
            return RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                random_state=42)
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                random_state=42)
        elif self.model_type == "Logistic Regression":
            return Ridge(alpha=trial.suggest_float('alpha', 0.01, 50.0))

    def _build_classifier_from_params(self, params):
        """
        Строит финальный классификатор с уже известными лучшими параметрами.
        Вызывается после завершения оптимизации Optuna.
        **params распаковывает словарь {'n_estimators': 120, 'max_depth': 7} в kwargs

        """
        if self.model_type == "Random Forest":
            return RandomForestClassifier(**params, random_state=42)
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingClassifier(**params, random_state=42)
        elif self.model_type == "Logistic Regression":
            return LogisticRegression(**params, max_iter=1000, random_state=42)

    def _build_regressor_from_params(self, params):
        """Аналог _build_classifier_from_params() для регрессии"""
        if self.model_type == "Random Forest":
            return RandomForestRegressor(**params, random_state=42)
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingRegressor(**params, random_state=42)
        elif self.model_type == "Logistic Regression":
            return Ridge(**params)

    def _build_ensemble(self):
        """
        Создаёт ансамбль из трёх разных алгоритмов.

        Идея: разные алгоритмы делают разные ошибки. RF устойчив к выбросам,
        GB хорошо улавливает нелинейные паттерны, LogReg/Ridge — линейные зависимости.
        В среднем их ошибки компенсируют друг друга.

        voting='soft' для классификации: финальный класс определяется по средней
        вероятности (P_RF + P_GB + P_LR) / 3, а не простым голосованием.
        Soft voting точнее когда модели уверены с разной степенью.

        """
        if self.task_type == 'classification':
            return VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                  max_depth=5, random_state=42)),
                ('lr', LogisticRegression(C=1.0, max_iter=500, random_state=42)),
            ], voting='soft')
        else:
            # VotingRegressor усредняет числовые предсказания трёх моделей
            return VotingRegressor(estimators=[
                ('rf',    RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)),
                ('gb',    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                    max_depth=5, random_state=42)),
                ('ridge', Ridge(alpha=1.0)),
            ])

    # ------------------------------------------------------------------
    # Человекочитаемое объяснение модели
    # ------------------------------------------------------------------
    def generate_human_explanation(self):
        """
        Формирует текстовое объяснение решения модели на русском языке.
        Называет топ-2 признака по важности и параметры найденные Optuna.

        Для деревьев важность = среднее снижение примеси (Gini impurity) по всем деревьям.
        Для линейных моделей важность = абсолютный вес коэффициента |w_i|.

        """
        model        = self.pipeline.named_steps['model']
        preprocessor = self.pipeline.named_steps['preprocessor']

        if self.model_type == "Ансамбль (Ensemble)":
            t = "классификации" if self.task_type == 'classification' else "регрессии"
            return (f"🌟 **Ансамбль** для задачи {t}: "
                    "консенсус RF + GradientBoosting + Ridge/LogReg.")

        # Восстанавливаем порядок признаков из ColumnTransformer
        num_cols = list(preprocessor.transformers_[0][2])
        cat_cols = list(preprocessor.transformers_[1][2])
        ordered_features = num_cols + cat_cols

        # Извлекаем важности — атрибут зависит от типа модели
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return "Интерпретация недоступна для данного типа модели."

        # Сортируем признаки по убыванию важности
        fi   = sorted(zip(ordered_features, importances), key=lambda x: x[1], reverse=True)
        top1 = fi[0]
        top2 = fi[1] if len(fi) > 1 else None

        tl = "классификации" if self.task_type == 'classification' else "регрессии"
        md = ("Ridge Regression (авто-замена)"
              if self.model_type == "Logistic Regression" and self.task_type == 'regression'
              else self.model_type)

        text = (f"Модель **{md}** (задача {tl}) "
                f"в первую очередь опирается на **{top1[0]}**.")
        if top2:
            text += f" Вторая по значимости — **{top2[0]}**."

        # Форматируем параметры Optuna для отображения
        ps = ", ".join(
            f"{k}={round(v, 4) if isinstance(v, float) else v}"
            for k, v in self.best_params.items()
        )
        text += f"\n\n🤖 **Optuna:** `{ps}`."
        return text

    # ──────────────────────────────────────────────────────────────────────────
    # СОХРАНЕНИЕ МОДЕЛИ
    # ──────────────────────────────────────────────────────────────────────────
    def save_model(self, path="model.pkl"):
        """
        Сохраняет всё необходимое для последующего инференса в один файл.

        Структура сохранённого словаря:
        - 'model'        -> обученный Pipeline (препроцессор + модель в одном объекте)
        - 'features'     -> список признаков в правильном порядке (для reindex при инференсе)
        - 'task_type'    -> 'classification' или 'regression' (для правильного отображения в API)
        - 'class_labels' -> отсортированный список классов (для confusion matrix и predict_proba)

        """
        joblib.dump({
            "model":        self.pipeline,
            "features":     self.features,
            "task_type":    self.task_type,
            "class_labels": self.class_labels,
        }, path)