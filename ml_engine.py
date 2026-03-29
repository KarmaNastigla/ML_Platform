import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    learning_curve  # ← добавлен импорт (был NameError при построении кривой обучения)
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import optuna
import joblib
import warnings

warnings.filterwarnings('ignore')


class UniversalMLEngine:
    """
    Ядро платформы. Инкапсулирует полный жизненный цикл одного ML-эксперимента:
    препроцессинг -> оптимизация гиперпараметров -> обучение -> оценка -> SHAP -> сохранение.
    """

    def __init__(self, model_type="Random Forest"):
        self.pipeline = None
        self.features = []
        self.best_params = {}
        self.model_type = model_type
        self.task_type = None
        self.y_test = None
        self.y_pred = None
        self.class_labels = None
        self.conf_matrix = None
        self.cv_scores = None
        self.cv_mean = None
        self.cv_std = None
        self.learning_curve_data = None

    def detect_task_type(self, y: pd.Series) -> str:
        if y.dtype in ['float64', 'float32']:
            return 'regression'
        if y.nunique() > 20:
            return 'regression'
        return 'classification'

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_trials: int = 10,
        use_cv: bool = False,
        cv_folds: int = 5,
    ):
        n_before = len(df)
        df = df.dropna(subset=[target_col])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"[ml_engine] Удалено {n_dropped} строк с NaN в '{target_col}'.")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.features = list(X.columns)
        self.task_type = self.detect_task_type(y)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

        if self.task_type == 'classification':
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scoring = 'accuracy'
        else:
            cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scoring = 'r2'

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        def objective(trial):
            model = (self._build_classifier(trial)
                     if self.task_type == 'classification'
                     else self._build_regressor(trial))
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            if use_cv:
                scores = cross_val_score(pipe, X, y, cv=cv_splitter,
                                         scoring=cv_scoring, n_jobs=1)
                return scores.mean()
            else:
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                return (accuracy_score(y_test, preds)
                        if self.task_type == 'classification'
                        else r2_score(y_test, preds))

        if self.model_type == "Ансамбль (Ensemble)":
            final_model = self._build_ensemble()
            self.best_params = {"Инфо": "VotingEnsemble (RF + GradientBoosting + Ridge/LogReg)"}
        else:
            study = optuna.create_study(direction='maximize')
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=n_trials)
            self.best_params = study.best_params
            final_model = (self._build_classifier_from_params(self.best_params)
                           if self.task_type == 'classification'
                           else self._build_regressor_from_params(self.best_params))

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', final_model)
        ])

        if use_cv:
            cv_scores = cross_val_score(
                self.pipeline, X, y,
                cv=cv_splitter, scoring=cv_scoring, n_jobs=1
            )
            self.cv_scores = cv_scores
            self.cv_mean = float(cv_scores.mean())
            self.cv_std = float(cv_scores.std())
            self.pipeline.fit(X, y)

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
            self.cv_scores = None
            self.cv_mean = None
            self.cv_std = None
            self.pipeline.fit(X_train, y_train)
            preds = self.pipeline.predict(X_test)

        self.y_test = y_test
        self.y_pred = preds

        if self.task_type == 'classification':
            self.class_labels = sorted(y.unique().tolist())
            self.conf_matrix = confusion_matrix(y_test, preds, labels=self.class_labels)
            metrics = {
                "Accuracy":  round(accuracy_score(y_test, preds), 3),
                "Precision": round(precision_score(y_test, preds, average='macro', zero_division=0), 3),
                "Recall":    round(recall_score(y_test, preds, average='macro', zero_division=0), 3),
            }
            if use_cv:
                metrics["CV Accuracy (mean)"] = round(self.cv_mean, 3)
                metrics["CV Accuracy (±std)"] = round(self.cv_std, 3)
        else:
            self.class_labels = None
            self.conf_matrix = None
            metrics = {
                "R²":   round(r2_score(y_test, preds), 3),
                "MAE":  round(mean_absolute_error(y_test, preds), 3),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
            }
            if use_cv:
                metrics["CV R² (mean)"] = round(self.cv_mean, 3)
                metrics["CV R² (±std)"] = round(self.cv_std, 3)

        # ── Learning Curve ──────────────────────────────────────────────────
        try:
            lc_cv = (StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                     if self.task_type == 'classification'
                     else KFold(n_splits=3, shuffle=True, random_state=42))
            lc_scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
            lc_sizes, lc_train, lc_val = learning_curve(
                self.pipeline, X, y,
                train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
                cv=lc_cv,
                scoring=lc_scoring,
                n_jobs=1,
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
            self.learning_curve_data = None

        return metrics

    def compute_shap_values(self, row_df: pd.DataFrame):
        try:
            import shap
        except ImportError:
            return None

        preprocessor = self.pipeline.named_steps['preprocessor']
        model = self.pipeline.named_steps['model']

        num_cols = list(preprocessor.transformers_[0][2])
        cat_cols = list(preprocessor.transformers_[1][2])
        feature_names = num_cols + cat_cols

        row_aligned = row_df.reindex(columns=self.features)
        X_transformed = preprocessor.transform(row_aligned)

        actual_model = model
        if hasattr(model, 'voting') or (
            hasattr(model, 'estimators_') and isinstance(model.estimators_, list)
        ):
            actual_model = model.estimators_[0]

        try:
            if hasattr(actual_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(actual_model)
                shap_raw = explainer.shap_values(X_transformed)
                base_val = explainer.expected_value
            else:
                explainer = shap.LinearExplainer(actual_model, X_transformed)
                shap_raw = explainer.shap_values(X_transformed)
                base_val = explainer.expected_value

            sv_arr = np.array(shap_raw)
            ev_arr = np.atleast_1d(np.array(base_val).ravel())

            if isinstance(shap_raw, list):
                sv_arr = np.array(shap_raw[-1])
                ev_scalar = float(ev_arr[-1])
            elif sv_arr.ndim == 3:
                sv_arr = sv_arr[0, :, -1]
                ev_scalar = float(ev_arr[-1])
            elif sv_arr.ndim == 2:
                sv_arr = sv_arr[0]
                ev_scalar = float(ev_arr[-1] if len(ev_arr) > 1 else ev_arr[0])
            else:
                ev_scalar = float(ev_arr[0])

            return np.array(sv_arr), ev_scalar, feature_names

        except Exception as e:
            print(f"[SHAP] Ошибка: {e}")
            return None

    def _build_classifier(self, trial):
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
                max_iter=1000,
                random_state=42)

    def _build_regressor(self, trial):
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
        if self.model_type == "Random Forest":
            return RandomForestClassifier(**params, random_state=42)
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingClassifier(**params, random_state=42)
        elif self.model_type == "Logistic Regression":
            return LogisticRegression(**params, max_iter=1000, random_state=42)

    def _build_regressor_from_params(self, params):
        if self.model_type == "Random Forest":
            return RandomForestRegressor(**params, random_state=42)
        elif self.model_type == "Gradient Boosting":
            return GradientBoostingRegressor(**params, random_state=42)
        elif self.model_type == "Logistic Regression":
            return Ridge(**params)

    def _build_ensemble(self):
        if self.task_type == 'classification':
            return VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                  max_depth=5, random_state=42)),
                ('lr', LogisticRegression(C=1.0, max_iter=500, random_state=42)),
            ], voting='soft')
        else:
            return VotingRegressor(estimators=[
                ('rf',    RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)),
                ('gb',    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                    max_depth=5, random_state=42)),
                ('ridge', Ridge(alpha=1.0)),
            ])

    def generate_human_explanation(self):
        model = self.pipeline.named_steps['model']
        preprocessor = self.pipeline.named_steps['preprocessor']

        if self.model_type == "Ансамбль (Ensemble)":
            t = "классификации" if self.task_type == 'classification' else "регрессии"
            return (f"🌟 **Ансамбль** для задачи {t}: "
                    "консенсус RF + GradientBoosting + Ridge/LogReg.")

        num_cols = list(preprocessor.transformers_[0][2])
        cat_cols = list(preprocessor.transformers_[1][2])
        ordered_features = num_cols + cat_cols

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return "Интерпретация недоступна для данного типа модели."

        fi = sorted(zip(ordered_features, importances), key=lambda x: x[1], reverse=True)
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

        ps = ", ".join(
            f"{k}={round(v, 4) if isinstance(v, float) else v}"
            for k, v in self.best_params.items()
        )
        text += f"\n\n🤖 **Optuna:** `{ps}`."
        return text

    def save_model(self, path="model.pkl"):
        joblib.dump({
            "model":        self.pipeline,
            "features":     self.features,
            "task_type":    self.task_type,
            "class_labels": self.class_labels,
        }, path)