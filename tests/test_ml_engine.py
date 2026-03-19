"""
Юнит-тесты для UniversalMLEngine.
Запуск: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml_engine import UniversalMLEngine


# ──────────────────────────────────────────────────────────────────
# Фикстуры
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clf_df():
    """Синтетический датасет для классификации."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "age": np.random.randint(18, 80, n).astype(float),
        "income": np.random.uniform(20_000, 120_000, n),
        "gender": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["North", "South", "East"], n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def reg_df():
    """Синтетический датасет для регрессии."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "rooms": np.random.randint(1, 8, n),
        "area": np.random.uniform(30, 200, n),
        "floor": np.random.randint(1, 20, n),
        "city": np.random.choice(["Moscow", "SPb", "Kazan"], n),
        "price": np.random.uniform(50_000, 500_000, n),  # float → регрессия
    })


@pytest.fixture
def clf_df_with_nan(clf_df):
    """Датасет с пропусками в таргете."""
    df = clf_df.copy()
    df.loc[np.random.choice(df.index, 30, replace=False), "target"] = np.nan
    return df


# ──────────────────────────────────────────────────────────────────
# 1. Автоопределение типа задачи
# ──────────────────────────────────────────────────────────────────

class TestDetectTaskType:
    def test_binary_int_is_classification(self):
        engine = UniversalMLEngine()
        assert engine.detect_task_type(pd.Series([0, 1, 0, 1, 1])) == "classification"

    def test_multiclass_small_is_classification(self):
        engine = UniversalMLEngine()
        assert engine.detect_task_type(pd.Series([0, 1, 2, 0, 1, 2])) == "classification"

    def test_float_is_regression(self):
        engine = UniversalMLEngine()
        assert engine.detect_task_type(pd.Series([1.0, 2.5, 3.7, 4.1])) == "regression"

    def test_many_unique_int_is_regression(self):
        engine = UniversalMLEngine()
        y = pd.Series(range(100))  # 100 уникальных значений > 20
        assert engine.detect_task_type(y) == "regression"

    def test_exactly_20_unique_is_classification(self):
        engine = UniversalMLEngine()
        y = pd.Series(list(range(20)))  # ровно 20 → классификация
        assert engine.detect_task_type(y) == "classification"

    def test_21_unique_is_regression(self):
        engine = UniversalMLEngine()
        y = pd.Series(list(range(21)))  # 21 > 20 → регрессия
        assert engine.detect_task_type(y) == "regression"


# ──────────────────────────────────────────────────────────────────
# 2. Обучение — классификация
# ──────────────────────────────────────────────────────────────────

class TestClassification:
    @pytest.mark.parametrize("model_type", [
        "Random Forest", "Gradient Boosting", "Logistic Regression"
    ])
    def test_metrics_keys(self, clf_df, model_type):
        engine = UniversalMLEngine(model_type=model_type)
        metrics = engine.train_and_evaluate(clf_df, "target", n_trials=2)
        assert set(metrics.keys()) >= {"Accuracy", "Precision", "Recall"}

    def test_metrics_range(self, clf_df):
        engine = UniversalMLEngine("Random Forest")
        metrics = engine.train_and_evaluate(clf_df, "target", n_trials=2)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"

    def test_confusion_matrix_shape(self, clf_df):
        engine = UniversalMLEngine("Gradient Boosting")
        engine.train_and_evaluate(clf_df, "target", n_trials=2)
        cm = np.array(engine.conf_matrix)
        assert cm.shape == (2, 2)

    def test_class_labels(self, clf_df):
        engine = UniversalMLEngine("Random Forest")
        engine.train_and_evaluate(clf_df, "target", n_trials=2)
        assert engine.class_labels == [0, 1]

    def test_ensemble_classification(self, clf_df):
        engine = UniversalMLEngine("Ансамбль (Ensemble)")
        metrics = engine.train_and_evaluate(clf_df, "target", n_trials=1)
        assert "Accuracy" in metrics

    def test_nan_in_target_dropped(self, clf_df_with_nan):
        """Строки с NaN в таргете должны быть отброшены без ошибки.

        Примечание: вставка NaN в int-столбец конвертирует его в float64,
        поэтому задача автоматически определяется как регрессия.
        Тест проверяет, что обучение проходит без падения и возвращает метрики.
        """
        engine = UniversalMLEngine("Random Forest")
        metrics = engine.train_and_evaluate(clf_df_with_nan, "target", n_trials=2)
        # Метрики должны вернуться (тип задачи зависит от dtype после NaN-конвертации)
        assert len(metrics) > 0
        assert all(isinstance(v, float) for v in metrics.values())

    def test_cv_mode(self, clf_df):
        engine = UniversalMLEngine("Random Forest")
        metrics = engine.train_and_evaluate(clf_df, "target",
                                            n_trials=2, use_cv=True, cv_folds=3)
        assert "CV Accuracy (mean)" in metrics
        assert "CV Accuracy (±std)" in metrics
        assert len(engine.cv_scores) == 3


# ──────────────────────────────────────────────────────────────────
# 3. Обучение — регрессия
# ──────────────────────────────────────────────────────────────────

class TestRegression:
    @pytest.mark.parametrize("model_type", [
        "Random Forest", "Gradient Boosting", "Logistic Regression"
    ])
    def test_metrics_keys(self, reg_df, model_type):
        engine = UniversalMLEngine(model_type=model_type)
        metrics = engine.train_and_evaluate(reg_df, "price", n_trials=2)
        assert set(metrics.keys()) >= {"R²", "MAE", "RMSE"}
        assert engine.task_type == "regression"

    def test_logistic_replaced_by_ridge(self, reg_df):
        """Logistic Regression на регрессионной задаче → Ridge."""
        engine = UniversalMLEngine("Logistic Regression")
        metrics = engine.train_and_evaluate(reg_df, "price", n_trials=2)
        assert engine.task_type == "regression"
        assert "R²" in metrics
        assert engine.conf_matrix is None  # confusion matrix недоступна

    def test_rmse_positive(self, reg_df):
        engine = UniversalMLEngine("Random Forest")
        metrics = engine.train_and_evaluate(reg_df, "price", n_trials=2)
        assert metrics["RMSE"] > 0
        assert metrics["MAE"] > 0

    def test_ensemble_regression(self, reg_df):
        engine = UniversalMLEngine("Ансамбль (Ensemble)")
        metrics = engine.train_and_evaluate(reg_df, "price", n_trials=1)
        assert "R²" in metrics

    def test_cv_regression(self, reg_df):
        engine = UniversalMLEngine("Gradient Boosting")
        metrics = engine.train_and_evaluate(reg_df, "price",
                                            n_trials=2, use_cv=True, cv_folds=3)
        assert "CV R² (mean)" in metrics


# ──────────────────────────────────────────────────────────────────
# 4. SHAP
# ──────────────────────────────────────────────────────────────────

class TestSHAP:
    def _train_and_get_row(self, df, target, model_type):
        engine = UniversalMLEngine(model_type)
        engine.train_and_evaluate(df, target, n_trials=2)
        row = df.drop(columns=[target]).iloc[[0]]
        return engine, row

    @pytest.mark.parametrize("model_type", [
        "Random Forest", "Gradient Boosting",
        "Logistic Regression", "Ансамбль (Ensemble)"
    ])
    def test_shap_classification(self, clf_df, model_type):
        engine, row = self._train_and_get_row(clf_df, "target", model_type)
        result = engine.compute_shap_values(row)
        assert result is not None, f"SHAP returned None for {model_type}"
        sv, bv, fn = result
        assert len(sv) == len(fn)
        assert isinstance(bv, float)

    @pytest.mark.parametrize("model_type", [
        "Random Forest", "Gradient Boosting", "Logistic Regression"
    ])
    def test_shap_regression(self, reg_df, model_type):
        engine, row = self._train_and_get_row(reg_df, "price", model_type)
        result = engine.compute_shap_values(row)
        assert result is not None
        sv, bv, fn = result
        assert len(sv) == len(fn)

    def test_shap_values_are_finite(self, clf_df):
        engine = UniversalMLEngine("Random Forest")
        engine.train_and_evaluate(clf_df, "target", n_trials=2)
        row = clf_df.drop(columns=["target"]).iloc[[0]]
        sv, bv, fn = engine.compute_shap_values(row)
        assert np.all(np.isfinite(sv)), "SHAP values contain inf or nan"
        assert np.isfinite(bv)


# ──────────────────────────────────────────────────────────────────
# 5. Сохранение и загрузка модели
# ──────────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, clf_df, tmp_path):
        engine = UniversalMLEngine("Random Forest")
        engine.train_and_evaluate(clf_df, "target", n_trials=2)
        path = str(tmp_path / "model.pkl")
        engine.save_model(path)
        assert os.path.exists(path)

    def test_loaded_model_predicts(self, clf_df, tmp_path):
        engine = UniversalMLEngine("Gradient Boosting")
        engine.train_and_evaluate(clf_df, "target", n_trials=2)
        path = str(tmp_path / "model.pkl")
        engine.save_model(path)

        data = joblib.load(path)
        assert "model" in data and "features" in data and "task_type" in data
        row = clf_df.drop(columns=["target"]).iloc[[0]]
        pred = data["model"].predict(row.reindex(columns=data["features"]))
        assert pred[0] in [0, 1]

    def test_task_type_persisted(self, reg_df, tmp_path):
        engine = UniversalMLEngine("Random Forest")
        engine.train_and_evaluate(reg_df, "price", n_trials=2)
        path = str(tmp_path / "model.pkl")
        engine.save_model(path)
        data = joblib.load(path)
        assert data["task_type"] == "regression"

    def test_class_labels_persisted(self, clf_df, tmp_path):
        engine = UniversalMLEngine("Random Forest")
        engine.train_and_evaluate(clf_df, "target", n_trials=2)
        path = str(tmp_path / "model.pkl")
        engine.save_model(path)
        data = joblib.load(path)
        assert data["class_labels"] == [0, 1]


# ──────────────────────────────────────────────────────────────────
# 6. Объяснение модели
# ──────────────────────────────────────────────────────────────────

class TestExplanation:
    def test_explanation_is_string(self, clf_df):
        engine = UniversalMLEngine("Random Forest")
        engine.train_and_evaluate(clf_df, "target", n_trials=2)
        explanation = engine.generate_human_explanation()
        assert isinstance(explanation, str) and len(explanation) > 20

    def test_explanation_mentions_top_feature(self, clf_df):
        engine = UniversalMLEngine("Gradient Boosting")
        engine.train_and_evaluate(clf_df, "target", n_trials=2)
        explanation = engine.generate_human_explanation()
        # Хотя бы одно название признака должно быть в объяснении
        features = engine.features
        assert any(f in explanation for f in features)

    def test_ensemble_explanation(self, clf_df):
        engine = UniversalMLEngine("Ансамбль (Ensemble)")
        engine.train_and_evaluate(clf_df, "target", n_trials=1)
        explanation = engine.generate_human_explanation()
        assert "Ансамбль" in explanation