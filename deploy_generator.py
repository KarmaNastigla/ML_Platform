import os

# Только зависимости, нужные для продакшен-инференса.
# Streamlit, Plotly, Optuna, SHAP и прочие dev-инструменты сюда не попадают.
# Это уменьшает Docker-образ с ~2 ГБ до ~300 МБ.
PRODUCTION_REQUIREMENTS = """fastapi==0.109.0
uvicorn==0.27.0
scikit-learn==1.4.0
pandas==2.2.0
joblib==1.3.2
numpy>=1.26.0
"""

def generate_deployment_files():
    os.makedirs("deploy", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. FastAPI-приложение
    # ------------------------------------------------------------------
    api_code = '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Auto-Generated ML API",
    description="Модель, задеплоенная через Universal ML Platform",
    version="1.0.0",
)

data = joblib.load("model.pkl")
model = data["model"]
features = data["features"]
task_type = data.get("task_type", "classification")


class InferenceData(BaseModel):
    data: dict


@app.get("/health")
def health():
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "task_type": task_type, "features": features}


@app.post("/predict")
def predict(input_data: InferenceData):
    """
    Сделать предсказание.

    Передай JSON вида: {"data": {"feature1": value1, "feature2": value2, ...}}

    Отсутствующие признаки будут заполнены NaN — Pipeline обработает их автоматически.
    Лишние ключи игнорируются.
    """
    # Проверяем, что хотя бы один из ожидаемых признаков присутствует
    provided_keys = set(input_data.data.keys())
    known_keys = set(features)
    if not provided_keys.intersection(known_keys):
        raise HTTPException(
            status_code=422,
            detail=f"Ни один из переданных признаков не совпадает с ожидаемыми: {features}"
        )

    df = pd.DataFrame([input_data.data])
    # reindex выравнивает колонки по тренировочным данным:
    # отсутствующие → NaN (обрабатывается SimpleImputer внутри Pipeline)
    # лишние → отбрасываются
    df = df.reindex(columns=features)

    prediction = model.predict(df)
    result = prediction[0]

    # Приводим numpy-тип к стандартному Python для корректной JSON-сериализации
    if hasattr(result, 'item'):
        result = result.item()

    return {
        "prediction": result,
        "task_type": task_type,
    }
'''
    with open("deploy/api.py", "w", encoding="utf-8") as f:
        f.write(api_code.strip())

    # ------------------------------------------------------------------
    # 2. Dockerfile — минимальный образ на python:3.10-slim
    # ------------------------------------------------------------------
    docker_code = '''FROM python:3.10-slim

WORKDIR /app

# Копируем только то, что нужно для инференса
COPY model.pkl api.py requirements.txt /app/

# Устанавливаем только продакшен-зависимости (без Streamlit/Plotly/Optuna)
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    with open("deploy/Dockerfile", "w", encoding="utf-8") as f:
        f.write(docker_code.strip())

    # ------------------------------------------------------------------
    # 3. Lean requirements.txt — ТОЛЬКО продакшен-зависимости
    #    (не копируем dev requirements.txt из корня проекта!)
    # ------------------------------------------------------------------
    with open("deploy/requirements.txt", "w", encoding="utf-8") as f:
        f.write(PRODUCTION_REQUIREMENTS.strip())

    # ------------------------------------------------------------------
    # 4. docker-compose.yml — для удобного локального запуска
    # ------------------------------------------------------------------
    compose_code = '''version: "3.9"
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model.pkl:/app/model.pkl
    restart: unless-stopped
'''
    with open("deploy/docker-compose.yml", "w", encoding="utf-8") as f:
        f.write(compose_code.strip())