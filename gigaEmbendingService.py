from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel
import torch.nn.functional as F

# Константы
EMBEDDING_MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"
MODEL_CACHE_DIR = "../model_cache"
PASSAGE_INSTRUCTION = "Создайте эмбеддинги для следующего текста, чтобы использовать их в системе семантического поиска."  # Инструкция для пассажей (пустая строка)
QUERY_INSTRUCTION = "Создайте эмбеддинги для следующего запроса, чтобы найти релевантные тексты:\nзапрос: "  # Инструкция для запросов

# Загрузка модели при запуске сервиса
try:
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель: {e}")

# Определение моделей Pydantic для входных и выходных данных
class TextInput(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]

# Создание FastAPI приложения
app = FastAPI()

# Эндпоинт для генерации эмбеддингов пассажей
@app.post("/embed_passage", response_model=EmbeddingResponse)
def embed_passage(input: TextInput):
    try:
        # Генерация эмбеддинга для текста пассажа
        embedding = model.encode([input.text], instruction=PASSAGE_INSTRUCTION)
        # Нормализация эмбеддинга и преобразование в список
        embedding = F.normalize(embedding, p=2, dim=1).tolist()[0]
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации эмбеддинга: {e}")

# Эндпоинт для генерации эмбеддингов запросов
@app.post("/embed_query", response_model=EmbeddingResponse)
def embed_query(input: TextInput):
    try:
        # Генерация эмбеддинга для текста запроса
        embedding = model.encode([input.text], instruction=QUERY_INSTRUCTION)
        # Нормализация эмбеддинга и преобразование в список
        embedding = F.normalize(embedding, p=2, dim=1).tolist()[0]
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации эмбеддинга: {e}")
