from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, BitsAndBytesConfig, AutoTokenizer
import torch.nn.functional as F

# Константы
EMBEDDING_MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"
MODEL_CACHE_DIR = "cache_dir"
PASSAGE_INSTRUCTION = ""  # Инструкция для пассажей (пустая строка)
QUERY_INSTRUCTION = "Создайте эмбеддинги для следующего запроса, чтобы найти релевантные тексты:\nзапрос: "  # Инструкция для запросов

# Загрузка модели при запуске сервиса
try:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE_DIR,
        device_map="auto",
        quantization_config=quantization_config

    )
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
    )
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель: {e}")

# Определение моделей Pydantic для входных и выходных данных

class TextType(str, Enum):
    passage = "passage"
    query = "query"
class TextInput(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]

class TokenCountInput(BaseModel):
    text: str
    text_type: TextType = TextType.passage

class TokenCountResponse(BaseModel):
    token_count: int
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


@app.post("/count_tokens", response_model=TokenCountResponse)
def count_tokens(input: TokenCountInput):
    try:
        # Определение инструкции в зависимости от типа текста
        instruction = PASSAGE_INSTRUCTION if input.text_type == TextType.passage else QUERY_INSTRUCTION
        # Составление полного текста (инструкция + текст)
        full_text = instruction + input.text
        # Токенизация текста и подсчет токенов
        token_ids = tokenizer.encode(full_text)
        token_count = len(token_ids)
        return {"token_count": token_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при подсчете токенов: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)