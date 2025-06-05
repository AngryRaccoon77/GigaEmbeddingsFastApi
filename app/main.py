from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.embeddings import router as embeddings_router
from models.embedding_model import embedding_model_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код, который выполнится при старте приложения
    print("Загрузка ML-модели...")
    embedding_model_instance.load()
    yield
    # Код, который выполнится при остановке приложения (если нужно)
    print("Приложение останавливается.")


app = FastAPI(
    title="Embedding Generation API",
    description="API для генерации эмбеддингов с использованием Giga-Embeddings",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(embeddings_router, prefix="/api/v1", tags=["Embeddings"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Embedding API. Visit /docs for documentation."}

if __name__ == "__main__":
    import uvicorn
    # Обратите внимание, что мы запускаем 'main:app'
    uvicorn.run(app, host="0.0.0.0", port=8085)