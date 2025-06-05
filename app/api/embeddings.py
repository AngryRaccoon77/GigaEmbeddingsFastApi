# api/v1/embeddings.py
from fastapi import APIRouter, HTTPException, Depends
from GigaEmbeddingsService.app.schemas.embedding_schemas import TextInput, EmbeddingResponse
from GigaEmbeddingsService.app.services.embedding_service import EmbeddingService, embedding_service_instance

router = APIRouter()

def get_embedding_service() -> EmbeddingService:
    return embedding_service_instance

@router.post("/embed_passage", response_model=EmbeddingResponse)
def embed_passage(
    input_data: TextInput,
    service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Эндпоинт для генерации эмбеддинга для пассажа текста.
    """
    try:
        embedding = service.create_passage_embedding(input_data.text)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации эмбеддинга: {e}")

@router.post("/embed_query", response_model=EmbeddingResponse)
def embed_query(
    input_data: TextInput,
    service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Эндпоинт для генерации эмбеддинга для поискового запроса.
    """
    try:
        embedding = service.create_query_embedding(input_data.text)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации эмбеддинга: {e}")