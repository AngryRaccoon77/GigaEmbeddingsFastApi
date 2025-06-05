from pydantic import BaseModel

class TextInput(BaseModel):
    """Схема для входного текста."""
    text: str

class EmbeddingResponse(BaseModel):
    """Схема для ответа с эмбеддингом."""
    embedding: list[float]