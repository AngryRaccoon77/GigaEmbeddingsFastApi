from GigaEmbeddingsService.app.models.embedding_model import embedding_model_instance
from GigaEmbeddingsService.app.core.config import settings

class EmbeddingService:
    """
    Сервис для генерации эмбеддингов. Изолирует бизнес-логику
    от деталей реализации модели и API.
    """
    def __init__(self, model):
        self.model = model

    def create_passage_embedding(self, text: str) -> list[float]:
        """
        Создает эмбеддинг для пассажа.
        """
        return self.model.get_embedding(text, instruction=settings.PASSAGE_INSTRUCTION)

    def create_query_embedding(self, text: str) -> list[float]:
        """
        Создает эмбеддинг для поискового запроса.
        """
        return self.model.get_embedding(text, instruction=settings.QUERY_INSTRUCTION)

embedding_service_instance = EmbeddingService(model=embedding_model_instance)