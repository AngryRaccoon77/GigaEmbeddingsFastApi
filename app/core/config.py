from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Класс для хранения конфигурации приложения.
    """
    EMBEDDING_MODEL_NAME: str = "ai-sage/Giga-Embeddings-instruct"
    MODEL_CACHE_DIR: str = "cache_dir"
    PASSAGE_INSTRUCTION: str = ""  # Инструкция для пассажей
    # Инструкция для запросов
    QUERY_INSTRUCTION: str = "Создайте эмбеддинги для следующего запроса, чтобы найти релевантные тексты:\nзапрос: "

settings = Settings()