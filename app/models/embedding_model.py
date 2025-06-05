import torch.nn.functional as F
from transformers import AutoModel, BitsAndBytesConfig, AutoTokenizer
from GigaEmbeddingsService.app.core.config import settings


class EmbeddingModel:
    """
    Класс-обертка для управления жизненным циклом и использованием
    модели для генерации эмбеддингов.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        """
        Загружает модель и токенизатор в память.
        Вызывается один раз при старте приложения.
        """
        if self.model is None:
            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModel.from_pretrained(
                    settings.EMBEDDING_MODEL_NAME,
                    trust_remote_code=True,
                    cache_dir=settings.MODEL_CACHE_DIR,
                    device_map="auto",
                    quantization_config=quantization_config
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    settings.EMBEDDING_MODEL_NAME,
                    cache_dir=settings.MODEL_CACHE_DIR,
                )
                print("Модель и токенизатор успешно загружены.")
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить модель: {e}")

    def get_embedding(self, text: str, instruction: str) -> list[float]:
        """
        Генерирует эмбеддинг для заданного текста и инструкции.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Модель не загружена. Вызовите метод load() перед использованием.")

        embedding = self.model.encode([text], instruction=instruction)

        normalized_embedding = F.normalize(embedding, p=2, dim=1).tolist()[0]

        return normalized_embedding


embedding_model_instance = EmbeddingModel()