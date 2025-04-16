# Базовый образ с Python
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файлов приложения
COPY . /app

# Установка Python-зависимостей
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    torch \
    transformers \
    accelerate

# Опциональная установка зависимостей для CUDA (если используется NVIDIA GPU)
ARG USE_CUDA=false
RUN if [ "$USE_CUDA" = "true" ]; then \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118; \
    fi

# Опциональная установка зависимостей для Metal (если используется macOS с Apple Silicon)
ARG USE_METAL=false
RUN if [ "$USE_METAL" = "true" ]; then \
    pip install --no-cache-dir torch-metal; \
    fi

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/app/model_cache

# Создание директории для кэша модели
RUN mkdir -p /app/model_cache

# Открытие порта
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]