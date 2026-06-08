FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Sistema: herramientas para compilar dependencias nativas (p. ej. llama-cpp-python)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       git \
       libssl-dev \
       libffi-dev \
       libopenblas-dev \
       libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copiar el código (no copiamos modelos grandes, se montan como volumen)
COPY . /app

# Puerto expuesto (opcional, el script actual es interactivo)
EXPOSE 8000

CMD ["python", "IA/preguntar_modelo.py"]
