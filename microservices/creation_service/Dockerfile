# Dockerfile de creation_service
FROM python:3.11

# Set environment variables
ENV HUGGINGFACE_TOKEN=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

# Rediriger le cache vers le disque dur
ENV HF_HOME=/mnt/volume_externe/my_project/ionosphere/hf_cache

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    cmake \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Installer Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN /bin/bash -c "source $HOME/.cargo/env && rustup component add rls rust-analysis rust-src"

# Copier les fichiers nécessaires du microservice
COPY ./microservices/creation_service/requirements.txt /app/
COPY ./microservices/creation_service /app

# Ajouter le dossier optimization_scripts au PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/optimization_scripts"

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#EXPOSE 8000
