# Dockerfile de routing_node
FROM python:3.9-buster

WORKDIR /app

# Installer les dépendances système et Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    cmake \
    libssl-dev \
    pkg-config \
    libclang-dev \
    clang \
    libz-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.56.0 \
    && export PATH="/root/.cargo/bin:$PATH" \
    && rustup component add rust-src \
    && rm -rf /var/lib/apt/lists/*

# Ajouter la variable d'environnement pour les téléchargements Rust
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

# Copier les fichiers nécessaires
COPY ./nodes/routing_node/requirements.txt /app/
COPY ./nodes/routing_node /app

# Ajouter le dossier optimization_scripts au PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/optimization_scripts"

# Installer les dépendances Python
RUN pip install --upgrade pip && pip install --only-binary :all: tokenizers && pip install --no-cache-dir -r /app/requirements.txt

# Correction du port exposé (décommenter si nécessaire)
# EXPOSE 8000

# Mise à jour de la commande de démarrage (décommenter si nécessaire)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
