# Usa Python 3.10 slim. Esta imagem suporta nativamente x86_64 (Linux/Windows) e arm64 (Mac M1/M2)
FROM python:3.10-slim

# Variáveis de ambiente para o Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências de sistema necessárias (sndfile é necessário para o torchaudio e librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho dentro do container
WORKDIR /workspace/IARA

# Copia e instala as dependências
COPY requirements.txt .

# O pip install torch vai baixar a versão com CUDA no Linux x86 (para Nvidia) 
# e a versão MPS/CPU nativa no Mac M1 automaticamente graças ao suporte multi-arch do PyTorch.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# O container vai iniciar em modo ocioso. A instalação do módulo "iara" 
# será feita dinamicamente via docker-compose para refletir mudanças locais.
CMD ["tail", "-f", "/dev/null"]
