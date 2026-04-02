FROM python:3.12-slim

# Evita arquivos .pyc e permite logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia apenas o servidor SSE (não copia .env nem service_account.json)
# O GKE gerencia credenciais via Workload Identity (KSA -> GSA)
COPY veo_mcp_sse.py .

EXPOSE 8080

# Inicia o servidor SSE
CMD ["python", "veo_mcp_sse.py"]
