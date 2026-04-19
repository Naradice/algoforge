FROM python:3.12-slim

WORKDIR /app

# Install system deps for asyncpg and playwright
RUN apt-get update && apt-get install -y gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

# Copy editable-install sources first (layer caching)
COPY finance_client /deps/finance_client

COPY algoforge/backend/requirements.txt .
# --no-deps skips Windows-only finance_client deps (e.g. MetaTrader5) that fail on Linux.
# requirements.txt must include every finance_client dep needed at runtime.
RUN pip install --no-cache-dir --no-deps /deps/finance_client
RUN pip install --no-cache-dir -r requirements.txt

COPY algoforge/backend/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
