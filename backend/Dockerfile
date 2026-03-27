FROM python:3.12-slim

WORKDIR /app

# Install system deps for asyncpg and playwright
RUN apt-get update && apt-get install -y gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

# Copy editable-install sources first (layer caching)
COPY ../../finance_client /deps/finance_client
COPY ../../trade_strategy /deps/trade_strategy

COPY requirements.txt .
RUN pip install --no-cache-dir /deps/finance_client /deps/trade_strategy
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
