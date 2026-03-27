FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY ../../finance_client /deps/finance_client
COPY requirements.txt .
RUN pip install --no-cache-dir /deps/finance_client
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "worker.py"]
