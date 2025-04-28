# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- System Setup ----------
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- Environment Variables ----------
ENV PYTHONUNBUFFERED=1

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Install Python Dependencies ----------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt      

# ---------- Copy Application Code ----------
COPY . /app

# ---------- Expose Port ----------
EXPOSE 8001

# ---------- Run FastAPI ----------
CMD ["uvicorn", "run_pipeline:app", "--host", "0.0.0.0", "--port", "8001"]