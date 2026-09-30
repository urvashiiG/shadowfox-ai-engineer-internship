FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu" \
    && printf 'torch==2.14.0+cpu\n' > /tmp/torch-constraints.txt \
    && python -m pip install --no-cache-dir --constraint /tmp/torch-constraints.txt -r requirements.txt \
    && rm /tmp/torch-constraints.txt
COPY app ./app
COPY frontend ./frontend
RUN mkdir -p /app/data/indexes /app/data/uploads
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

