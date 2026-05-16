FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake in the pre-built database
COPY chroma_db/ ./chroma_db/
COPY vector_data/ ./vector_data/
COPY data/professors_clustered.json ./data/

# App code
COPY app.py pipeline.py ./

EXPOSE 7860

CMD ["streamlit", "run", "app.py",
     "--server.port=7860",
     "--server.address=0.0.0.0",
     "--server.headless=true"]
