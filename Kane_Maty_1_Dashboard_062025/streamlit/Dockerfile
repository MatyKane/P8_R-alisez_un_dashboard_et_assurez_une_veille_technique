FROM python:3.11-slim

WORKDIR /app

# Installer dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Installer dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier l’application
COPY streamlit_app.py .
COPY logo.png .

# Secrets et configuration
COPY .streamlit /root/.streamlit

# Port Streamlit
EXPOSE 8501

ENV ENV=dev

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]