FROM python:3.11-slim

# Copier le code API
COPY ./api /app/api
COPY api/ /app
COPY src /app/src

# Copier le dossier data depuis la racine du projet
COPY data/ /app/data/
COPY data/ /data/



# Installer unzip pour pouvoir dézipper data.zip
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*
# Copier uniquement le zip
COPY data.zip /app/data.zip
# Décompresser dans /app/data
RUN unzip /app/data.zip -d /app/data



# Copier le modèle MLflow (chemin relatif utilisé dans le code)
COPY api/model /app/model

# Copier le modèle MLflow (structure complète)
COPY mlruns /app/mlruns

WORKDIR /app

# Installer les outils de build + libsnappy
RUN apt-get update && \
    apt-get install -y build-essential gcc libsnappy-dev && \
    rm -rf /var/lib/apt/lists/*

# Mettre pip à jour
RUN pip install --upgrade pip

# Vérifier que le fichier est bien là
RUN ls -l /app

# Installer les dépendances du modèle
RUN pip install --no-cache-dir -r requirements_model.txt

# Installer les dépendances FastAPI
RUN pip install --no-cache-dir -r requirements.txt

# Ajout du chemin API au PYTHONPATH
ENV PYTHONPATH=/app

# Variable d'environnement par défaut pour MLflow (dev local)
ENV ENV=dev

# Exposer le port de l'API
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8001"]