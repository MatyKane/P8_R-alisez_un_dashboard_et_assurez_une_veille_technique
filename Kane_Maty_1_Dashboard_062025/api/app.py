import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi import FastAPI, HTTPException
from api.model_utils import (
    set_tracking_uri,
    load_model,
    load_model_lightgbm,
    load_client_data,
    convert_numeric_columns_to_model_dtype,
    predict_default,
    get_shap_global,
    get_shap_local,
    get_feature_distribution,
)
import uvicorn

app = FastAPI(title="API Scoring Crédit")

try:
    # Fixe le tracking URI MLflow selon ENV (prod ou dev)
    set_tracking_uri()

    model_pyfunc = load_model()           # modèle pyfunc (local ou distant selon URI)
    model_native = load_model_lightgbm()  # modèle LightGBM natif (utilise URI fixé par set_tracking_uri)
    df_clients = load_client_data()
    df_clients.index = df_clients.index.astype(int)

    X_background = df_clients.head(100).copy()
    X_background = convert_numeric_columns_to_model_dtype(model_pyfunc, X_background)

except Exception as e:
    raise RuntimeError(f"Erreur au démarrage de l'API : {e}")

@app.get("/clients")
def get_client_ids():
    """Retourne la liste des identifiants clients disponibles"""
    try:
        return df_clients.index.astype(int).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération IDs : {e}")
    
@app.get("/")
def root():
    return {"message": "API prédiction risque de défaut prête"}

@app.get("/predict/{client_id}")
def predict(client_id: int, seuil: float = 
0.5454545454545455):
    result = predict_default(model_pyfunc, client_id, df_clients, seuil_metier=seuil)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/shap/global")
def shap_global():
    try:
        X_bg = X_background.reset_index()
        return get_shap_global(model_native, X_bg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP global : {e}")

@app.get("/shap/local/{client_id}")
def shap_local(client_id: int):
    if client_id not in df_clients.index:
        raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé.")
    try:
        client_data = df_clients.loc[[client_id]].copy().reset_index()
        client_data = convert_numeric_columns_to_model_dtype(model_pyfunc, client_data)
        return get_shap_local(model_native, client_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP local : {e}")

@app.get("/client_feature_distribution/{feature_name}")
def feature_distribution(client_id: int, feature_name: str):
    """
    Retourne les valeurs de la feature pour toute la population
    + la valeur du client spécifié.
    """
    result = get_feature_distribution(client_id, feature_name, df_clients)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

from fastapi.responses import JSONResponse  # ajouter l'import si ce n'est pas déjà fait

@app.get("/bivariate_analysis")
def get_bivariate_data(x: str, y: str):
    if x not in df_clients.columns or y not in df_clients.columns:
        return JSONResponse(status_code=400, content={"error": "Colonnes invalides."})
    
    return {
        "x": df_clients[x].dropna().tolist(),
        "y": df_clients[y].dropna().tolist()
    }

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8001)