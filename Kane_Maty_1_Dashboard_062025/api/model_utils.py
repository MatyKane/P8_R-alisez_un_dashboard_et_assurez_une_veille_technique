import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from src.config import MODEL_NAME, MODEL_STAGE, MLFLOW_REMOTE_URI
import mlflow.lightgbm

def set_tracking_uri():
    env = os.getenv("ENV", "dev")
    if env == "prod":
        print("ENV=prod : serveur distant MLflow")
        mlflow.set_tracking_uri(MLFLOW_REMOTE_URI)
    else:
        print("ENV=dev : MLflow local")
        # Fixe explicitement le chemin dans le conteneur
        mlflow.set_tracking_uri("file:///app/mlruns")

def load_model():
    # Charge modèle pyfunc depuis le chemin local ou distant selon tracking URI
    model_path = os.path.join(os.path.dirname(__file__), "model")
    return mlflow.pyfunc.load_model(model_path)

def load_model_lightgbm():
    model_path = os.path.join(os.path.dirname(__file__), "model")
    print(f"Chargement modèle LightGBM depuis : {model_path}")
    return mlflow.lightgbm.load_model(model_path)


def load_client_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "clients_test.csv")
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f" Fichier clients_test.csv introuvable à : {abs_path}")
    df = pd.read_csv(abs_path)
    df.set_index("SK_ID_CURR", inplace=True)
    return df

def convert_numeric_columns_to_model_dtype(model, df):
    input_schema = model.metadata.get_input_schema()
    if input_schema is None:
        print("Modèle sans schéma : conversion non appliquée.")
        return df

    type_map = {}
    for input_col in input_schema.inputs:
        col_name = input_col.name
        col_type = str(input_col.type).lower()
        if "double" in col_type or "float" in col_type:
            type_map[col_name] = 'float32'
        elif "int64" in col_type or "int" in col_type:
            type_map[col_name] = 'int32'
        else:
            type_map[col_name] = None

    for col, dtype in type_map.items():
        if dtype is not None and col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Conversion échouée sur {col} en {dtype}: {e}")
    return df

def predict_default(model, client_id, df_clients, seuil_metier=
0.5454545454545455):
    if client_id not in df_clients.index:
        return {"error": f"Client {client_id} non trouvé."}

    client_data = df_clients.loc[[client_id]].copy()
    client_data["SK_ID_CURR"] = client_id
    client_data = convert_numeric_columns_to_model_dtype(model, client_data)

    probas = model.predict(client_data)

    prediction = int(probas[0] >= seuil_metier)
    sexe = "F" if client_data.get("CODE_GENDER_F", False).values[0] else \
           ("M" if client_data.get("CODE_GENDER_M", False).values[0] else "N/A")

    return {
        "SK_ID_CURR": int(client_id),
        "CODE_GENDER (Sexe)": sexe,
        "CNT_CHILDREN (Nombre d'enfants)": int(client_data["CNT_CHILDREN"].values[0]),
        "AMT_INCOME_TOTAL (revenu total)": float(client_data["AMT_INCOME_TOTAL"].values[0]),
        "probability_default": float(probas[0]),
        "prediction": prediction,
        "seuil_metier": seuil_metier
    }

def get_shap_global(model_native, X_background):
    import shap
    explainer = shap.TreeExplainer(model_native)
    shap_values = explainer.shap_values(X_background)
    mean_shap = abs(shap_values).mean(axis=0)
    return {"features": X_background.columns.tolist(), "values": mean_shap.tolist()}

def get_shap_local(model_native, client_data):
    import shap
    explainer = shap.TreeExplainer(model_native)
    shap_values = explainer.shap_values(client_data)
    expected_value = explainer.expected_value
    return {
        "shap_values": shap_values[0].tolist() if isinstance(shap_values, list) else shap_values.tolist(),
        "expected_value": expected_value[0] if isinstance(expected_value, (list, tuple)) else expected_value,
        "features": client_data.iloc[0].to_dict()
    }

def get_feature_distribution(client_id, feature_name, df_clients):
    """
    Retourne les valeurs de la feature pour toute la population et la valeur du client.
    """
    if client_id not in df_clients.index:
        return {"error": f"Client {client_id} non trouvé."}

    if feature_name not in df_clients.columns:
        return {"error": f"Variable {feature_name} non trouvée dans les données."}

    population_values = df_clients[feature_name].dropna().tolist()
    client_value = df_clients.loc[client_id, feature_name]

    return {
        "feature": feature_name,
        "client_value": client_value,
        "population_values": population_values
    }