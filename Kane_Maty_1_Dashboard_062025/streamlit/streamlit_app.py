import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# Configuration Streamlit
st.set_page_config(page_title="Interface Scoring Crédit", layout="wide")
st.title("Interface Scoring Crédit")

st.markdown("""
    <style>
    body {
        font-size: 18px;
        line-height: 1.6;
    }
    .stButton>button {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Afficher le logo
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    st.image(image, width=250)
else:
    st.warning("Logo non trouvé.")

# Détection de l’URL de l’API (mettre avant l’appel à check_api_available)
API_URL = os.getenv("API_URL", "http://fastapi-dashboard:8001")

def display_score_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': "Probabilité de défaut (%)", 'font': {'size': 24}}
    ))
    fig.update_layout(height=350)  # Taille réduite
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def check_api_available():
    try:
        resp = requests.get(f"{API_URL}/")
        return resp.status_code == 200
    except Exception:
        return False

if not check_api_available():
    st.error("API FastAPI non disponible. Veuillez réessayer plus tard.")
    st.stop()

# Navigation latérale
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choisissez une section :", [
    "🏠 Accueil",
    "🏦 Scoring & Analyse"
])
st.sidebar.markdown(f"API utilisée : `{API_URL}`")

# Récupération des ID clients
@st.cache_data
def get_client_ids():
    try:
        response = requests.get(f"{API_URL}/clients")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors de la récupération des ID clients : {e}")
        return []

client_ids = get_client_ids()
client_id = None
if client_ids:
    client_id = st.selectbox("Choisir un ID client", client_ids)

    # Réinitialiser les affichages quand l'ID client change
    if "last_client_id" not in st.session_state:
        st.session_state.last_client_id = client_id
    if client_id != st.session_state.last_client_id:
        st.session_state.show_prediction = False
        st.session_state.show_shap_global = False
        st.session_state.show_shap_local = False
        st.session_state.show_comparison = False
        st.session_state.show_bivariate = False
        st.session_state.last_client_id = client_id

# ----- Accueil -----
if section == "🏠 Accueil":
    st.markdown("""
    Bienvenue sur l'application de scoring de risque de défaut.  
                
    Cette application prédit la probabilité qu'un client ne rembourse pas son crédit,  
    grâce à un modèle de machine learning entraîné sur des données réelles.
    
    Saisissez un identifiant client pour obtenir la prédiction et des explications SHAP.
    """)


# Initialisation dans session_state
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
if "show_shap_global" not in st.session_state:
    st.session_state.show_shap_global = False
if "show_shap_local" not in st.session_state:
    st.session_state.show_shap_local = False
if "show_comparison" not in st.session_state:
    st.session_state.show_comparison = False
if "show_bivariate" not in st.session_state: 
    st.session_state.show_bivariate = False

# "Dashboard complet"
if section == "🏦 Scoring & Analyse":

    st.header("📈 Prédiction")
    if st.button("Prédire le risque"):
        st.session_state.show_prediction = True

    if st.session_state.show_prediction:
        if client_id is None:
            st.error("Aucun ID client sélectionné.")
        elif client_id <= 0:
            st.error("Veuillez saisir un ID client valide (>0).")
        else:
            try:
                resp = requests.get(f"{API_URL}/predict/{client_id}")
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    score = data.get("prediction", 0)
                    proba = data.get("probability_default", 0)
            
                    st.subheader("Résultat de la prédiction :")
                    st.markdown(f"### Crédit : {'Refusé' if score==1 else 'Accepté'}")
                    st.markdown(f"### Probabilité de défaut : **{round(proba * 100)} %**")
                    display_score_gauge(proba)

                    st.json(data)
            except Exception as e:
                st.error(f"Erreur lors de la requête API : {e}")
    
    # ----- SHAP Global -----
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🔍 SHAP Global", expanded=st.session_state.show_shap_global):
            if st.button("Afficher l'explication globale (SHAP)", key="btn_shap_global"):
                st.session_state.show_shap_global = True
            if st.session_state.show_shap_global:
                try:
                    resp = requests.get(f"{API_URL}/shap/global")
                    resp.raise_for_status()
                    shap_data = resp.json()

                    df_shap = pd.DataFrame({
                        "Feature": shap_data["features"],
                        "Importance": shap_data["values"]
                    }).sort_values("Importance", ascending=True)

                    top_features = df_shap.tail(10)

                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.barh(top_features["Feature"], top_features["Importance"])
                    ax.set_title("Importance des variables (SHAP global)", fontweight="bold")
                    plt.close(fig)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur lors de la récupération SHAP global : {e}")
    
    # ----- SHAP Local -----
    with col2:
        with st.expander("🎯 SHAP Local", expanded=st.session_state.show_shap_local):
            if st.button("Afficher l'explication locale (SHAP)", key="btn_shap_local"):
                st.session_state.show_shap_local = True
            if st.session_state.show_shap_local:
                if client_id is None:
                    st.error("Veuillez sélectionner un ID client valide.")
                else:
                    try:
                        resp = requests.get(f"{API_URL}/shap/local/{client_id}")
                        resp.raise_for_status()
                        shap_data = resp.json()

                        shap_values = np.array(shap_data["shap_values"])
                        expected_value = shap_data["expected_value"]
                        features = shap_data["features"]

                        explainer = shap.Explanation(
                            values=shap_values,
                            base_values=expected_value,
                            data=pd.DataFrame([features]),
                            feature_names=list(features.keys())
                        )

                        shap.plots.waterfall(explainer[0], max_display=10, show=False)
                        fig = plt.gcf()
                        fig.suptitle("Explication locale (SHAP local)", fontsize=12, fontweight="bold")
                        st.pyplot(fig)
                        plt.close(fig)

                    except Exception as e:
                        st.error(f"Erreur lors de la récupération SHAP local : {e}")
    
    # ----- comparaison Clients -----
    col_comp, col_biv = st.columns(2)

    with col_comp:
        with st.expander("👥 Comparaison Client", expanded=st.session_state.show_comparison):
            feature = st.selectbox("Choisir une variable à comparer", ["AMT_INCOME_TOTAL", "CNT_CHILDREN"])
            if st.button("Afficher la comparaison"):
                st.session_state.show_comparison = True
            if st.session_state.show_comparison:
                try:
                    response = requests.get(f"{API_URL}/client_feature_distribution/{feature}?client_id={client_id}")
                    response.raise_for_status()
                    data = response.json()

                    client_value = data["client_value"]
                    all_values = data["population_values"]

                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    ax.hist(all_values, bins=30, alpha=0.7, color='#69b3a2', label='Population')
                    ax.axvline(client_value, color='crimson', linestyle='--', linewidth=2, label='Client')
                    ax.set_title(f"Distribution de {feature}", fontsize=11, fontweight="bold")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Nombre de clients")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Erreur récupération données de comparaison : {e}")

        # ----- Analyse Bivariée -----
    with col_biv:
        with st.expander("📊 Analyse Bivariée"):
            st.markdown("Sélectionnez deux variables pour analyser leur relation dans la population :")

            feature_x = st.selectbox("Variable X", ["AMT_INCOME_TOTAL", "CNT_CHILDREN"], key="bivariate_x")
            feature_y = st.selectbox("Variable Y", ["AMT_INCOME_TOTAL", "CNT_CHILDREN"], key="bivariate_y")

            if st.button("Afficher l'analyse bivariée"):
                try:
                    response = requests.get(f"{API_URL}/bivariate_analysis?x={feature_x}&y={feature_y}")
                    response.raise_for_status()
                    data = response.json()

                    x_values = data["x"]
                    y_values = data["y"]

                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    scatter = ax.scatter(
                        x_values, y_values, 
                        c=y_values,  # Couleur selon y
                        cmap='viridis',
                        alpha=0.7
                    )
                    ax.set_xlabel(feature_x)
                    ax.set_ylabel(feature_y)
                    ax.set_title(f"{feature_x} vs {feature_y}", fontsize=12, fontweight="bold")
                    plt.tight_layout()
                    fig.colorbar(scatter, ax=ax, label=feature_y)
                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Erreur lors de la récupération des données bivariées : {e}")