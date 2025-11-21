# =====================================================
# 🏦 Credit Risk Dashboard – Versión Ejecutiva
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Configuración de Streamlit
# -------------------------------
st.set_page_config(
    page_title="💳 Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar corporativo
# -------------------------------
st.sidebar.image("assets/bank_logo.png", use_column_width=True)
st.sidebar.title("📌 Menú")
page = st.sidebar.radio(
    "Selecciona sección:",
    ["🏦 Dashboard Corporativo", "🧠 Predicción Crediticia", "📄 Reporte Entrenamiento"]
)

# -------------------------------
# Carga de modelos y preprocesador
# -------------------------------
@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessor.joblib")
    model_nn = load_model("keras_model.h5")
    model_rf = joblib.load("rf_best.joblib")
    model_lgbm = joblib.load("lgb_best.joblib")
    return preprocessor, model_nn, model_rf, model_lgbm

try:
    preprocessor, model_nn, model_rf, model_lgbm = load_models()
    models_loaded = True
except:
    models_loaded = False


uri = "mongodb+srv://Mongo:Herrera123@mongoscar.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
db_name = "CreditDB"
collection = "LoanApproval"

# -------------------------------
# Función para cargar datos desde MongoDB Azure
# -------------------------------

@st.cache_data
def load_data_mongo(uri, db_name, collection, n_rows=50000):
    client = MongoClient(uri)
    db = client[db_name]
    col = db[collection]
    data = list(col.find().limit(n_rows))
    df = pd.DataFrame(data)
    return df

# -------------------------------
# Sidebar de conexión MongoDB
# -------------------------------
if page in ["🏦 Dashboard Corporativo", "🧠 Predicción Crediticia"]:
    st.sidebar.subheader("Conexión MongoDB Azure")
    mongo_uri = st.sidebar.text_input("Mongo URI", "mongodb+srv://<user>:<password>@cluster0.mongodb.net/test")
    db_name = st.sidebar.text_input("Base de datos", "credit_db")
    collection_name = st.sidebar.text_input("Colección", "applications")
    
    if st.sidebar.button("🔄 Cargar datos"):
        with st.spinner("Cargando datos desde MongoDB Azure..."):
            df = load_data_mongo(mongo_uri, db_name, collection_name)
            st.success(f"Datos cargados: {df.shape[0]} filas x {df.shape[1]} columnas")

# ================================
# 1️⃣ Dashboard Corporativo Ejecutiva
# ================================
if page == "🏦 Dashboard Corporativo":
    st.title("🏦 Credit Risk Dashboard – Executive Edition")

    if 'df' in locals():
        # KPIs dinámicos
        st.markdown("### 📊 KPIs Financieros")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        approval_rate = df['loan_status'].mean() * 100
        kpi1.metric("📈 Tasa Aprobación", f"{approval_rate:.2f}%")
        kpi2.metric("💰 Promedio Ingreso Anual", f"${df['annual_income'].mean():,.0f}")
        kpi3.metric("🏦 Promedio Deuda", f"${df['current_debt'].mean():,.0f}")
        kpi4.metric("💳 Promedio Monto Préstamo", f"${df['loan_amount'].mean():,.0f}")

        st.markdown("---")

        # Distribuciones numéricas con Plotly
        numeric_cols = [
            'age', 'years_employed', 'annual_income', 'credit_score',
            'credit_history_years', 'savings_assets', 'current_debt',
            'loan_amount', 'interest_rate', 'debt_to_income_ratio',
            'loan_to_income_ratio', 'payment_to_income_ratio'
        ]
        st.markdown("### 📈 Distribuciones Numéricas")
        for col in numeric_cols:
            fig = px.histogram(
                df, x=col, color='loan_status',
                color_discrete_map={0:'firebrick', 1:'green'},
                marginal="box", nbins=50,
                title=f"Distribución de {col}"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Distribuciones categóricas
        categorical_cols = ['occupation_status', 'product_type', 'loan_intent']
        st.markdown("### 🏷 Distribuciones Categóricas")
        for col in categorical_cols:
            fig = px.histogram(
                df, x=col, color='loan_status',
                color_discrete_map={0:'firebrick',1:'green'},
                title=f"{col} vs Loan Status"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Matriz de correlación
        st.markdown("### 🔗 Correlaciones Numéricas")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Matriz de Correlación"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # PCA 3D interactivo
        st.markdown("### 🎯 PCA 3D – Separación por Loan Status")
        df_numeric = df[numeric_cols].fillna(0)
        pca = PCA(n_components=3)
        pca_res = pca.fit_transform(df_numeric)
        df['pca1'], df['pca2'], df['pca3'] = pca_res[:,0], pca_res[:,1], pca_res[:,2]
        fig3d = px.scatter_3d(
            df, x='pca1', y='pca2', z='pca3',
            color='loan_status',
            color_discrete_map={0:'firebrick',1:'green'},
            opacity=0.7,
            title="PCA 3D: Loan Status"
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("Carga los datos desde el sidebar para ver el dashboard.")

# ================================
# 2️⃣ Predicción Crediticia Automática
# ================================
if page == "🧠 Predicción Crediticia":
    st.title("🧠 Predicción Crediticia – Executive Edition")

    if 'df' not in locals():
        st.info("Carga los datos desde MongoDB Azure primero.")
    elif not models_loaded:
        st.warning("No se han cargado los modelos.")
    else:
        st.success("Modelos cargados correctamente.")

        with st.form("input_form"):
            st.subheader("Ingrese datos del solicitante")
            col1, col2 = st.columns(2)
            # Col1
            with col1:
                age = st.number_input("Edad:", 18,70,30)
                years_employed = st.number_input("Años Empleado:",0,40,3)
                annual_income = st.number_input("Ingreso Anual ($):",15000,250000,50000)
                credit_score = st.number_input("Credit Score:",300,850,650)
                credit_history_years = st.number_input("Historial Crediticio (años):",0,30,5)
                savings_assets = st.number_input("Ahorros/Assets:",0,300000,5000)
                current_debt = st.number_input("Deuda Actual:",0,200000,10000)
                defaults_on_file = st.number_input("Defaults on file (0/1):",0,1,0)
                delinquencies_last_2yrs = st.number_input("Delinquencies last 2yrs:",0,10,0)
                derogatory_marks = st.number_input("Derogatory marks:",0,5,0)
            # Col2
            with col2:
                product_type = st.selectbox("Tipo Producto:",['Credit Card','Personal Loan','Line of Credit'])
                loan_intent = st.selectbox("Intención Préstamo:",['Personal','Education','Medical','Business','Home Improvement','Debt Consolidation'])
                loan_amount = st.number_input("Monto Préstamo ($):",500,100000,10000)
                interest_rate = st.number_input("Tasa de Interés (%):",6,23,15)
                debt_to_income_ratio = st.number_input("Debt-to-Income Ratio:",0.0,0.8,0.3)
                loan_to_income_ratio = st.number_input("Loan-to-Income Ratio:",0.0,2.0,0.7)
                payment_to_income_ratio = st.number_input("Payment-to-Income Ratio:",0.0,0.7,0.2)
                occupation_status = st.selectbox("Ocupación:",['Employed','Self-Employed','Student'])

            submitted = st.form_submit_button("🔮 Predecir")

        if submitted:
            new_data = pd.DataFrame({
                "age":[age], "years_employed":[years_employed], "annual_income":[annual_income],
                "credit_score":[credit_score], "credit_history_years":[credit_history_years],
                "savings_assets":[savings_assets], "current_debt":[current_debt],
                "defaults_on_file":[defaults_on_file],
                "delinquencies_last_2yrs":[delinquencies_last_2yrs],
                "derogatory_marks":[derogatory_marks],
                "product_type":[product_type], "loan_intent":[loan_intent],
                "loan_amount":[loan_amount], "interest_rate":[interest_rate],
                "debt_to_income_ratio":[debt_to_income_ratio],
                "loan_to_income_ratio":[loan_to_income_ratio],
                "payment_to_income_ratio":[payment_to_income_ratio],
                "occupation_status":[occupation_status]
            })

            # Transformar con pipeline
            new_transformed = preprocessor.transform(new_data)

            # Predicciones
            pred_nn = model_nn.predict(new_transformed)
            pred_nn_label = int((pred_nn >= 0.5)[0][0])
            pred_rf = model_rf.predict(new_transformed)[0]
            pred_lgbm = model_lgbm.predict(new_transformed)[0]

            st.markdown("### Resultados individuales")
            st.write(f"🔹 Neural Network: {'Aprobado ✅' if pred_nn_label==1 else 'Rechazado ❌'}")
            st.write(f"🔹 Random Forest: {'Aprobado ✅' if pred_rf==1 else 'Rechazado ❌'}")
            st.write(f"🔹 LightGBM: {'Aprobado ✅' if pred_lgbm==1 else 'Rechazado ❌'}")

            # Consenso
            final = int((pred_nn_label + pred_rf + pred_lgbm) >= 2)
            st.markdown("---")
            if final:
                st.success("💳 PREDICCIÓN FINAL: APROBADO")
            else:
                st.error("❌ PREDICCIÓN FINAL: RECHAZADO")

# ================================
# 3️⃣ Reporte HTML de entrenamiento
# ================================
if page == "📄 Reporte Entrenamiento":
    st.title("📄 Reporte HTML – Entrenamiento")
    html_files = list(Path("reports/").glob("*.html"))
    if html_files:
        option = st.selectbox("Selecciona Reporte", html_files)
        html_content = Path(option).read_text()
        st.components.v1.html(html_content, height=1200, scrolling=True)
    else:
        st.info("No se encontraron reportes HTML en la carpeta 'reports/'.")
