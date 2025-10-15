# streamlit_bl_app.py
# Streamlit app that reproduces the models in Black_Litterman_portmodelo.ipynb

import os
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from pypfopt import BlackLittermanModel, EfficientFrontier, objective_functions
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Expected assets and renaming from your notebook
ORIGINAL_TO_RENAMED = {
    'SPX Index': 'USA',
    'MXEUG Index': 'Europa equities',
    'UKX Index': 'UK',
    'MXJP Index': 'Japon',
    'MXAPJ Index': 'Asia',
    'MXLA Index': 'Latam',
    'LF98TRUU Index': 'US HY',
    'LUACTRUU Index': 'US IG',
    'LBEATRUH Index': 'Europa bonds',
    'BSELTRUU Index': 'Latam corp',
    'BSSUTRUU Index': 'Emerging sov',
    'CABS Index': 'ABS',
    'BCOMTR Index': 'Commodities',
    'GLD US EQUITY': 'Oro',
    'Private Debt': 'Private Debt', 
    'Private Equity': 'Private Equity',
    'MXWD Index': 'World equities'
}

# Column order for models (first 6 equities, next 6 bonds, last 2 commodities)
MODEL_ASSETS = [
    "USA", "Europa equities", "UK", "Japon", "Asia", "Latam",
    "US HY", "US IG", "Europa bonds", "Latam corp", "Emerging sov", "ABS",
    "Commodities", "Oro", "Private Debt", "Private Equity", 
]

# Default priors, views, confidences from notebook
DEFAULT_PRIORS = {
    "USA": 0.055, "Europa equities": 0.083, "UK": 0.067, "Japon": 0.06,
    "Asia": 0.08, "Latam": 0.066, "US HY": 0.052, "US IG": 0.045,
    "Europa bonds": 0.024, "Latam corp": 0.074, "Emerging sov": 0.071,
    "ABS": 0.05, "Commodities": 0.045, "Oro": 0.09, "Private Debt": 0.085,"Private Equity": 0.080
}

DEFAULT_VIEWS = {
    "USA": 0.10, "Europa equities": 0.06, "UK": 0.07, "Japon": 0.06,
    "Asia": 0.085, "Latam": 0.07, "US HY": 0.06, "US IG": 0.045,
    "Europa bonds": 0.05, "Latam corp": 0.075, "Emerging sov": 0.075,
    "ABS": 0.05, "Commodities": 0.05, "Oro": 0.10, "Private Debt": 0.085,
    "Private Equity": 0.080
}

DEFAULT_CONFIDENCES = {
    "USA": 0.6, "Europa equities": 0.5, "UK": 0.5, "Japon": 0.5,
    "Asia": 0.6, "Latam": 0.4, "US HY": 0.5, "US IG": 0.75,
    "Europa bonds": 0.5, "Latam corp": 0.6, "Emerging sov": 0.5,
    "ABS": 0.8, "Commodities": 0.5, "Oro": 0.5,
    "Private Debt": 0.85, "Private Equity": 0.55
}

# ---------- Helpers (risk metrics and constraints) ----------
def cvar_loss(w, S, alpha=0.05):
    # CVaR of portfolio returns sampled from historical returns matrix S (T x N or using vectorized returns)
    portf_rets = S @ w
    var = np.percentile(portf_rets, 100 * alpha)
    if var >= 0:
        var = 0
    cvar = (var - (1 / (alpha * len(portf_rets))) * np.sum(np.maximum(var - portf_rets, 0))) * np.sqrt(365)
    return -cvar

def var_loss(w, mu, alpha=0.05):
    portf_rets = mu @ w
    var = np.percentile(portf_rets, 100 * alpha) * np.sqrt(365)
    if var >= 0:
        var = 0
    return -var

def port_vol(w, cov):
    return np.sqrt(np.dot(w.T, np.dot(cov, w)))

def max_weight_constraint_list(returns_modelos, portafolio_modelo):
    constraints = []
    for asset in portafolio_modelo.index:

        constraints.append({
            'type': 'ineq', 
            'fun': lambda w, a=asset: 
                (1.2 * portafolio_modelo[a]) - w[returns_modelos.columns.get_loc(a)]
        })
    return constraints 

def min_weight_constraint_list(returns_modelos, portafolio_modelo):
    constraints = []
    for asset in portafolio_modelo.index:
        constraints.append({
            'type': 'ineq', 
            'fun': lambda w, a=asset: 
                w[returns_modelos.columns.get_loc(a)] - (0.8 * portafolio_modelo[a])
        })
    return constraints 

def neg_sharpe_penalizado(w, mu, cov):
    port_return = np.dot(w, mu)
    sigma = np.sqrt(np.dot(w, np.dot(cov, w))) * np.sqrt(365)
    sharpe = port_return / sigma if sigma > 0 else 0
    sharpe = max(sharpe, 1e-5)  # avoid negative zeroing
    return -sharpe

# ---------- Data loading ----------
def load_data_from_excel(file_obj_or_path):
    excel_file = pd.ExcelFile(file_obj_or_path)
    df = excel_file.parse(
        sheet_name='Retornos', index_col=0, parse_dates=True
    )

    df = df.sort_index(ascending=True)
    df = df.rename(columns=ORIGINAL_TO_RENAMED)

    # Keep only known assets to avoid surprises
    df = df[[c for c in df.columns if c in MODEL_ASSETS]]

    # Returns and slice for models
    returns = df.pct_change().dropna(how="all")
    returns_modelos = returns[MODEL_ASSETS].dropna(how="all")

    # Load portfolio weights
    portafolio_modelo = excel_file.parse(
        sheet_name='Pesos', index_col=0, header=None).squeeze("columns").to_dict()
    portafolio_modelo = pd.Series(portafolio_modelo)
    portafolio_modelo = portafolio_modelo.reindex(MODEL_ASSETS)

    return df, returns, returns_modelos, portafolio_modelo

# ---------- Models ----------
def run_black_litterman(returns_modelos, priors, views, confidences):
    S = returns_modelos.cov() * 365  # annualized
    bl = BlackLittermanModel(
        S,
        pi=pd.Series(priors),
        absolute_views=views,
        omega="idzorek",
        view_confidences=list(confidences.values())
    )
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()
    vol = np.sqrt(np.diag(S_bl))   # Volatility of the posterior covariance

    ef = EfficientFrontier(ret_bl, S_bl)  # expects pandas
    ef.add_objective(objective_functions.L2_reg)

    #Restricciones por asset class
    for asset in portafolio_modelo.index:
        ef.add_constraint(lambda w, asset_name=asset: w[returns_modelos.columns.get_loc(asset_name)] >= 0.8 * portafolio_modelo[asset_name])

    for asset in portafolio_modelo.index:
        ef.add_constraint(lambda w, asset_name=asset: w[returns_modelos.columns.get_loc(asset_name)] <= 1.2 * portafolio_modelo[asset_name])


    ef.max_sharpe()
    weights = pd.Series(ef.clean_weights())
    weights = weights.reindex(MODEL_ASSETS, fill_value=0.0)

    portbl_vol = port_vol(weights, S_bl.values) * 100.0

    port_return = float((weights @ ret_bl).round(4)) * 100.0
    return weights, ret_bl, vol, port_return, portbl_vol

def run_mvo(returns_modelos, priors, views, confidences):
    # Use BL posterior mu & cov, then custom SLSQP with constraints
    S = returns_modelos.cov() * 365
    bl = BlackLittermanModel(
        S,
        pi=pd.Series(priors),
        absolute_views=views,
        omega="idzorek",
        view_confidences=list(confidences.values())
    )
    mu = bl.bl_returns()
    cov = bl.bl_cov().values
    vol = np.sqrt(np.diag(cov))  # Volatility of the posterior covariance 
    n = len(MODEL_ASSETS)

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: w}  # non-negative weights
    ]

    cons += max_weight_constraint_list(returns_modelos, portafolio_modelo)
    cons += min_weight_constraint_list(returns_modelos, portafolio_modelo)

    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(
        neg_sharpe_penalizado, w0, args=(mu.values, cov), method='SLSQP',
        bounds=bounds, constraints=cons, options={'maxiter': 1000}
    )
    if not res.success:
        raise RuntimeError(f"MVO optimization failed: {res.message}")

    weights = pd.Series(res.x, index=MODEL_ASSETS)
    port_return = float((weights @ mu).round(4)) * 100.0
    port_mvo_vol = port_vol(weights, cov) * 100.0
    return weights, mu, cov, port_return, port_mvo_vol

def create_asset_risk_return_plot(dataframe, x_col, y_col, title):
    """Crear gráfico de retorno vs riesgo"""
    fig = px.scatter(
        dataframe, 
        x=x_col, 
        y=y_col,
        text='Activo', # Usa la columna 'Activo' para etiquetar cada punto
        title=title,
        # El hover_data ahora usa las columnas que sí existen en el DataFrame
        hover_data={'Activo': True, 'Volatilidad': ':.2%', 'Retorno': ':.2%'}
    )

    # Personalizar el gráfico
    # Personalizar para mayor legibilidad
    fig.update_traces(
        textposition='top center', 
        marker=dict(size=10, opacity=0.7)
    )
    fig.update_layout(
        xaxis_title='Volatilidad Anualizada',
        yaxis_title='Retorno Anualizado',
        xaxis_tickformat='.0%',
        yaxis_tickformat='.0%',
        showlegend=False # No necesitamos leyenda para esto
    )
    
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Black-Litterman Portfolio Models", layout="wide")
st.title("Black-Litterman and Portfolio Optimization")

# Sidebar: data input
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload Excel with price index series", type=["xlsx"])

# (Asegúrate de tener tus constantes MODEL_ASSETS, DEFAULT_PRIORS, etc., definidas antes)
st.subheader("Priors, Views, and Confidences")

param_df = pd.DataFrame({
    "asset": MODEL_ASSETS,
    "prior": [DEFAULT_PRIORS.get(a, 0) for a in MODEL_ASSETS],
    "view": [DEFAULT_VIEWS.get(a, 0) for a in MODEL_ASSETS],
    "confidence": [DEFAULT_CONFIDENCES.get(a, 0) for a in MODEL_ASSETS],
})

# --- PASO 1: Preparar una copia para el editor, con valores en porcentaje (ej: 5.0) ---
param_df_display = param_df.copy()
cols_to_format = ["prior", "view", "confidence"]
for col in cols_to_format:
    param_df_display[col] = param_df_display[col] * 100

# --- PASO 2: Usar el DataFrame de visualización en el data_editor y ajustar el formato ---
edited_params_display = st.data_editor(
    param_df_display,
    num_rows="fixed",
    column_config={
        "asset": st.column_config.TextColumn(disabled=True),
        "prior": st.column_config.NumberColumn(format="%.2f%%", step=0.1),
        "view": st.column_config.NumberColumn(format="%.2f%%", step=0.1),
        "confidence": st.column_config.NumberColumn(format="%.2f%%", step=5.0, min_value=0.0, max_value=100.0),
    }
)

# --- PASO 3: Convertir los datos editados de vuelta a su escala decimal para los cálculos ---
edited_params = edited_params_display.copy()
for col in cols_to_format:
    edited_params[col] = edited_params[col] / 100

# --- PASO 4: Preparar los diccionarios con los valores correctos (en decimal) ---
priors = dict(zip(edited_params["asset"], edited_params["prior"]))
views = dict(zip(edited_params["asset"], edited_params["view"]))
confidences = dict(zip(edited_params["asset"], edited_params["confidence"]))

run_btn = st.button("Run models")
import altair as alt

# Main run
if run_btn:
    try:
        # --- PASO 1: CARGA DE DATOS ---
        if uploaded is not None:
            data, returns, returns_modelos, portafolio_modelo = load_data_from_excel(uploaded)
        else:
            st.error("Please upload an Excel file or provide a valid absolute path.")
            st.stop()

        returns_modelos = returns_modelos[MODEL_ASSETS] # Asegura el orden
        st.success("Data loaded successfully.")

        # --- PASO 2: EJECUTAR TODOS LOS MODELOS ---
        # Se ejecutan todos los cálculos primero para tener los resultados listos.
        bl_w, bl_mu, bl_S, bl_ret, bl_vol = run_black_litterman(returns_modelos, priors, views, confidences)
        mvo_w, _, _, mvo_ret, mvo_vol = run_mvo(returns_modelos, priors, views, confidences)

        # --- PASO 3: VISUALIZACIÓN DE RESULTADOS ---

        # 3.1: Mostrar los Retornos Posteriores (el resultado principal de Black-Litterman)
        st.divider()
        st.header("1. Retornos Esperados Posteriores (Anualizados)")
        st.write("Estos son los retornos ajustados por el modelo Black-Litterman, que se usarán como base para todas las estrategias de optimización.")

        # 1. Crea dos columnas de igual tamaño
        col1, col2 = st.columns(2)

        # 2. Usa la primera columna (col1) para mostrar el dataframe
        with col1:
            st.dataframe(
                pd.Series(bl_mu)
                .multiply(100)
                .to_frame(name="Retorno (%)")
                .style.format("{:.2f}%")
            )

        # 3.2: Comparar las Ponderaciones y Retornos de cada estrategia
        st.divider()
        st.header("2. Comparación de Estrategias")

        # Función auxiliar para crear gráficos y no repetir código
        def create_weights_chart(weights_series):
            weights_df = weights_series.rename("Ponderación").reset_index()
            weights_df.columns = ["Activo", "Ponderación"]
            chart = alt.Chart(weights_df).mark_bar().encode(
                x=alt.X('Activo:N', sort=None, title=None),
                y=alt.Y('Ponderación:Q', title="Ponderación (%)", axis=alt.Axis(format=".2%")),
                tooltip=[
                    alt.Tooltip('Activo', title='Activo'),
                    alt.Tooltip('Ponderación', title='Ponderación', format='.2%')
                ]
            )
            return chart

        # Crear 3 columnas para mostrar los resultados lado a lado
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Black-Litterman (Max Sharpe)")
            st.metric("Retorno del Portafolio", f"{bl_ret:.2f}%")
            st.metric("Volatilidad del Portafolio", f"{bl_vol:.2f}%")
            chart_bl = create_weights_chart(bl_w) # Gráfico con los pesos correctos
            st.altair_chart(chart_bl, use_container_width=True)

        with col2:
            st.subheader("Portafolio Modelo")
            # Usamos st.metric con un valor que reserve el espacio vertical
            # y evitamos mostrar información al usuario (etiquetas vacías con &nbsp;)
            st.metric(label="&nbsp;", value="-") 
            st.metric(label="&nbsp;", value="-")
            chart_modelo = create_weights_chart(portafolio_modelo) # Gráfico con los pesos correctos
            st.altair_chart(chart_modelo, use_container_width=True)

        with col3:
            st.subheader("Mean-Variance Optimization")
            st.metric("Retorno del Portafolio", f"{mvo_ret:.2f}%")
            st.metric("Volatilidad del Portafolio", f"{mvo_vol:.2f}%")
            chart_mvo = create_weights_chart(mvo_w) # Gráfico con los pesos correctos
            st.altair_chart(chart_mvo, use_container_width=True)


        # Tabla comparativa de ponderaciones
        df_comparacion = pd.DataFrame({
            'Black-Litterman': bl_w,
            'Portafolio Modelo': portafolio_modelo,
            'Diferencia': bl_w - portafolio_modelo
        }).reindex(MODEL_ASSETS).reset_index().rename(columns={'index': 'Activo'})

        df_comparacion_equities = pd.DataFrame({
            'Black-Litterman': bl_w[0:6]/bl_w[0:6].sum(),
            'Portafolio Modelo': portafolio_modelo[0:6]/portafolio_modelo[0:6].sum(),
            'Diferencia': (bl_w[0:6]/bl_w[0:6].sum()) - (portafolio_modelo[0:6]/portafolio_modelo[0:6].sum())
        }).reset_index().rename(columns={'index': 'Activo'})
        
        # 1. Calculamos las sumas para las columnas relevantes usando .loc para mayor claridad
        suma_bl = df_comparacion['Black-Litterman'].iloc[0:6].sum()
        suma_modelo = df_comparacion['Portafolio Modelo'].iloc[0:6].sum()

        # 2. Creamos la nueva fila como un DataFrame de 1xN
        #    Añadimos también la columna 'Activo' y 'Diferencia' para que coincida con el esquema
        total_equities_df = pd.DataFrame(
            {
                'Activo': ['Total Equity'],
                'Black-Litterman': [suma_bl],
                'Portafolio Modelo': [suma_modelo],
                'Diferencia': [suma_bl - suma_modelo]
            }
        )

        # 3. Concatenamos la nueva fila a la cabeza del DataFrame original
        #    Pasamos la nueva fila y el DataFrame original dentro de una LISTA []
        df_comparacion_equities = pd.concat(
            [total_equities_df, df_comparacion_equities],  # <- Lista de DataFrames
            ignore_index=True
        )

        #COMPARACIÓN BONOS
        BOND_ASSETS = MODEL_ASSETS[6:12]  

        df_comparacion_bonos = pd.DataFrame({
            'Black-Litterman': bl_w[6:12]/bl_w[6:12].sum(),
            'Portafolio Modelo': portafolio_modelo[6:12]/portafolio_modelo[6:12].sum(),
            'Diferencia': (bl_w[6:12]/bl_w[6:12].sum()) - (portafolio_modelo[6:12]/portafolio_modelo[6:12].sum())
        }).reindex(BOND_ASSETS).reset_index().rename(columns={'index': 'Activo'})

        # 1. Calculamos las sumas para las columnas relevantes usando .loc para mayor claridad
        suma_bl = df_comparacion['Black-Litterman'].iloc[6:12].sum()
        suma_modelo = df_comparacion['Portafolio Modelo'].iloc[6:12].sum()

        # 2. Creamos la nueva fila como un DataFrame de 1xN
        #    Añadimos también la columna 'Activo' y 'Diferencia' para que coincida con el esquema
        total_bonos_df = pd.DataFrame(
            {
                'Activo': ['Total Bonos'],
                'Black-Litterman': [suma_bl],
                'Portafolio Modelo': [suma_modelo],
                'Diferencia': [suma_bl - suma_modelo]
            }
        )

        # 3. Concatenamos la nueva fila a la cabeza del DataFrame original
        df_comparacion_bonos = pd.concat(
            [total_bonos_df, df_comparacion_bonos], 
            ignore_index=True
        )

        df_comparacion_alternativos = pd.DataFrame({
            'Black-Litterman': bl_w[12:16]/bl_w[12:16].sum(),
            'Portafolio Modelo': portafolio_modelo[12:16]/portafolio_modelo[12:16].sum(),
            'Diferencia': (bl_w[12:16]/bl_w[12:16].sum()) - (portafolio_modelo[12:16]/portafolio_modelo[12:16].sum())
        }).reset_index().rename(columns={'index': 'Activo'})

        # 1. Calculamos las sumas para las columnas relevantes usando .loc para mayor claridad
        suma_bl = df_comparacion['Black-Litterman'].iloc[12:16].sum()
        suma_modelo = df_comparacion['Portafolio Modelo'].iloc[12:16].sum()

        # 2. Creamos la nueva fila como un DataFrame de 1xN
        #    Añadimos también la columna 'Activo' y 'Diferencia' para que coincida con el esquema
        total_alternativos_df = pd.DataFrame(
            {
                'Activo': ['Total Alternativos'],
                'Black-Litterman': [suma_bl],
                'Portafolio Modelo': [suma_modelo],
                'Diferencia': [suma_bl - suma_modelo]
            }
        )

        # 3. Concatenamos la nueva fila a la cabeza del DataFrame original
        df_comparacion_alternativos = pd.concat(
            [total_alternativos_df, df_comparacion_alternativos], 
            ignore_index=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Comparación de Ponderaciones")
            st.dataframe(
                df_comparacion.set_index('Activo').style.format({
                    'Black-Litterman': '{:.2%}',
                    'Portafolio Modelo': '{:.2%}',
                    'Diferencia': '{:.2%}'
                }),
                height=500 # Ajusta la altura de la tabla
            )

        with col2:
            st.write("#### Comparación de Ponderaciones Renta Variable")
            st.dataframe(
                df_comparacion_equities.set_index('Activo').style.format({
                    'Black-Litterman': '{:.2%}',
                    'Portafolio Modelo': '{:.2%}',
                    'Diferencia': '{:.2%}'
                }),
            )        

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Comparación de Ponderaciones Renta Fija")
            st.dataframe(
                df_comparacion_bonos.set_index('Activo').style.format({
                    'Black-Litterman': '{:.2%}',
                    'Portafolio Modelo': '{:.2%}',
                    'Diferencia': '{:.2%}'
                }),
            )

        with col2:
            st.write("#### Comparación de Ponderaciones Alternativos")
            st.dataframe(
                df_comparacion_alternativos.set_index('Activo').style.format({
                    'Black-Litterman': '{:.2%}',
                    'Portafolio Modelo': '{:.2%}',
                    'Diferencia': '{:.2%}'
                }),
            )     

        # --- NUEVA SECCIÓN: Gráfico de Riesgo vs Retorno por Activo ---
        st.divider()
        st.subheader("3. Análisis de Riesgo-Retorno por Activo (Posterior)")
        st.write("Visualización de los retornos esperados y la volatilidad para cada activo después del ajuste de Black-Litterman.")

        # 1. Preparar los datos 
        df_risk_return = pd.DataFrame({
            'Retorno': bl_mu,
            'Volatilidad': bl_S
        }).reset_index().rename(columns={'index': 'Activo'})
                           
        # 2. Crear y mostrar el gráfico en una columna
        col1, col2 = st.columns([2, 1]) # Damos más espacio al gráfico

        with col1:
            # --- LLAMADA A LA NUEVA Y CORRECTA FUNCIÓN ---
            fig1 = create_asset_risk_return_plot(
                df_risk_return, 
                'Volatilidad',
                'Retorno',
                'Retorno vs. Volatilidad por Activo'
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.write("#### Datos de los Activos:")
            # Muestra la tabla de datos para referencia
            st.dataframe(
                df_risk_return.set_index('Activo').style.format({
                    'Retorno': '{:.2%}',
                    'Volatilidad': '{:.2%}'
                }),
                height=500 # Ajusta la altura de la tabla
            )

        # 3.3: Resumen de las restricciones aplicadas
        st.divider()
        st.subheader("Restricciones Aplicadas a los modelos")
        st.markdown(
            "- Pesos no negativos; la suma de pesos es 100%\n"
            "- **Renta Variable** (primeros 6): min 25%, max 70%\n"
            "- **Renta Fija** (siguientes 6): min 25%, max 70%\n"
            "- **Commodities** (últimos 2): max 15%\n"
            "- **Máximo por activo individual**: 15%"
        )

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")