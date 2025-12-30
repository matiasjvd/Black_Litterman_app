# streamlit_bl_frontier_app.py
# Run with:
#   streamlit run streamlit_bl_frontier_app.py
#
# Expects an Excel (.xlsx) with sheets:
#   - "Retornos": price index levels (Date index in first column)
#   - "Pesos": portfolio model weights (either 2 cols [asset, weight] or index+1col)

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.optimize import minimize
from pypfopt import BlackLittermanModel

warnings.filterwarnings("ignore")

# -----------------------------
# 1) Asset universe (pretty names)
# -----------------------------
ORIGINAL_TO_RENAMED = {
    "SPX Index": "USA",
    "MXEUG Index": "Europa equities",
    "UKX Index": "UK",
    "MXJP Index": "Japon",
    "MXAPJ Index": "Asia",
    "MXLA Index": "Latam",
    "LF98TRUU Index": "US HY",
    "LUACTRUU Index": "US IG",
    "LBEATRUH Index": "Europa bonds",
    "BSELTRUU Index": "Latam corp",
    "BSSUTRUU Index": "Emerging sov",
    "CABS Index": "ABS",
    "BCOMTR Index": "Commodities",
    "GLD US EQUITY": "Oro",
    "Private Debt": "Private Debt",
    "Private Equity": "Private Equity",
    "MXWD Index": "World equities",
}

MODEL_ASSETS = [
    "USA", "Europa equities", "UK", "Japon", "Asia", "Latam",
    "US HY", "US IG", "Europa bonds", "Latam corp", "Emerging sov", "ABS",
    "Commodities", "Oro", "Private Debt", "Private Equity",
]

DEFAULT_PRIORS = {
    "USA": 0.055, "Europa equities": 0.083, "UK": 0.067, "Japon": 0.060,
    "Asia": 0.080, "Latam": 0.066, "US HY": 0.052, "US IG": 0.045,
    "Europa bonds": 0.024, "Latam corp": 0.074, "Emerging sov": 0.071,
    "ABS": 0.050, "Commodities": 0.045, "Oro": 0.090,
    "Private Debt": 0.085, "Private Equity": 0.080,
}

DEFAULT_VIEWS = {
    "USA": 0.123, "Europa equities": 0.102, "UK": 0.132, "Japon": 0.075,
    "Asia": 0.18, "Latam": 0.158, "US HY": 0.0590, "US IG": 0.0470,
    "Europa bonds": 0.0280, "Latam corp": 0.0550, "Emerging sov": 0.0480,
    "ABS": 0.0450, "Commodities": 0.0500, "Oro": 0.0650,
    "Private Debt": 0.0700, "Private Equity": 0.0800,
}

DEFAULT_CONFIDENCES = {
    "USA": 0.5, "Europa equities": 0.5, "UK": 0.5, "Japon": 0.5,
    "Asia": 0.5, "Latam": 0.5, "US HY": 0.8, "US IG": 0.8,
    "Europa bonds": 0.8, "Latam corp": 0.8, "Emerging sov": 0.8,
    "ABS": 0.8, "Commodities": 0.8, "Oro": 0.8,
    "Private Debt": 0.8, "Private Equity": 0.8,
}

# -----------------------------
# 2) Optimization helpers (SLSQP)
# -----------------------------
RISK_FREE_DEFAULT = 0.04


def portfolio_performance(w, mu, cov):
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol


def neg_sharpe(w, mu, cov, rf):
    ret, vol = portfolio_performance(w, mu, cov)
    if vol <= 0:
        return 1e9
    return -(ret - rf) / vol


def min_variance_obj(w, mu, cov):
    _, vol = portfolio_performance(w, mu, cov)
    return vol


def max_return_obj(w, mu, cov):
    return -float(np.dot(w, mu))


def build_additive_constraints(w_model, max_dev=0.05):
    """
    Restricción banda aditiva vs modelo: w_i ∈ [w_model - max_dev, w_model + max_dev]
    """
    # Aseguramos que w_model sea un array de numpy para indexación posicional segura
    w_arr = np.array(w_model, dtype=float)
    n = len(w_arr)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    cons.append({"type": "ineq", "fun": lambda w: w})  # w >= 0

    for i in range(n):
        lb = max(float(w_arr[i] - max_dev), 0.0)
        ub = min(float(w_arr[i] + max_dev), 1.0)
        # Usamos default values en el lambda para capturar el valor actual de i, lb, ub
        cons.append({"type": "ineq", "fun": lambda w, lb=lb, i=i: w[i] - lb})
        cons.append({"type": "ineq", "fun": lambda w, ub=ub, i=i: ub - w[i]})
    return cons


def make_feasible_start(w_model):
    """
    Punto inicial razonable y normalmente factible: el mismo portafolio modelo normalizado.
    """
    w0 = np.array(w_model, dtype=float).copy()
    s = float(w0.sum())
    if s <= 0:
        w0 = np.ones_like(w0) / len(w0)
    else:
        w0 = w0 / s
    return w0


def override_vol_preserve_corr(cov_df, asset_name, new_vol):
    """
    Ajusta volatilidad de 'asset_name' preservando correlaciones:
    - Escala fila/columna por k = new_vol / old_vol
    - Var nueva queda new_vol^2
    """
    cov = cov_df.copy()
    if asset_name not in cov.index:
        return cov

    i = cov.index.get_loc(asset_name)
    old_var = float(cov.iloc[i, i])
    old_vol = float(np.sqrt(max(old_var, 0.0)))

    if old_vol <= 0:
        cov.iloc[i, :] = 0.0
        cov.iloc[:, i] = 0.0
        cov.iloc[i, i] = float(new_vol**2)
        return cov

    k = float(new_vol / old_vol)
    cov.iloc[i, :] = cov.iloc[i, :] * k
    cov.iloc[:, i] = cov.iloc[:, i] * k
    cov.iloc[i, i] = float(new_vol**2)
    return cov


def run_black_litterman_posterior(mu_prior, cov_prior, views_dict, conf_dict):
    """
    mu_prior: pd.Series (annualized), index=MODEL_ASSETS
    cov_prior: pd.DataFrame (annualized), index/cols=MODEL_ASSETS
    """
    views = pd.Series(views_dict).reindex(mu_prior.index).astype(float)
    conf = pd.Series(conf_dict).reindex(mu_prior.index).astype(float).values

    bl = BlackLittermanModel(
        cov_prior,
        pi=mu_prior,
        absolute_views=views,
        omega="idzorek",
        view_confidences=conf,
    )
    return bl.bl_returns(), bl.bl_cov()


def optimize_portfolios(mu_s, cov_df, w_model, max_dev=0.05, rf=RISK_FREE_DEFAULT):
    mu = mu_s.values.astype(float)
    cov = cov_df.values.astype(float)
    n = len(mu)

    cons = build_additive_constraints(w_model, max_dev=max_dev)
    bounds = [(0.0, 1.0)] * n
    w0 = make_feasible_start(w_model)

    res_sh = minimize(
        neg_sharpe, w0,
        args=(mu, cov, float(rf)),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 4000, "ftol": 1e-12},
    )
    if not res_sh.success:
        raise RuntimeError(f"MaxSharpe failed: {res_sh.message}")

    res_mv = minimize(
        min_variance_obj, w0,
        args=(mu, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 4000, "ftol": 1e-12},
    )
    if not res_mv.success:
        raise RuntimeError(f"MinVar failed: {res_mv.message}")

    w_sh = pd.Series(res_sh.x, index=mu_s.index)
    w_mv = pd.Series(res_mv.x, index=mu_s.index)

    r_sh, v_sh = portfolio_performance(res_sh.x, mu, cov)
    r_mv, v_mv = portfolio_performance(res_mv.x, mu, cov)

    return w_sh, (r_sh, v_sh), w_mv, (r_mv, v_mv)


def feasible_extremes(mu_s, cov_df, w_model, max_dev=0.05):
    """
    Encuentra extremos factibles bajo las MISMAS restricciones:
      - Min Var
      - Max Return
    Esto define el rango real para targets de frontera eficiente.
    """
    mu = mu_s.values.astype(float)
    cov = cov_df.values.astype(float)
    n = len(mu)

    cons = build_additive_constraints(w_model, max_dev=max_dev)
    bounds = [(0.0, 1.0)] * n
    
    # Probamos dos puntos de inicio para mayor robustez
    starts = [make_feasible_start(w_model), np.ones(n)/n]
    
    best_res_mv = None
    best_res_mr = None

    for w0 in starts:
        res_mv = minimize(
            min_variance_obj, w0,
            args=(mu, cov),
            method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        if res_mv.success:
            if best_res_mv is None or res_mv.fun < best_res_mv.fun:
                best_res_mv = res_mv
        
        res_mr = minimize(
            max_return_obj, w0,
            args=(mu, cov),
            method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        if res_mr.success:
            if best_res_mr is None or res_mr.fun < best_res_mr.fun:
                best_res_mr = res_mr

    if best_res_mv is None or best_res_mr is None:
        # Fallback si falla la optimización con restricciones
        r_avg = np.dot(make_feasible_start(w_model), mu)
        v_avg = np.sqrt(make_feasible_start(w_model) @ cov @ make_feasible_start(w_model))
        return (make_feasible_start(w_model), (r_avg, v_avg)), (make_feasible_start(w_model), (r_avg, v_avg))

    r_mv, v_mv = portfolio_performance(best_res_mv.x, mu, cov)
    r_mr, v_mr = portfolio_performance(best_res_mr.x, mu, cov)

    return (best_res_mv.x, (r_mv, v_mv)), (best_res_mr.x, (r_mr, v_mr))


def efficient_frontier(mu_s, cov_df, w_model, max_dev=0.05, n_points=100, constrained=True):
    """
    Frontera eficiente:
      - Si constrained=True: usa bandas aditivas vs modelo.
      - Si constrained=False: solo suma(w)=1 y w>=0 (teórica clásica),
        cubriendo desde el min(mu) hasta max(mu) para visualización completa.
    """
    mu = mu_s.values.astype(float)
    cov = cov_df.values.astype(float)
    n = len(mu)
    bounds = [(0.0, 1.0)] * n
    
    if constrained:
        base_cons = build_additive_constraints(w_model, max_dev=max_dev)
        (w_mv, (r_mv, v_mv)), (w_mr, (r_mr, v_mr)) = feasible_extremes(
            mu_s, cov_df, w_model, max_dev=max_dev
        )
        # Expandimos ligeramente el rango de targets para asegurar que cubrimos los extremos
        r_start, r_end = r_mv, r_mr
        w_init = w_mv
    else:
        # Unconstrained: abarcamos todo el espectro de retornos de los activos
        base_cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        r_start, r_end = np.min(mu), np.max(mu)
        w_init = np.ones(n) / n

    if abs(r_end - r_start) < 1e-8:
        targets = np.array([r_start, r_end])
    else:
        targets = np.linspace(r_start, r_end, n_points)

    vols = []
    w_prev = w_init.copy() 

    for t in targets:
        cons = list(base_cons)
        cons.append({"type": "eq", "fun": lambda w, t=t, mu=mu: np.dot(w, mu) - t})

        res = minimize(
            min_variance_obj, w_prev,
            args=(mu, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        if res.success:
            vols.append(res.fun)
            w_prev = res.x
        else:
            # Re-intento con start neutro o modelo si falla el warm start
            for w0 in [w_init, make_feasible_start(w_model)]:
                res_retry = minimize(
                    min_variance_obj, w0,
                    args=(mu, cov),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=cons,
                    options={"maxiter": 2000, "ftol": 1e-10},
                )
                if res_retry.success:
                    vols.append(res_retry.fun)
                    w_prev = res_retry.x
                    break
            else:
                vols.append(np.nan)

    return targets, np.array(vols, dtype=float)


def plot_frontier(
    assets,
    mu_bl,
    cov_bl,
    targets_bl,
    vols_bl,
    targets_unconstrained,
    vols_unconstrained,
    pt_sh_bl,
    pt_mv_bl,
    pt_sh_unconstrained,
):
    vol_assets_bl = np.sqrt(np.diag(cov_bl.values))

    fig = plt.figure(figsize=(14, 9))

    # Assets scatter (solo BL posterior)
    plt.scatter(
        vol_assets_bl * 100, 
        mu_bl.values * 100, 
        s=200, alpha=0.8, c='steelblue',
        marker='o', label='Activos (Posterior BL)',
        edgecolors='navy', linewidth=1.5
    )

    for i, asset in enumerate(assets):
        plt.annotate(
            asset,
            (float(vol_assets_bl[i]) * 100, float(mu_bl.loc[asset]) * 100),
            fontsize=8, ha='right'
        )

    # Frontiers
    # 1) Unconstrained (Reference)
    mu_inf = np.isfinite(vols_unconstrained)
    if mu_inf.any():
        plt.plot(
            vols_unconstrained[mu_inf] * 100, targets_unconstrained[mu_inf] * 100, 
            color='gray', linestyle=':', alpha=0.6, linewidth=2.0, label="Frontera Teórica (sin bandas)"
        )

    # 2) BL Constrained
    m2 = np.isfinite(vols_bl)
    if m2.any():
        plt.plot(
            vols_bl[m2] * 100, targets_bl[m2] * 100, 
            linestyle='--', color='royalblue', linewidth=2.5, label='Frontera Eficiente (Black-Litterman)', alpha=0.9
        )

    # Optimal points
    plt.scatter(pt_sh_bl[1] * 100, pt_sh_bl[0] * 100, marker='*', color='purple', s=600,
                label='Máximo Sharpe (BL)', edgecolors='darkviolet', linewidth=1.5, zorder=5)

    if pt_sh_unconstrained is not None:
        plt.scatter(pt_sh_unconstrained[1] * 100, pt_sh_unconstrained[0] * 100, marker='*', color='gold', s=700,
                    label='Máximo Sharpe Teórico (Sin Bandas)', edgecolors='black', linewidth=1.5, zorder=6)

    plt.scatter(pt_mv_bl[1] * 100, pt_mv_bl[0] * 100, marker='D', color='cyan', s=120,
                label='Mínima Varianza (BL)', edgecolors='darkcyan', linewidth=1.5)

    plt.xlabel('Volatilidad Anual (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Retorno Anual (%)', fontsize=13, fontweight='bold')
    plt.title('Frontera Eficiente con Restricciones vs Portafolio Modelo', fontsize=15, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig


def plot_weights_comparison(df_comp):
    """
    Gráfico de barras comparativo de pesos: Modelo vs Black-Litterman.
    """
    assets = df_comp.index.tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=assets,
        y=df_comp["Modelo"] * 100,
        name="Portafolio Modelo",
        marker_color="lightslategrey"
    ))
    
    fig.add_trace(go.Bar(
        x=assets,
        y=df_comp["BL (MaxSharpe)"] * 100,
        name="Portafolio Black-Litterman",
        marker_color="crimson"
    ))
    
    fig.update_layout(
        title="Comparación de Pesos: Modelo vs Black-Litterman (Max Sharpe)",
        xaxis_title="Activos",
        yaxis_title="Peso (%)",
        barmode="group",
        legend=dict(x=0, y=1.1, orientation="h"),
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode="x unified"
    )
    
    return fig


# -----------------------------
# 3) Data loading (Excel: 'Retornos' + 'Pesos')
# -----------------------------
def load_data_from_excel(file_obj):
    xls = pd.ExcelFile(file_obj)

    # Prices
    df_prices = xls.parse(sheet_name="Retornos", index_col=0, parse_dates=True)
    df_prices = df_prices.sort_index()
    
    # Rename columns first so we can filter by MODEL_ASSETS
    df_prices.columns = [ORIGINAL_TO_RENAMED.get(c, c) for c in df_prices.columns]
    
    # Now filter by MODEL_ASSETS
    df_prices = df_prices[[c for c in df_prices.columns if c in MODEL_ASSETS]]
    df_prices = df_prices.dropna(how="all")

    returns = df_prices.pct_change().dropna(how="all")
    returns_modelos = returns.reindex(columns=MODEL_ASSETS).dropna(how="all")

    # Weights
    try:
        w_raw = xls.parse(sheet_name="Pesos", header=None)
        if w_raw.shape[1] >= 2:
            w_series = pd.Series(w_raw.iloc[:, 1].values, index=w_raw.iloc[:, 0].values)
        else:
            w_series = xls.parse(sheet_name="Pesos", index_col=0, header=None).squeeze("columns")
    except Exception:
        w_series = pd.Series({a: 0.0 for a in MODEL_ASSETS})

    w_series.index = [ORIGINAL_TO_RENAMED.get(x, x) for x in w_series.index]
    w_series = w_series.reindex(MODEL_ASSETS).fillna(0.0).astype(float)

    # Normalize if not exactly 1
    s = float(w_series.sum())
    if s > 0 and abs(s - 1.0) > 1e-6:
        w_series = w_series / s

    return df_prices, returns_modelos, w_series


def band_feasibility_check(w_model, max_dev=0.05):
    """
    Quick check: sum(LB) <= 1 <= sum(UB) for active positions.
    """
    w = np.array(w_model, dtype=float)
    lb = np.maximum(w - max_dev, 0.0)
    ub = np.minimum(w + max_dev, 1.0)
    return float(lb.sum()), float(ub.sum())


# -----------------------------
# 4) Streamlit UI
# -----------------------------
st.set_page_config(page_title="BL + Efficient Frontier (Band vs Model)", layout="wide")
st.title("Black-Litterman + Frontera Eficiente (restricción vs Portafolio Modelo)")

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Sube Excel (.xlsx) con sheets: Retornos y Pesos", type=["xlsx"])

st.sidebar.header("Tipo de Prior (π)")
prior_type = st.sidebar.radio(
    "Selecciona el tipo de prior:",
    ["Prior histórico", "Prior de equilibrio (π = δ Σ w_modelo)"]
)

delta = 2.5
if prior_type == "Prior de equilibrio (π = δ Σ w_modelo)":
    delta = st.sidebar.number_input("Delta (aversión al riesgo δ)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)

st.sidebar.header("Restricción vs Modelo (Aditiva)")
max_deviation = st.sidebar.slider("Desviación máxima (ej. 0.05 = +/-5%)", 0.0, 1.0, 0.05, 0.01)
rf = st.sidebar.number_input("Risk-free (anual)", min_value=0.0, max_value=0.20, value=RISK_FREE_DEFAULT, step=0.005)

st.sidebar.header("Volatilidad manual Private Debt")
override_pd = st.sidebar.checkbox("Override vol Private Debt", value=False)
pd_vol = st.sidebar.number_input("Vol anual Private Debt (ej 0.12 = 12%)", min_value=0.0, value=0.12, step=0.01)

st.sidebar.header("Frontera")
n_points = st.sidebar.slider("Puntos de la frontera", 20, 500, 100, 10)

# Logic to load data and calculate priors reactively
mu_prior_calc = pd.Series({a: DEFAULT_PRIORS.get(a, 0.0) for a in MODEL_ASSETS})
cov_prior_calc = None
w_model_calc = None

if uploaded:
    try:
        _, returns_modelos, w_model_series = load_data_from_excel(uploaded)
        w_model_calc = w_model_series
        cov_prior_calc = returns_modelos.cov() * 365
        
        if override_pd:
            cov_prior_calc = override_vol_preserve_corr(cov_prior_calc, "Private Debt", float(pd_vol))
        
        if prior_type == "Prior histórico":
            mu_prior_calc = returns_modelos.mean() * 365
        else:
            # Implied equilibrium
            w_vals = w_model_series.reindex(MODEL_ASSETS).fillna(0).values.astype(float)
            pi_eq = delta * (cov_prior_calc.values @ w_vals)
            mu_prior_calc = pd.Series(pi_eq, index=MODEL_ASSETS)
            
        mu_prior_calc = mu_prior_calc.reindex(MODEL_ASSETS).fillna(0.0)
    except Exception as e:
        st.error(f"Error cargando datos para priors: {e}")

st.subheader("Priors, Views y Confidences (edita en %)")

# Prepare DataFrame for editor
param_df = pd.DataFrame({
    "asset": MODEL_ASSETS,
    "prior": [mu_prior_calc.get(a, 0.0) for a in MODEL_ASSETS],
    "view": [DEFAULT_VIEWS.get(a, 0.0) for a in MODEL_ASSETS],
    "confidence": [DEFAULT_CONFIDENCES.get(a, 1.0) for a in MODEL_ASSETS],
})
param_df_display = param_df.copy()
param_df_display["prior"] *= 100
param_df_display["view"] *= 100
param_df_display["confidence"] *= 100

# Equilibrium prior should be read-only
prior_disabled = (prior_type == "Prior de equilibrio (π = δ Σ w_modelo)")

edited_display = st.data_editor(
    param_df_display,
    num_rows="fixed",
    column_config={
        "asset": st.column_config.TextColumn("Activo", disabled=True),
        "prior": st.column_config.NumberColumn("Prior (π) %", format="%.2f%%", step=0.1, disabled=prior_disabled),
        "view": st.column_config.NumberColumn("View (Q) %", format="%.2f%%", step=0.1),
        "confidence": st.column_config.NumberColumn("Confianza %", format="%.0f", step=5.0, min_value=0.0, max_value=100.0),
    },
    use_container_width=True
)

edited = edited_display.copy()
edited["prior"] = edited["prior"] / 100.0
edited["view"] = edited["view"] / 100.0
edited["confidence"] = edited["confidence"] / 100.0

priors_dict = dict(zip(edited["asset"], edited["prior"]))
views_dict = dict(zip(edited["asset"], edited["view"]))
confidences_dict = dict(zip(edited["asset"], edited["confidence"]))

run_btn = st.button("Ejecutar Optimización")

if run_btn:
    if uploaded is None:
        st.error("Sube un Excel con sheets 'Retornos' y 'Pesos'.")
        st.stop()

    try:
        df_prices, returns_modelos, w_model_series = load_data_from_excel(uploaded)
        st.success("Datos procesados correctamente.")

        # Final mu_prior from table
        mu_prior = pd.Series(priors_dict).reindex(MODEL_ASSETS)
        cov_prior = returns_modelos.cov() * 365
        if override_pd:
            cov_prior = override_vol_preserve_corr(cov_prior, "Private Debt", float(pd_vol))

        # BL posterior
        mu_bl, cov_bl = run_black_litterman_posterior(mu_prior, cov_prior, views_dict, confidences_dict)
        
        if override_pd:
            cov_bl = override_vol_preserve_corr(cov_bl, "Private Debt", float(pd_vol))

        # Comparison table (Prior vs View vs Posterior)
        st.subheader("Comparación de Retornos Esperados (%)")
        df_ret_comp = pd.DataFrame({
            "Prior (π)": mu_prior,
            "View (Q)": pd.Series(views_dict).reindex(MODEL_ASSETS),
            "Posterior (BL)": mu_bl
        })
        st.dataframe((df_ret_comp * 100).style.format("{:.2f}%"), use_container_width=True)

        # Optimize
        w_model = w_model_series.values.astype(float)
        lb_sum, ub_sum = band_feasibility_check(w_model, max_deviation)
        st.info(f"Chequeo factibilidad banda: sum(LB)={lb_sum:.3f} | sum(UB)={ub_sum:.3f}")
        
        w_sh_prior, pt_sh_prior, w_mv_prior, pt_mv_prior = optimize_portfolios(
            mu_prior, cov_prior, w_model, max_dev=max_deviation, rf=float(rf)
        )
        w_sh_bl, pt_sh_bl, w_mv_bl, pt_mv_bl = optimize_portfolios(
            mu_bl, cov_bl, w_model, max_dev=max_deviation, rf=float(rf)
        )

        # Frontiers
        targets_prior, vols_prior = efficient_frontier(
            mu_prior, cov_prior, w_model_series, max_dev=max_deviation, n_points=int(n_points), constrained=True
        )
        targets_bl, vols_bl = efficient_frontier(
            mu_bl, cov_bl, w_model_series, max_dev=max_deviation, n_points=int(n_points), constrained=True
        )
        targets_un, vols_un = efficient_frontier(
            mu_bl, cov_bl, w_model_series, n_points=int(n_points), constrained=False
        )

        # Theoretical Max Sharpe
        res_sh_un = minimize(
            neg_sharpe, np.ones(len(mu_bl))/len(mu_bl),
            args=(mu_bl.values, cov_bl.values, float(rf)),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(mu_bl),
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 4000, "ftol": 1e-12},
        )
        pt_sh_un = (portfolio_performance(res_sh_un.x, mu_bl.values, cov_bl.values)) if res_sh_un.success else None

        # Results summary
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Prior (Hist/Eq)")
            st.metric("Max Sharpe Ret", f"{pt_sh_prior[0]*100:.2f}%")
            st.metric("Max Sharpe Vol", f"{pt_sh_prior[1]*100:.2f}%")
        with c2:
            st.subheader("Black-Litterman")
            st.metric("Max Sharpe Ret", f"{pt_sh_bl[0]*100:.2f}%")
            st.metric("Max Sharpe Vol", f"{pt_sh_bl[1]*100:.2f}%")
        with c3:
            st.subheader("Parámetros")
            st.write(f"Prior: {prior_type}")
            if prior_type.startswith("Prior de equil"): st.write(f"Delta: {delta}")
            st.write(f"Banda: +/- {max_deviation*100:.1f} pp")

        # Weights comparison
        st.header("Pesos (Max Sharpe) vs Modelo")
        df_comp = pd.DataFrame({
            "Modelo": w_model_series,
            "Prior (MaxSharpe)": w_sh_prior,
            "BL (MaxSharpe)": w_sh_bl,
        })
        if res_sh_un.success:
            df_comp["BL Teórica (Sin Bandas)"] = pd.Series(res_sh_un.x, index=mu_bl.index)
        
        df_comp["Dif BL vs Modelo"] = df_comp["BL (MaxSharpe)"] - df_comp["Modelo"]
        st.dataframe(df_comp.style.format("{:.2%}"), use_container_width=True)

        # Bar chart for weights
        fig_weights = plot_weights_comparison(df_comp)
        st.plotly_chart(fig_weights, use_container_width=True)

        # Frontier Plot
        st.header("Frontera Eficiente")
        fig = plot_frontier(
            MODEL_ASSETS, mu_bl, cov_bl,
            targets_bl, vols_bl,
            targets_un, vols_un,
            pt_sh_bl, pt_mv_bl, pt_sh_un
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error en la ejecución: {e}")
        st.exception(e)
