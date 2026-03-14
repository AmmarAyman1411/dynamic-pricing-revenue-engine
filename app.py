"""
Streamlit dashboard: change price and see predicted quantity and revenue.
Run from repo root: streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from src.optimization import RevenueOptimizer
from src.explainability import DemandExplainer

# Paths relative to repo root
MODELS_DIR = Path("model")
MODEL_PATH = MODELS_DIR / "demand_model.joblib"
SCHEMA_PATH = MODELS_DIR / "model_input_schema.parquet"

# Must match notebook feature_cols
FEATURE_COLS = [
    "avg_price",
    "avg_competitor_price",
    "avg_competitor_gap",
    "rel_price_vs_comp",
    "customers",
    "weekday_days",
    "weekend_days",
    "holiday_days",
    "lag_units_1",
    "lag_units_2",
    "days_since_last_sale",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run the notebook save cell to create it.")
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_scenarios():
    if not SCHEMA_PATH.exists():
        st.error(f"Schema not found at {SCHEMA_PATH}. Run the notebook save cell to create it.")
        return None
    return pd.read_parquet(SCHEMA_PATH)


def predict_quantity_for_price(model, base_row: pd.Series, price: float) -> float:
    row = base_row.copy()
    row["avg_price"] = price
    if "avg_competitor_price" in row and row["avg_competitor_price"] > 0:
        row["rel_price_vs_comp"] = price / row["avg_competitor_price"]
        row["avg_competitor_gap"] = price - row["avg_competitor_price"]
    X_row = pd.DataFrame([row[FEATURE_COLS]])
    q_hat = model.predict(X_row)[0]
    return max(float(q_hat), 0.0)


def main():
    st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")
    st.title("Dynamic Pricing & Revenue Explorer")

    model = load_model()
    df = load_scenarios()
    if model is None or df is None:
        st.stop()

    # Scenario selector: by (product_id, date) or index
    df["_label"] = df["product_id"].astype(str) + " | " + pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    options = df["_label"].tolist()
    choice = st.selectbox("Select scenario (product | date)", options, index=0)
    idx = options.index(choice)
    base_row = df.iloc[idx]

    current_price = float(base_row["avg_price"])
    price_min = round(0.7 * current_price, 2)
    price_max = round(1.3 * current_price, 2)

    st.subheader("Set price and see predicted demand")
    price = st.slider(
        "Price",
        min_value=float(price_min),
        max_value=float(price_max),
        value=float(current_price),
        step=round((price_max - price_min) / 50, 2),
        format="%.2f",
    )

    q_pred = predict_quantity_for_price(model, base_row, price)
    revenue = price * q_pred

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Price", f"{price:.2f}")
    with col2:
        st.metric("Predicted quantity", f"{q_pred:.1f}")
    with col3:
        st.metric("Predicted revenue", f"{revenue:.2f}")

    # Recommend price using numerical optimization (Scipy)
    st.subheader("Revenue Optimization (Scipy-based)")
    optimizer = RevenueOptimizer(model, FEATURE_COLS)
    
    try:
        opt_price, opt_qty, opt_rev = optimizer.optimize_price(base_row, price_min, price_max)
        st.success(f"**Sweet spot price (Optimized):** {opt_price:.2f} → revenue **{opt_rev:.2f}**")
    except Exception as e:
        st.error(f"Optimization warning: {e}")

    st.subheader("Revenue Curve")
    n_grid = 40
    prices_grid = np.linspace(price_min, price_max, n_grid)
    revenues = [price_g * predict_quantity_for_price(model, base_row, price_g) for price_g in prices_grid]

    chart_df = pd.DataFrame({"Price": prices_grid, "Revenue": revenues})
    st.line_chart(chart_df.set_index("Price"))

    # Explainability with SHAP
    st.subheader("Explainable AI (SHAP Waterfall)")
    st.write("Understand the driving factors for the demand prediction at your currently selected price.")
    explainer = DemandExplainer(model, FEATURE_COLS)
    
    with st.spinner("Generating SHAP explanations..."):
        shap_explanation = explainer.explain_prediction(base_row, price)
        fig = explainer.plot_waterfall(shap_explanation)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
