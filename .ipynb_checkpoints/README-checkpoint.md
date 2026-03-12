## Dynamic Pricing & Revenue Optimization Engine

This repository implements an end-to-end **dynamic pricing and revenue optimization engine** for retail use cases.  
It is designed as a Tier-1 portfolio project for a data scientist, combining:

- **Demand forecasting** (quantity sold prediction)
- **Price optimization** (maximizing \( R(p) = p \cdot \hat{Q}(p) \))
- **Explainable AI (XAI)** using SHAP
- **Business-facing dashboard** using Streamlit

### Project Structure

- `config/` – Global configuration and paths.
- `data/`
  - `raw/` – Original Retail Price Optimization CSV files (read-only).
  - `interim/` – Cleaned / partially processed data.
  - `processed/` – Final feature tables for modeling and optimization.
- `notebooks/`
  - `01_eda.ipynb` – Exploratory data analysis and data understanding.
  - `02_feature_prototyping.ipynb` – Rapid feature engineering experiments.
- `src/`
  - `data/` – Loading and preprocessing logic.
  - `features/` – Feature engineering utilities (elasticity, temporal, etc.).
  - `models/` – Demand forecasting models and evaluation.
  - `optimization/` – Price optimization routines.
  - `explainability/` – SHAP explainability helpers.
  - `app/` – Streamlit dashboard code.
  - `tracking/` – MLflow tracking helpers.
- `scripts/` – CLI entry points for running EDA, training, and optimization workflows.

### Dataset

This project assumes a **Retail Price Optimization Dataset** with transaction-level records, including columns such as:

- `date`
- `product_id`
- `store_id`
- `price`
- `units_sold`
- `promo_flag`
- `competitor_price`

You should place the downloaded CSV file(s) into:

- `data/raw/retail_price_optimization.csv` (or a similar, clearly named file).

The exact file name and column mapping can be configured in `config/config.yaml`.

### Getting Started

1. **Create and activate a virtual environment** (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your dataset into `data/raw/`.
4. Run the EDA notebook `notebooks/01_eda.ipynb` to explore and validate the data.

### Roadmap

1. Phase 1 – Advanced EDA & feature engineering (elasticity, competitor gap, recency).
2. Phase 2 – Demand forecasting model (XGBoost / LightGBM).
3. Phase 3 – Price optimization (maximize revenue given predicted demand).
4. Phase 4 – Explainability with SHAP.
5. Phase 5 – Streamlit dashboard for interactive pricing simulations.

