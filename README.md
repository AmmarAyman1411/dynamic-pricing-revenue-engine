# Dynamic Pricing & Revenue Optimization Engine 📈

An End-to-End Dynamic Pricing and Revenue Optimization Engine built to maximize eCommerce revenue by analyzing demand elasticity, competitor pricing, and inventory levels.

This project goes beyond simple price prediction. It leverages an XGBoost forecasting model combined with `scipy.optimize` to find the exact "Sweet Spot" price that yields the highest revenue, and uses SHAP Explainable AI (XAI) to interpret the driving factors behind consumer demand.

## Features
- **Advanced EDA & Feature Engineering**: Extracts price elasticity, competitor price gaps, and lag features to mimic real-world e-commerce dynamics.
- **Demand Forecasting**: Employs an XGBoost regressor to accurately forecast unit sales based on environmental and competitive factors.
- **Revenue Optimization**: Uses `scipy.optimize.minimize_scalar` to search through price constraints and recommend the revenue-maximizing price point.
- **Explainable AI (XAI)**: Integrates SHAP (Shapley Additive Explanations) to provide business stakeholders with transparent waterfall plots detailing feature impact.
- **Interactive Business Dashboard**: A Streamlit web application allowing users to simulate price changes, view real-time revenue impacts, and analyze optimal pricing curves. 

## Technology Stack
* **Language**: Python (Pandas, NumPy)
* **Machine Learning**: XGBoost
* **Explainability**: SHAP
* **Optimization**: SciPy
* **Frontend**: Streamlit

## Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/AmmarAyman1411/dynamic-pricing-revenue-engine.git
cd dynamic-pricing-revenue-engine
python -m venv .venv

# On Windows:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Running the Dashboard
Make sure your virtual environment is activated, then run the Streamlit app:
```bash
streamlit run app.py
```
This will launch the interactive pricing dashboard in your default web browser.

## Project Structure
- `app.py`: The main Streamlit dashboard application.
- `src/optimization.py`: Contains the `RevenueOptimizer` tailored for `scipy` bounded scalar minimization.
- `src/explainability.py`: Contains the `DemandExplainer` for generating SHAP waterfall plots.
- `src/notebooks/`: Jupyter notebooks detailing the exploratory data analysis (EDA) and XGBoost model training.
- `model/`: Serialized models and input schemas.
- `data/`: Raw and processed datasets.

## About the Dataset
This project is structured around the Brazilian E-commerce Public Dataset by Olist, exploring real-world e-commerce behavior including seasonality, holiday spikes, and price sensitivity.
