import shap
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any

class DemandExplainer:
    """
    SHAP-based Explainable AI (XAI) to interpret the demand model predictions.
    """
    def __init__(self, model: Any, feature_cols: List[str]):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model (e.g., XGBoost).
            feature_cols: List of expected features.
        """
        self.model = model
        self.feature_cols = feature_cols
        # Use Explainer which handles various model types
        self.explainer = shap.Explainer(self.model)
        
    def explain_prediction(self, base_row: pd.Series, price: float) -> shap.Explanation:
        """
        Generates a SHAP explanation for a single prediction at a specific price.
        
        Args:
            base_row: Base features for the scenario.
            price: The price to evaluate at.
            
        Returns:
            SHAP Explanation object representing the breakdown at this price.
        """
        row = base_row.copy()
        row["avg_price"] = price
        if "avg_competitor_price" in row and row["avg_competitor_price"] > 0:
            row["rel_price_vs_comp"] = price / row["avg_competitor_price"]
            row["avg_competitor_gap"] = price - row["avg_competitor_price"]
            
        X_df = pd.DataFrame([row[self.feature_cols]])
        
        shap_values = self.explainer(X_df)
        return shap_values[0]
        
    def plot_waterfall(self, shap_explanation: shap.Explanation, max_display: int = 10) -> plt.Figure:
        """
        Helper method to plot a SHAP waterfall chart and return the matplotlib Figure.
        
        Args:
            shap_explanation: The SHAP Explanation object.
            max_display: Maximum number of features to show.
            
        Returns:
            A matplotlib Figure containing the waterfall plot.
        """
        fig = plt.figure(figsize=(8, 5))
        # shap plots modify the current matplotlib figure, so we ensure one is active
        shap.plots.waterfall(shap_explanation, max_display=max_display, show=False)
        plt.tight_layout()
        return fig
