import numpy as np
import pandas as pd
from typing import List, Tuple, Any
from scipy.optimize import minimize_scalar

class RevenueOptimizer:
    """
    Optimizer for finding the sweet-spot price that maximizes predicted revenue,
    using scipy.optimize and the trained demand forecasting model.
    """
    
    def __init__(self, model: Any, feature_cols: List[str]):
        """
        Initialize the optimizer.
        
        Args:
            model: Trained model (e.g., XGBoost) that predicts demand quantity.
            feature_cols: The exact list of feature columns the model expects.
        """
        self.model = model
        self.feature_cols = feature_cols
        
    def _revenue_objective(self, price: float, base_row: pd.Series) -> float:
        """
        Objective function to minimize. Since we want to maximize Revenue (P * Q),
        we minimize -Revenue.
        """
        row = base_row.copy()
        
        # Update price-dependent features
        row["avg_price"] = price
        if "avg_competitor_price" in row and row["avg_competitor_price"] > 0:
            row["rel_price_vs_comp"] = price / row["avg_competitor_price"]
            row["avg_competitor_gap"] = price - row["avg_competitor_price"]
            
        X_row = pd.DataFrame([row[self.feature_cols]])
        
        q_pred = self.model.predict(X_row)[0]
        q_pred = max(float(q_pred), 0.0)
        
        revenue = price * q_pred
        return -revenue

    def optimize_price(self, base_row: pd.Series, min_price: float, max_price: float) -> Tuple[float, float, float]:
        """
        Finds the price within [min_price, max_price] that maximizes revenue.
        
        Args:
            base_row: A pandas Series containing all necessary features for the model.
            min_price: Lower bound for the price constraint.
            max_price: Upper bound for the price constraint.
            
        Returns:
            Tuple of (optimal_price, predicted_quantity_at_optimal, max_revenue)
        """
        result = minimize_scalar(
            self._revenue_objective,
            bounds=(min_price, max_price),
            args=(base_row,),
            method='bounded'
        )
        
        if result.success:
            optimal_price = result.x
            max_revenue = -result.fun
            optimal_quantity = max_revenue / optimal_price if optimal_price > 0 else 0.0
            return optimal_price, optimal_quantity, max_revenue
        else:
            raise ValueError(f"Optimization failed: {result.message}")
