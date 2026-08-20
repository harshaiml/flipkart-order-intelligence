"""
Part 3 Tasks 3-4 -- Real tool implementations that load Part 1 and Part 2's
saved artifacts. Nothing here is a hardcoded stand-in.
"""
import json
import joblib
import pandas as pd
import os

# ---- Task 3: check_return_risk ----

_return_risk_pipeline = None
_t_star_rf = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_return_risk_model():
    global _return_risk_pipeline, _t_star_rf
    if _return_risk_pipeline is None:
        model_path = os.path.join(REPO_ROOT, "models", "return_risk_model.pkl")
        _return_risk_pipeline = joblib.load(model_path)
        t_star_path = os.path.join(REPO_ROOT, "models", "t_star_rf.txt")
        with open(t_star_path) as f:
            _t_star_rf = float(f.read().strip())
    return _return_risk_pipeline, _t_star_rf


def check_return_risk(order_features: dict) -> dict:
    """Loads Part 1's tuned Random Forest pipeline (models/return_risk_model.pkl)
    and returns the model's predicted return probability plus a risk bucket.

    Bucket cut points are anchored to t*_rf (the F1-maximising threshold
    computed on the Random Forest's own predict_proba in Part 1 Task 9), NOT
    fixed values -- since a fixed 0.3/0.6 split is not self-calibrating across
    different valid trained models.

    order_features must contain: product_category, price_inr, discount_pct,
    payment_method, customer_tenure_days, num_previous_orders,
    num_previous_returns, delivery_distance_km, delivery_days,
    is_weekend_order, rating_given (may be None/NaN).
    """
    pipeline, t_star_rf = _load_return_risk_model()

    required_cols = [
        "product_category", "price_inr", "discount_pct", "payment_method",
        "customer_tenure_days", "num_previous_orders", "num_previous_returns",
        "delivery_distance_km", "delivery_days", "is_weekend_order", "rating_given",
    ]
    row = {col: order_features.get(col) for col in required_cols}
    X = pd.DataFrame([row])

    proba = float(pipeline.predict_proba(X)[:, 1][0])

    if proba < t_star_rf:
        bucket = "Low"
    elif proba >= t_star_rf + 0.15:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "return_probability": round(proba, 4),
        "risk_bucket": bucket,
        "threshold_used": t_star_rf,
    }


# ---- Task 4: classify_product_image ----
# Reuses Part 2's common.py loading logic (frozen ResNet-18 backbone + saved head)
import sys
sys.path.insert(0, REPO_ROOT)
from src.part2_image_classifier.common import classify_product_image  # noqa: E402, F401
# classify_product_image(image_path: str) -> dict  (predicted_category, confidence)
# is imported directly and used as-is; it already loads models/product_classifier_head.pt
# and the frozen pretrained ResNet-18 backbone, and runs against real .png files.


if __name__ == "__main__":
    # Quick smoke test of check_return_risk (does not need internet / pretrained weights)
    sample_order = {
        "product_category": "Electronics", "price_inr": 25000, "discount_pct": 35,
        "payment_method": "COD", "customer_tenure_days": 40, "num_previous_orders": 2,
        "num_previous_returns": 1, "delivery_distance_km": 300, "delivery_days": 6,
        "is_weekend_order": 1, "rating_given": None,
    }
    result = check_return_risk(sample_order)
    print("check_return_risk result:", json.dumps(result, indent=2))
