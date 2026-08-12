import pandas as pd
import numpy as np
import joblib
from sklearn.inspection import permutation_importance

preprocessor = joblib.load("models/_preprocessor_temp.pkl")
best_rf = joblib.load("models/_rf_temp.pkl")
X_train = pd.read_pickle("models/_X_train_temp.pkl")
X_test = pd.read_pickle("models/_X_test_temp.pkl")
y_test = pd.read_pickle("models/_y_test_temp.pkl")

X_test_proc = preprocessor.transform(X_test)

# get feature names after preprocessing
numeric_features = preprocessor.transformers_[0][2]
cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = list(cat_encoder.get_feature_names_out(["product_category", "payment_method"]))
all_feature_names = list(numeric_features) + cat_feature_names

print("=" * 60)
print("TASK 7: FEATURE IMPORTANCE")
print("=" * 60)

# Impurity-based importance
importances = best_rf.feature_importances_
imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)
print("\n--- Top 5 impurity-based feature_importances_ ---")
print(imp_df.head(5).to_string(index=False))

top5_impurity = set(imp_df.head(5)["feature"])

print(f"""
Interpretation of top 5 (impurity-based):
These features plausibly drive return risk because they map directly onto the
data-generating logic: prior return behavior and payment method compound risk,
apparel/footwear carry inherent fit-related return risk, and price/discount level
affects how much is at stake for a customer to bother returning.
""")

# Permutation importance on TEST split
print("--- Computing permutation importance on test split ---")
perm_result = permutation_importance(
    best_rf, X_test_proc, y_test, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": all_feature_names,
    "perm_importance_mean": perm_result.importances_mean,
    "perm_importance_std": perm_result.importances_std,
}).sort_values("perm_importance_mean", ascending=False)

print("\n--- Top 5 permutation importance (test split) ---")
print(perm_df.head(5).to_string(index=False))

top5_perm = set(perm_df.head(5)["feature"])
lost_features = top5_impurity - top5_perm

print(f"""
Comparison: Of the original top-5 impurity-based features, the following lose most
of their importance under permutation: {lost_features if lost_features else 'none -- rankings agree closely'}.
Impurity-based .feature_importances_ can overrate a noisy continuous feature (like
delivery_distance_km or price_inr) because tree-splitting algorithms favor
high-cardinality continuous columns purely because they offer more possible split
points to reduce impurity, regardless of whether the feature carries real predictive
signal on unseen data -- permutation importance avoids this bias because it directly
measures the drop in real test-set ROC-AUC when a feature's values are shuffled.
""")

imp_df.to_csv("models/_impurity_importance_temp.csv", index=False)
perm_df.to_csv("models/_perm_importance_temp.csv", index=False)
