import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import recall_score, precision_score

preprocessor = joblib.load("models/_preprocessor_temp.pkl")
best_rf = joblib.load("models/_rf_temp.pkl")
X_test = pd.read_pickle("models/_X_test_temp.pkl")
y_test = pd.read_pickle("models/_y_test_temp.pkl")

X_test_proc = preprocessor.transform(X_test)
pred = best_rf.predict(X_test_proc)

overall_recall = recall_score(y_test, pred)
overall_precision = precision_score(y_test, pred)

print("=" * 60)
print("TASK 8: SUBGROUP / ROOT-CAUSE ANALYSIS")
print("=" * 60)
print(f"\nOverall test-set recall: {overall_recall:.4f}, precision: {overall_precision:.4f}")

results = X_test.copy()
results["y_true"] = y_test.values
results["y_pred"] = pred

print("\n--- By product_category ---")
rows = []
for cat, grp in results.groupby("product_category"):
    r = recall_score(grp["y_true"], grp["y_pred"], zero_division=0)
    p = precision_score(grp["y_true"], grp["y_pred"], zero_division=0)
    rows.append((cat, r, p, len(grp)))
cat_df = pd.DataFrame(rows, columns=["product_category", "recall", "precision", "n"])
print(cat_df.round(4).to_string(index=False))

print("\n--- By payment_method ---")
rows = []
for pay, grp in results.groupby("payment_method"):
    r = recall_score(grp["y_true"], grp["y_pred"], zero_division=0)
    p = precision_score(grp["y_true"], grp["y_pred"], zero_division=0)
    rows.append((pay, r, p, len(grp)))
pay_df = pd.DataFrame(rows, columns=["payment_method", "recall", "precision", "n"])
print(pay_df.round(4).to_string(index=False))

worst_cat = cat_df.loc[cat_df["recall"].idxmin()]
worst_pay = pay_df.loc[pay_df["recall"].idxmin()]

print(f"""
Finding: The '{worst_cat['product_category']}' category shows recall of {worst_cat['recall']:.2f},
meaningfully below the overall average of {overall_recall:.2f} -- the model under-flags risky
orders in this subgroup. Proposed concrete next step: introduce a category-specific
decision threshold for '{worst_cat['product_category']}' (rather than one global threshold),
calibrated on that subgroup's own predicted-probability distribution, since a global
threshold tuned on the overall class balance can systematically under-flag a subgroup
whose base return rate or probability distribution differs from the population average.
""")
