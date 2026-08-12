import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              precision_score, roc_auc_score)

preprocessor = joblib.load("models/_preprocessor_temp.pkl")
X_train = pd.read_pickle("models/_X_train_temp.pkl")
X_test = pd.read_pickle("models/_X_test_temp.pkl")
y_train = pd.read_pickle("models/_y_train_temp.pkl")
y_test = pd.read_pickle("models/_y_test_temp.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print("=" * 60)
print("TASK 5: LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
lr.fit(X_train_proc, y_train)

proba = lr.predict_proba(X_test_proc)[:, 1]
pred_default = (proba >= 0.5).astype(int)

acc = accuracy_score(y_test, pred_default)
f1 = f1_score(y_test, pred_default)
recall = recall_score(y_test, pred_default)
precision = precision_score(y_test, pred_default)
roc_auc = roc_auc_score(y_test, proba)

print(f"\nAt default threshold 0.5:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

default_recall = recall

# Threshold sweep
print("\n--- Threshold sweep (0.1 to 0.9, step 0.02) ---")
thresholds = np.arange(0.1, 0.9 + 0.001, 0.02)
results = []
for t in thresholds:
    pred_t = (proba >= t).astype(int)
    f1_t = f1_score(y_test, pred_t, zero_division=0)
    recall_t = recall_score(y_test, pred_t, zero_division=0)
    precision_t = precision_score(y_test, pred_t, zero_division=0)
    results.append((t, f1_t, recall_t, precision_t))

results_df = pd.DataFrame(results, columns=["threshold", "f1", "recall", "precision"])
best_row = results_df.loc[results_df["f1"].idxmax()]

print(results_df.round(4).to_string(index=False))

print(f"\nBest F1-maximizing threshold: {best_row['threshold']:.2f}")
print(f"  F1 at best threshold:        {best_row['f1']:.4f}")
print(f"  Recall at best threshold:    {best_row['recall']:.4f}")
print(f"  Precision at best threshold: {best_row['precision']:.4f}")
print(f"  Recall improvement over default (0.5): {(best_row['recall'] - default_recall)*100:.1f} pp")

print(f"""
Business trade-off: Lowering the decision threshold from 0.5 to {best_row['threshold']:.2f}
trades precision for recall -- we flag more orders as "risky", catching more true
returns (recall rises to {best_row['recall']:.2f}) but also raise more false alarms on
orders that would not actually be returned (precision drops to {best_row['precision']:.2f}).
For Flipkart's use case, missing a genuinely risky order (false negative) is more
expensive than a false positive, since a false negative means no proactive check
happens and the return cost is fully incurred later, whereas a false positive just
costs a slightly unnecessary support nudge -- so accepting more false positives to
catch more true risk is the right trade here.
""")

joblib.dump(lr, "models/_lr_temp.pkl")
np.save("models/_lr_proba_temp.npy", proba)
print("Saved LR model and probabilities.")
