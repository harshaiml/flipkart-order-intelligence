import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, precision_score

preprocessor = joblib.load("models/_preprocessor_temp.pkl")
best_rf = joblib.load("models/_rf_temp.pkl")
X_train = pd.read_pickle("models/_X_train_temp.pkl")
X_test = pd.read_pickle("models/_X_test_temp.pkl")
y_train = pd.read_pickle("models/_y_train_temp.pkl")
y_test = pd.read_pickle("models/_y_test_temp.pkl")

print("=" * 60)
print("TASK 9: SAVE FINAL ARTIFACT")
print("=" * 60)

# Build ONE combined pipeline: preprocessing + tuned Random Forest
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", best_rf),
])
# Refit combined pipeline cleanly on train data (preprocessor already fitted the
# same way; refit end-to-end so the saved object is self-contained and consistent)
final_pipeline.fit(X_train, y_train)

# Re-run Task 5's threshold-sweep procedure, but on the RF's own predict_proba
test_proba_rf = final_pipeline.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9 + 0.001, 0.02)
results = []
for t in thresholds:
    pred_t = (test_proba_rf >= t).astype(int)
    f1_t = f1_score(y_test, pred_t, zero_division=0)
    recall_t = recall_score(y_test, pred_t, zero_division=0)
    precision_t = precision_score(y_test, pred_t, zero_division=0)
    results.append((t, f1_t, recall_t, precision_t))

results_df = pd.DataFrame(results, columns=["threshold", "f1", "recall", "precision"])
best_row = results_df.loc[results_df["f1"].idxmax()]
t_star_rf = round(float(best_row["threshold"]), 2)

print(f"\nRandom Forest's own F1-maximising threshold sweep (on predict_proba, test split):")
print(results_df.round(4).to_string(index=False))
print(f"\nt*_rf (F1-maximising threshold on RF's own predict_proba) = {t_star_rf}")
print(f"  F1 at t*_rf:        {best_row['f1']:.4f}")
print(f"  Recall at t*_rf:    {best_row['recall']:.4f}")
print(f"  Precision at t*_rf: {best_row['precision']:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(final_pipeline, "models/return_risk_model.pkl")

# Save t_star_rf alongside so Part 3's tool can load and use it
with open("models/t_star_rf.txt", "w") as f:
    f.write(str(t_star_rf))

print(f"\nSaved final pipeline to models/return_risk_model.pkl")
print(f"Saved t*_rf = {t_star_rf} to models/t_star_rf.txt")

# Verify it loads and works standalone
reloaded = joblib.load("models/return_risk_model.pkl")
sample = X_test.iloc[[0]]
sample_proba = reloaded.predict_proba(sample)[:, 1][0]
print(f"\nVerification: reloaded pipeline predicts proba={sample_proba:.4f} on a sample order.")

# Clean up temp files
for f in os.listdir("models"):
    if f.startswith("_") and f.endswith(("_temp.pkl", "_temp.npy", "_temp.csv")):
        pass  # keep for now, will clean at the end
