import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

preprocessor = joblib.load("models/_preprocessor_temp.pkl")
X_train = pd.read_pickle("models/_X_train_temp.pkl")
X_test = pd.read_pickle("models/_X_test_temp.pkl")
y_train = pd.read_pickle("models/_y_train_temp.pkl")
y_test = pd.read_pickle("models/_y_test_temp.pkl")

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print("=" * 60)
print("TASK 6: RANDOM FOREST + GridSearchCV")
print("=" * 60)

rf = RandomForestClassifier(class_weight="balanced", random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10, None],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(rf, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_train_proc, y_train)

print(f"\nBest params: {grid.best_params_}")
print(f"Best cross-validated ROC-AUC: {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
test_proba = best_rf.predict_proba(X_test_proc)[:, 1]
test_roc_auc = roc_auc_score(y_test, test_proba)
print(f"Held-out test-set ROC-AUC: {test_roc_auc:.4f}")
print(f"Difference (CV - test): {abs(grid.best_score_ - test_roc_auc):.4f}")

joblib.dump(best_rf, "models/_rf_temp.pkl")
np.save("models/_rf_proba_temp.npy", test_proba)
print("\nSaved tuned Random Forest and test probabilities.")
