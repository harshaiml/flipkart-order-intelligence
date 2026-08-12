import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("orders_dataset.csv")

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

# ---- Task 3: Preprocess without leakage ----
categorical_features = ["product_category", "payment_method"]
numeric_features = [c for c in X.columns if c not in categorical_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train return rate: {y_train.mean():.4f}, Test return rate: {y_test.mean():.4f}")

# Fit preprocessor on TRAIN ONLY, transform both
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
print(f"\nProcessed train shape: {X_train_proc.shape}")

# ---- Task 4: Baseline DummyClassifier ----
print("\n" + "=" * 60)
print("TASK 4: BASELINE (DummyClassifier, most_frequent)")
print("=" * 60)

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train_proc, y_train)
dummy_pred = dummy.predict(X_test_proc)

dummy_acc = accuracy_score(y_test, dummy_pred)
dummy_f1 = f1_score(y_test, dummy_pred, pos_label=1)

print(f"Baseline Accuracy: {dummy_acc:.4f}")
print(f"Baseline F1 (returned=1): {dummy_f1:.4f}")
print(f"""
Explanation: The baseline achieves {dummy_acc*100:.1f}% accuracy simply by always predicting
"not returned" (the majority class), yet its F1-score for the returned=1 class is
{dummy_f1:.2f}. This is the classic "high accuracy, zero recall" trap: because ~{(1-y.mean())*100:.0f}%
of orders are never returned, a model that ignores the minority class entirely still
looks accurate on paper, while being completely useless for the actual business goal
of catching risky orders before they are returned. This is why accuracy alone is a
misleading metric here, and why F1/recall/precision on the returned=1 class (and ROC-AUC)
are the honest evaluation metrics this task is built on.
""")

import joblib
import os
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/_preprocessor_temp.pkl")
X_train.to_pickle("models/_X_train_temp.pkl")
X_test.to_pickle("models/_X_test_temp.pkl")
y_train.to_pickle("models/_y_train_temp.pkl")
y_test.to_pickle("models/_y_test_temp.pkl")
print("Saved intermediate artifacts for next step.")
