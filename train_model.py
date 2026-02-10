# -------------------------------------------------------------
# FULL WORKING TRAINING SCRIPT (NO PATH ERRORS)
# -------------------------------------------------------------

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

print("🔍 Searching for CSV file...")

# -------------------------------------------------------------
# AUTO-FIND ANY CSV IN THIS FOLDER
# -------------------------------------------------------------
csv_files = [f for f in os.listdir() if f.endswith(".csv")]

if len(csv_files) == 0:
    raise FileNotFoundError("❌ No CSV file found! Place your dataset CSV in this folder.")

if len(csv_files) > 1:
    print("⚠ Multiple CSV files found, using:", csv_files[0])

DATA_PATH = csv_files[0]
print("📌 Using CSV file:", DATA_PATH)

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("✅ Loaded data:", df.shape)

# -------------------------------------------------------------
# DETECT TARGET COLUMN AUTOMATICALLY
# -------------------------------------------------------------
possible_targets = ["loan_status", "Loan_Status", "status", "target", "TARGET"]

TARGET = None
for col in df.columns:
    if col in possible_targets:
        TARGET = col
        break

if TARGET is None:
    raise ValueError(
        "❌ Could not find target column!\n"
        "Your CSV must contain one of these columns:\n"
        "loan_status, Loan_Status, status, target"
    )

print("🎯 Target column detected as:", TARGET)

# -------------------------------------------------------------
# SPLIT FEATURES / TARGET
# -------------------------------------------------------------
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# -------------------------------------------------------------
# HANDLE MISSING VALUES
# -------------------------------------------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

for c in num_cols:
    X[c] = X[c].fillna(X[c].median())

for c in cat_cols:
    X[c] = X[c].fillna(X[c].mode().iloc[0])

# -------------------------------------------------------------
# LABEL ENCODERS
# -------------------------------------------------------------
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# -------------------------------------------------------------
# SCALE NUMERIC COLUMNS
# -------------------------------------------------------------
scaler = StandardScaler()
if len(num_cols) > 0:
    X[num_cols] = scaler.fit_transform(X[num_cols])

# -------------------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------------------------------------
# MODELS
# -------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}

print("\n📊 Training Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)

    results[name] = (model, f1)
    print(f"{name}: F1={f1:.4f}, ACC={acc:.4f}, PREC={prec:.4f}, REC={rec:.4f}")

# -------------------------------------------------------------
# PICK BEST MODEL
# -------------------------------------------------------------
best_model_name = max(results, key=lambda k: results[k][1])
best_model = results[best_model_name][0]

print("\n🏆 Best Model:", best_model_name)

# -------------------------------------------------------------
# SAVE ARTIFACTS
# -------------------------------------------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/best_loan_model.pkl")
joblib.dump(scaler, "model/loan_scaler.pkl")
joblib.dump(label_encoders, "model/loan_label_encoders.pkl")

print("\n🎉 Training complete!")
print("📁 Saved files:")
print(" - model/best_loan_model.pkl")
print(" - model/loan_scaler.pkl")
print(" - model/loan_label_encoders.pkl")
print("\nNow run:  streamlit run app.py")
