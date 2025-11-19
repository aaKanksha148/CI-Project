# ==========================================
# Stroke Prediction Model
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

print("🔄 Training Started...")

# Loading dataset
df = pd.read_csv(r"C:\Users\AAKANKSHA\Downloads\synthetic_stroke_data.csv")

# Fixing categories to match GUI
df['work_type'] = df['work_type'].str.replace("Govt job", "Govt_job", regex=False)
df['smoking_status'] = df['smoking_status'].str.replace("formerly smokes","formerly smoked", regex=False)
df['smoking_status'] = df['smoking_status'].str.replace("never smokes", "never smoked", regex=False)

# Droping ID column
df.drop(columns=['id'], inplace=True)

# Handling missing BMI
imputer = SimpleImputer(strategy="median")
df['bmi'] = imputer.fit_transform(df[['bmi']])

# Creating stroke label
df['risk_score'] = 0
df['risk_score'] += np.where(df['age'] >= 60, 2, 0)
df['risk_score'] += np.where(df['hypertension'] == 1, 2, 0)
df['risk_score'] += np.where(df['heart_disease'] == 1, 2, 0)
df['risk_score'] += np.where(df['avg_glucose_level'] >= 140, 2, 0)
df['risk_score'] += np.where(df['bmi'] >= 30, 1, 0)
df['risk_score'] += np.where(df['smoking_status'].isin(["smokes","formerly smoked"]), 1, 0)

df['stroke'] = np.where(df['risk_score'] >= 5, 1, 0)
df.drop(columns=["risk_score"], inplace=True)

print(df['stroke'].value_counts())

# Encoding categoricals
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Splitting
X = df.drop('stroke', axis=1)
y = df['stroke']
feature_cols = list(X.columns)

# Balancing stroke class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Training model
model = RandomForestClassifier(
    n_estimators=350,
    class_weight={0:1, 1:3},
    random_state=42
)
model.fit(X_train_scaled, y_train_bal)

# Evaluation
y_prob = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.45  # tuning for high sensitivity

y_pred = (y_prob >= threshold).astype(int)

print("\n=== MODEL RESULTS ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# y_test = true labels
# y_pred = predictions based on our threshold
# y_prob = model.predict_proba(X_test_scaled)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("===== MODEL PERFORMANCE =====")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-score       : {f1:.4f}")
print(f"ROC-AUC Score  : {roc_auc:.4f}")


# Saving artifacts
output_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(model, os.path.join(output_dir, "stroke_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
joblib.dump(imputer, os.path.join(output_dir, "imputer.pkl"))
joblib.dump(encoders, os.path.join(output_dir, "encoders.pkl"))
joblib.dump(feature_cols, os.path.join(output_dir, "feature_cols.pkl"))
joblib.dump(threshold, os.path.join(output_dir, "threshold.pkl"))

print("\n🎯 NEW MEDICALLY CORRECTED MODEL SAVED SUCCESSFULLY!")
