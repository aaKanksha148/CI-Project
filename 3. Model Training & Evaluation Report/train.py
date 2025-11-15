# ==============================================
# Stroke Prediction using Random Forest Classifier
# ==============================================

# 1️⃣ Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import os

# Ensure output directory exists
output_dir = os.path.dirname(os.path.abspath(_file_))

# 2️⃣ Load dataset
file_path = r"C:\Software Projects\Software Projects\synthetic_stroke_prediction\synthetic_stroke_data.csv"
df = pd.read_csv(file_path)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# 3️⃣ Basic exploration
print("\nDataset info:")
print(df.info())
print("\nStatistical summary:\n", df.describe())

# Visualize stroke distribution
plt.figure(figsize=(6,4))
sns.countplot(x='stroke', data=df)
plt.title("Stroke Distribution (Imbalance Check)")
plt.savefig(os.path.join(output_dir, "stroke_distribution.png"), dpi=300, bbox_inches='tight')
plt.show()

# 4️⃣ Handle missing values (e.g., BMI)
imputer = SimpleImputer(strategy='median')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# 5️⃣ Encode categorical variables
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# 6️⃣ Feature-target split
X = df.drop(columns=['stroke', 'id'])
y = df['stroke']

# 7️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8️⃣ Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_bal))

# 9️⃣ Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# 🔟 Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train_bal)

# 11️⃣ Evaluation
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== RANDOM FOREST RESULTS ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.savefig(os.path.join(output_dir, "confusion_matrix_rf.png"), dpi=300, bbox_inches='tight')
plt.show()

# 12️⃣ Comparative model (Logistic Regression)
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_scaled, y_train_bal)
y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_log))

# Compare ROC curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob)
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)

plt.figure(figsize=(7,5))
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

# 13️⃣ Model deployment (save trained model)
joblib.dump(rf_model, os.path.join(output_dir, "stroke_random_forest_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
print("\n✅ Model and scaler saved successfully!")

# 14️⃣ Testing / Validation example
def predict_stroke(sample_input):
    """
    Input: dictionary containing patient info
    Output: stroke prediction (0 or 1)
    """
    input_df = pd.DataFrame([sample_input])

    # ⚠ Use same categorical columns and encoding mapping as training
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = encoder.fit_transform(input_df[col].astype(str))

    input_scaled = scaler.transform(input_df)
    prediction = rf_model.predict(input_scaled)[0]
    return "Stroke likely" if prediction == 1 else "No stroke detected"

# Example test
test_patient = {
    'gender': 'Male',
    'age': 67,
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 145.6,
    'bmi': 28.5,
    'smoking_status': 'formerly smoked'
}

print("\n🧍 Test Prediction:", predict_stroke(test_patient))