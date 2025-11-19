import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
encoders = joblib.load("encoders.pkl")
feature_cols = joblib.load("feature_cols.pkl")
threshold = joblib.load("threshold.pkl")

# GUI setup
root = tk.Tk()
root.title("Stroke Prediction System")
root.geometry("850x650")
root.configure(bg="#f0f4f7")

fields = {
    "Gender": ["Female", "Male"],
    "Age": None,
    "Hypertension": ["No", "Yes"],
    "Heart Disease": ["No", "Yes"],
    "Ever Married": ["Yes", "No"],
    "Work Type": ["Govt_job", "Self-employed", "Children", "Private"],
    "Residence Type": ["Urban", "Rural"],
    "Avg Glucose Level": None,
    "BMI": None,
    "Smoking Status": ["Unknown", "formerly smoked", "never smoked", "smokes"]
}

widgets = {}
form = ttk.Frame(root)
form.pack(pady=10)

for i, (label, opts) in enumerate(fields.items()):
    ttk.Label(form, text=label).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    if opts:
        cb = ttk.Combobox(form, values=opts, state="readonly", width=30)
        cb.grid(row=i, column=1)
        cb.current(0)
        widgets[label] = cb
    else:
        en = ttk.Entry(form, width=32)
        en.grid(row=i, column=1)
        widgets[label] = en

result_label = ttk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

# Suggestion box
suggestion_box = tk.Text(root, width=90, height=10, wrap="word", font=("Arial", 11))
suggestion_box.pack(pady=10)

# Personalized suggestions
def get_suggestions(pred, age, bmi, glucose, smoking, hypertension, heart_disease):
    suggestions = []

    if pred == 1:
        suggestions.append("⚠ Immediate lifestyle and medical monitoring recommended.")

        if age > 70:
            suggestions.append("• Higher age increases stroke risk — regular check-ups required.")

        if bmi >= 30:
            suggestions.append("• High BMI detected — weight management can significantly reduce risk.")

        if glucose > 140:
            suggestions.append("• Elevated glucose level — consider diabetes screening.")

        if smoking in ["smokes", "formerly smoked"]:
            suggestions.append("• Smoking increases blood vessel damage — quitting is strongly advised.")

        if hypertension == 1:
            suggestions.append("• High blood pressure detected — strict monitoring needed.")

        if heart_disease == 1:
            suggestions.append("• Heart condition increases stroke risk — follow your cardiologist regularly.")

        suggestions.append("\n👉 Please consult a healthcare professional for a detailed evaluation.")

    else:
        suggestions.append("✔ You are currently at LOW risk of stroke.")
        suggestions.append("• Maintain a balanced diet and regular exercise.")
        suggestions.append("• Go for routine health check-ups annually.")
        suggestions.append("• Monitor glucose levels and blood pressure.")
        suggestions.append("• Avoid smoking and alcohol overuse.")

    return "\n".join(suggestions)


# Prediction function
def predict():
    try:
        user = {lbl: wd.get() for lbl, wd in widgets.items()}
        df = pd.DataFrame([user])

        # Convert Yes/No to numeric
        df["Hypertension"] = df["Hypertension"].map({"No": 0, "Yes": 1})
        df["Heart Disease"] = df["Heart Disease"].map({"No": 0, "Yes": 1})

        # Rename for training format
        df.rename(columns={
            "Gender": "gender", "Age": "age", "Hypertension": "hypertension",
            "Heart Disease": "heart_disease", "Ever Married": "ever_married",
            "Work Type": "work_type", "Residence Type": "Residence_type",
            "Avg Glucose Level": "avg_glucose_level", "BMI": "bmi",
            "Smoking Status": "smoking_status"
        }, inplace=True)

        # Convert numeric
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['avg_glucose_level'] = pd.to_numeric(df['avg_glucose_level'], errors='coerce')
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

        # Encode categoricals
        for col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

        # Impute missing
        df[['bmi']] = imputer.transform(df[['bmi']])

        # Arrange columns
        df = df[feature_cols]

        # Scale
        scaled = scaler.transform(df)

        # Predict
        probability = model.predict_proba(scaled)[0][1]
        pred = 1 if probability >= threshold else 0

        # Update UI
        result_label.config(
            text="⚠ Stroke Likely!" if pred == 1 else "✔ No Stroke Detected",
            foreground="red" if pred == 1 else "green"
        )

        # Show suggestions
        suggestion_box.delete("1.0", tk.END)
        suggestion_text = get_suggestions(
            pred,
            df['age'].iloc[0],
            df['bmi'].iloc[0],
            df['avg_glucose_level'].iloc[0],
            df['smoking_status'].iloc[0],
            df['hypertension'].iloc[0],
            df['heart_disease'].iloc[0]
        )
        suggestion_box.insert(tk.END, suggestion_text)

    except Exception as e:
        messagebox.showerror("ERROR", f"Input processing failed:\n{str(e)}")


ttk.Button(root, text="Predict Stroke Risk", command=predict).pack(pady=15)

root.mainloop()
