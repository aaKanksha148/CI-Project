import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import numpy as np
import os
import traceback

# Optional: try to import google generative client (may not be available)
try:
    import google.generativeai as genai  # optional; not required
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ================================================
# CONFIGURATION
# ================================================
MODEL_PATH = "stroke_random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
# Do NOT hardcode API keys. If you have one and a compatible client, set this env var.
GEMINI_API_KEY = os.getenv("AIzaSyDyS2vVU9R3n3qx8_g6IX_wgqkQfYoSyLg", None)

# Configure optional AI model if available and key provided
model_ai = None
if GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_ai = genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        model_ai = None

# ================================================
# LOAD MODEL & SCALER
# ================================================
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Error", f"Model not found: {MODEL_PATH}")
    raise FileNotFoundError(f"Missing file: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    messagebox.showerror("Error", f"Scaler not found: {SCALER_PATH}")
    raise FileNotFoundError(f"Missing file: {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ================================================
# TKINTER UI
# ================================================
root = tk.Tk()
root.title("Stroke Prediction & Suggestions")
root.geometry("900x680")
root.configure(bg="#f4f6f8")

style = ttk.Style()
style.configure("TLabel", background="#f4f6f8", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), background="#f4f6f8")

header = ttk.Label(root, text="Stroke Risk Predictor", style="Header.TLabel")
header.pack(pady=12)

form_frame = ttk.Frame(root)
form_frame.pack(pady=6, padx=10, anchor="w")

fields = {
    "Gender": ["Female", "Male", "Other"],
    "Age": None,
    "Hypertension": ["No", "Yes"],
    "Heart Disease": ["No", "Yes"],
    "Ever Married": ["No", "Yes"],
    "Work Type": ["Private", "Self-employed", "Govt job", "Children", "Never worked"],
    "Residence Type": ["Rural", "Urban"],
    "Avg Glucose Level": None,
    "BMI": None,
    "Smoking Status": ["never smoked", "formerly smoked", "smokes", "Unknown"]
}

widgets = {}
for i, (label, options) in enumerate(fields.items()):
    ttk.Label(form_frame, text=label).grid(row=i, column=0, sticky="w", padx=8, pady=6)
    if options:
        combo = ttk.Combobox(form_frame, values=options, state="readonly", width=28)
        combo.current(0)
        combo.grid(row=i, column=1, padx=8, pady=6)
        widgets[label] = combo
    else:
        entry = ttk.Entry(form_frame, width=30)
        entry.grid(row=i, column=1, padx=8, pady=6)
        widgets[label] = entry

# Result and suggestions
result_label = ttk.Label(root, text="", font=("Segoe UI", 13, "bold"))
result_label.pack(pady=8)

suggestions_label = ttk.Label(root, text="AI / Recommendations:", style="Header.TLabel")
suggestions_label.pack(pady=(10, 0))
suggestion_box = tk.Text(root, width=100, height=10, wrap="word", font=("Segoe UI", 10))
suggestion_box.pack(pady=6)

# Helpful mapping consistent with training preprocessing
work_type_map = {"Private": 0, "Self-employed": 1, "Govt job": 2, "Children": 3, "Never worked": 4}
smoking_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}

def local_recommendations(age, bmi, hypertension, heart_disease, smoking_status, glucose):
    recs = []
    if bmi is not None and bmi >= 30:
        recs.append("Aim for gradual weight loss through balanced diet and regular exercise.")
    if hypertension == 1:
        recs.append("Control blood pressure: adhere to medications and reduce salt intake.")
    if heart_disease == 1:
        recs.append("Follow cardiologist guidance; regular check-ups and medication adherence.")
    if smoking_status in (1, 2):
        recs.append("Quit smoking — seek counselling or nicotine-replacement support.")
    if glucose is not None and glucose > 140:
        recs.append("Monitor and control blood glucose; consult your physician for diabetes care.")
    if not recs:
        recs = ["Maintain balanced diet, regular physical activity, and routine health check-ups."]
    return recs

def predict_and_suggest():
    try:
        # read and validate inputs
        gender_str = widgets["Gender"].get()
        gender = 1 if gender_str == "Male" else 0  # training encoded Male=1, others=0
        hypertension = 1 if widgets["Hypertension"].get() == "Yes" else 0
        heart_disease = 1 if widgets["Heart Disease"].get() == "Yes" else 0
        ever_married = 1 if widgets["Ever Married"].get() == "Yes" else 0
        work_type = work_type_map.get(widgets["Work Type"].get(), 0)
        residence = 1 if widgets["Residence Type"].get() == "Urban" else 0
        smoking_status = smoking_map.get(widgets["Smoking Status"].get(), 3)

        # numeric conversions with clear error messages
        try:
            age = float(widgets["Age"].get())
        except Exception:
            raise ValueError("Invalid Age. Enter a numeric value.")

        try:
            glucose = float(widgets["Avg Glucose Level"].get())
        except Exception:
            raise ValueError("Invalid Avg Glucose Level. Enter a numeric value.")

        try:
            bmi = float(widgets["BMI"].get())
        except Exception:
            raise ValueError("Invalid BMI. Enter a numeric value.")

        # prepare input in the same feature order used during training:
        # [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]
        input_data = [gender, age, hypertension, heart_disease, ever_married, work_type, residence, glucose, bmi, smoking_status]
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]

        result_text = "⚠ Stroke Likely Detected!" if int(prediction) == 1 else "✅ No Stroke Detected."
        result_label.config(text=result_text, foreground="red" if int(prediction) == 1 else "green")

        # Generate AI suggestions if configured, otherwise provide local fallback
        suggestion_box.delete("1.0", tk.END)
        if model_ai is not None:
            try:
                prompt = f"A patient with these health details: {dict(zip(fields.keys(), input_data))}. Prediction: {result_text}. Suggest 3 actionable recommendations."
                ai_response = model_ai.generate_content(prompt)
                # ai_response may differ by client; attempt to extract text
                text = getattr(ai_response, "text", None) or str(ai_response)
                suggestion_box.insert(tk.END, text.strip() + "\n\n")
            except Exception as e:
                # handle API errors (rate limit / 429 etc.)
                traceback.print_exc()
                messagebox.showwarning("AI Error", f"AI suggestion failed: {e}\nProviding local recommendations instead.")
        # local recommendations
        recs = local_recommendations(age=age, bmi=bmi, hypertension=hypertension, heart_disease=heart_disease, smoking_status=smoking_status, glucose=glucose)
        suggestion_box.insert(tk.END, "\n".join(f"- {r}" for r in recs))

    except Exception as e:
        messagebox.showerror("Error", f"Please check your input values.\n\n{e}")

btn = ttk.Button(root, text="Predict & Get Suggestions", command=predict_and_suggest)
btn.pack(pady=12)

# Start the app
root.mainloop()