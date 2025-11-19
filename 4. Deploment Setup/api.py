from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

# Load model and preprocessing tools
model = joblib.load("stroke_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
encoders = joblib.load("encoders.pkl")

app = Flask(__name__)

# Feature order used during training
feature_order = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = []

        # Validate & preprocess input
        for feature in feature_order:
            value = data.get(feature, None)
            if value is None:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

            # Encode categorical features using stored encoders
            if feature in encoders:
                le = encoders[feature]
                if value in le.classes_:
                    value = int(le.transform([value])[0])
                else:
                    value = -1  # For unseen category values

            input_data.append(value)

        # Convert to array
        input_array = np.array(input_data).reshape(1, -1)

        # Impute missing bmi
        bmi_index = feature_order.index('bmi')
        input_array[:, bmi_index] = imputer.transform(
            input_array[:, bmi_index].reshape(-1, 1)
        )

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Model prediction
        prediction = int(model.predict(input_scaled)[0])

        return jsonify({
            "stroke_prediction": prediction,
            "message": "⚠ Stroke Likely" if prediction == 1 else "✔ No Stroke Detected"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Stroke Prediction API Running Successfully!"})


if __name__ == "__main__":
    app.run(debug=True)
