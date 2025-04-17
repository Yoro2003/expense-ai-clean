from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model("model/expense_model.h5")

@app.route("/")
def home():
    return jsonify({"message": "API Keras en ligne ✅"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        amount = data.get("amount", 0)
        category = data.get("category", 0)  # Expected to be numeric already
        type_encoded = data.get("type", 0)  # Expected to be numeric already

        input_array = np.array([[amount, category, type_encoded]])
        prediction = model.predict(input_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        label_map = {0: "essentiel", 1: "non_essentiel"}
        return jsonify({
            "prediction": label_map.get(predicted_class, "inconnu"),
            "raw_output": prediction.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
