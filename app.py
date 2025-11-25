from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load model
model = YOLO("best.pt")

@app.route("/")
def home():
    return "Stain Classifier API Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded"}), 400

    save_path = "input.jpg"
    file.save(save_path)

    results = model(save_path)[0]
    predicted_class = results.names[results.probs.top1]
    confidence = float(results.probs.top1conf)

    return jsonify({
        "class": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
