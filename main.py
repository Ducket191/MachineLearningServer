from flask import Flask, jsonify, request
from flask_cors import CORS
from decisionTree import classifier
import os

app = Flask(__name__)
CORS(app)

latest_data = {}

@app.route("/")
def home():
    return jsonify(message="Flask server is running")

@app.route("/receive", methods=["POST"])
def receive():
    global latest_data
    latest_data = request.json
    print("Received from Node:", latest_data)
    return jsonify({"status": "received"}), 200

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json

    try:
        prediction = classifier.predict(data)
        print("Prediction:", prediction)
        return jsonify({"prediction": prediction[0]}), 200

    except Exception as e:
        print("Error during classification:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/data", methods=["GET"])
def data():
    return jsonify(latest_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
