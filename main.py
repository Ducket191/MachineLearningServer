from flask import Flask, jsonify, request
from flask_cors import CORS
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
    return jsonify({"status": "received"}), 200

@app.route("/classify", methods=["POST"])
def classify():
    from decisionTree import classifier 
    data = request.json
    prediction = classifier.predict(data)
    return jsonify({"prediction": prediction[0]}), 200

@app.route("/data", methods=["GET"])
def data():
    return jsonify(latest_data)
