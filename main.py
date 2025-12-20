from flask import Flask, jsonify, request
from decisionTree import classifier

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify(message="Hello!")

latest_data = {}

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
        print(f"Sending prediction back to Node: {prediction}")
        return jsonify({"prediction": prediction[0]}), 200
    
    except Exception as e:
        print(f"Error during classification: {e}")  
        return jsonify({"error": str(e)}), 500

@app.route("/data", methods=["GET"])
def data():
    return jsonify(latest_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)