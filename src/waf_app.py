from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model
MODEL_PATH = "../models/waf_pipeline.joblib"
print("Loading model...")
pipeline = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Capture *any* route dynamically
@app.route("/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/<path:path>", methods=["GET", "POST"])
def detect_request(path):
    # Combine path and query for analysis
    full_request = request.path
    if request.query_string:
        full_request += "?" + request.query_string.decode(errors="ignore")

    print(f"Inspecting: {full_request}")

    # Extract components from pipeline
    vectorizer = pipeline["vectorizer"]
    model = pipeline["model"]

    # Predict malicious or benign
    X_vec = vectorizer.transform([full_request])
    prediction = model.predict(X_vec)[0]
    prob = model.predict_proba(X_vec)[0].tolist()

    result = {
        "request": full_request,
        "malicious": bool(prediction),
        "probabilities": prob
    }

    # Return 403 if malicious, else 200
    if prediction == 1:
        return jsonify(result), 403
    else:
        return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
