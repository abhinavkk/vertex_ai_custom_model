from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model and tokenizer
model_dir = "/model"  # Path where the model is stored
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

@app.route("/", methods=['GET'])
def home():
    html = "<h3>Model Serving</h3>"
    return html.format(format)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the request JSON
        data = request.get_json()

        # Ensure `instances` is provided in the payload
        if "instances" not in data:
            return jsonify({"error": "The request payload must include an 'instances' key."}), 400

        # Extract instances
        instances = data["instances"]

        # Tokenize input instances
        inputs = tokenizer(instances, return_tensors="pt", padding=True, truncation=True)

        # Generate predictions
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1).tolist()

        # Format predictions with input data
        results = [{"input": text, "prediction": prediction} for text, prediction in zip(instances, predictions)]

        # Return predictions
        return jsonify({"predictions": results}), 200

    except Exception as e:
        # Handle exceptions and return error message
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
