from flask import Flask, request, jsonify
import joblib
import pandas as pd
from utils import PhishingDetectorPipeline, LogTransformer
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Load the model right away
logging.info("Loading the phishing detection model...")
model = joblib.load("best_phishing_model.pkl")
logging.info("Model’s loaded and good to go!")

expected_features = [
    "num_words",
    "num_unique_words",
    "num_stopwords",
    "num_links",
    "num_unique_domains",
    "num_email_addresses",
    "num_spelling_errors",
    "num_urgent_keywords",
]


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict if an email is phishing based on input features.
    Expects a JSON payload with all the features listed in expected_features.
    Returns a prediction (0 or 1) and the probability of phishing.
    """
    data = request.get_json()

    # Confirm we have all the features we need
    if not all(feature in data for feature in expected_features):
        missing = [f for f in expected_features if f not in data]
        error_msg = f"Missing some features: {', '.join(missing)}. Need: {', '.join(expected_features)}"
        logging.warning(error_msg)
        return jsonify({"error": error_msg}), 400

    try:
        # Shape the input into a DataFrame the model can handle
        input_data = {feature: [data[feature]] for feature in expected_features}
        input_df = pd.DataFrame(input_data)

        # Get the prediction and the phishing probability
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        logging.info(
            f"Predicted: {prediction}, Phishing probability: {probability:.4f}"
        )

        # Send back the results formatted
        return jsonify(
            {"prediction": int(prediction), "probability": float(probability)}
        )

    except ValueError as ve:
        # Handle bad input values specifically
        logging.error(f"ValueError during prediction: {str(ve)}")
        return jsonify({"error": "Invalid input data. Check your feature values."}), 400
    except Exception as e:
        # Catch anything else that goes wrong
        logging.error(f"Unexpected error: {str(e)}")
        return (
            jsonify({"error": "Something went wrong. Please try again."}),
            500,
        )


if __name__ == "__main__":
    print("Phishing detection API is starting...")
    app.run(debug=True)
