# Phishing Detection Solution

Phishing attacks remain a persistent challenge in cybersecurity, threatening both individuals and organizations. This project tackles that challenge head on by delivering a machine learning solution to identify phishing emails. Built around a trained Random Forest model, the solution classifies emails as phishing or safe based on carefully extracted features. To make it practical and accessible, a REST API is included for real time predictions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the API](#running-the-api)
- [Using the API](#using-the-api)
- [Input Features](#input-features)
- [Project Structure](#project-structure)

## Prerequisites

To get started ensure your environment meets these requirements:

- **Python 3.8 or higher**: The project relies on modern Python features and library compatibility.
- **pip**: The Python package installer, essential for dependency management.

You'll also need the following libraries:

- `flask`
- `pandas`
- `scikit-learn`
- `joblib`
- `numpy`
- `matplotlib`
- `seaborn`
- `imblearn`
- `xgboost`
- `lightgbm`
- `shap`

Installation instructions are provided in the [Setup](#setup) section below.

## Setup

Follow these steps to prepare the project for use:

1. **Install Dependencies**

   Use pip to install the required libraries with this command:

   ```bash
   pip install flask pandas scikit-learn joblib numpy matplotlib seaborn imblearn xgboost lightgbm shap
   ```

2. **Prepare the Model File**

   The API depends on a trained model stored in `best_phishing_model.pkl`, which must reside in the project directory. If you don’t already have this file, generate it by running the training script:

   ```bash
   python phishing_detector.py
   ```

   This script processes the data, trains the model, and saves it as `best_phishing_model.pkl`.

## Running the API

Launch the API server with this simple command from the project directory:

```bash
python api.py
```

This starts a Flask server, and the API becomes available at `http://127.0.0.1:5000/predict`. It’s ready to handle prediction requests immediately.

## Using the API

The API offers a single endpoint, `/predict`, designed to accept POST requests with a JSON payload containing email features. It returns a prediction—phishing (1) or safe (0)—along with the model’s confidence in the phishing classification.

### Example Request

Test the API using `curl`, Postman, or any HTTP client. Here’s an example with `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "num_words": 100,
  "num_unique_words": 50,
  "num_stopwords": 30,
  "num_links": 2,
  "num_unique_domains": 1,
  "num_email_addresses": 0,
  "num_spelling_errors": 5,
  "num_urgent_keywords": 1
}' http://127.0.0.1:5000/predict
```

### Expected Response

The API responds with a JSON object like this:

```json
{
  "prediction": 0,
  "probability": 0.1234
}
```

- **`prediction`**: 0 means the email is safe; 1 indicates phishing.
- **`probability`**: A value between 0 and 1, reflecting the model’s confidence in a phishing classification.

**Note**: If any required features are missing from the input JSON, the API returns an error message listing the absent features.

## Input Features

The API expects these features in the JSON payload, all as non-negative numbers:

- **`num_words`**: Total word count in the email body.
- **`num_unique_words`**: Number of distinct words.
- **`num_stopwords`**: Count of common stopwords (i.e. "the", "and", "is").
- **`num_links`**: Number of hyperlinks present.
- **`num_unique_domains`**: Count of unique domains in those links (i.e. "example.com").
- **`num_email_addresses`**: Number of email addresses in the text.
- **`num_spelling_errors`**: Count of misspelled words.
- **`num_urgent_keywords`**: Number of urgency-related terms (i.e. "urgent", "verify", "now").

These features should be extracted from the email using suitable preprocessing tools or methods.

- **Data Drift Detection**: The `PhishingDetectorPipeline` class uses Kolmogorov-Smirnov tests to spot shifts in incoming data distributions. If a feature’s p-value drops below 0.05, a warning is logged.
- **Prediction Insights**: It tracks the proportion of phishing predictions and average probabilities, logging these for each batch.
- **Performance Checks**: The `evaluate` method calculates the Area Under the Precision-Recall Curve (AUPRC) on validation data, flagging a warning if it dips below 0.9.

## Project Structure

Below is how the project is organized:

- **`phishing_detector.py`**: Core script handling data preprocessing, model training, evaluation, and interpretation.
- **`api.py`**: Flask API server for deploying the trained model.
- **`utils.py`**: Contains helper classes like `LogTransformer` for feature scaling and `PhishingDetectorPipeline` for modularity and monitoring.
- **`best_phishing_model.pkl`**: The serialized, trained model file.
- **`README.md`**: Guide to setup and usage.
- **`report.md`**: Report covering the approach, assumptions, results, and challenges.