import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew, ks_2samp
import logging
from sklearn.metrics import precision_recall_curve, auc

# Constant for p-value threshold
P_VALUE_THRESHOLD = 0.05


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a log transformation to features that exceed a skewness threshold.
    Useful for normalizing skewed data and stabilizing variance in ML pipelines.
    """

    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.features_to_transform = []

    def fit(self, X, y=None):
        # Check each column for skewness and flag the ones we want to transform
        for col in X.columns:
            if skew(X[col]) > self.threshold:
                self.features_to_transform.append(col)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.features_to_transform:
            # Using log1p to handle zeros
            X_transformed[col] = np.log1p(X_transformed[col])
        return X_transformed


class PhishingDetectorPipeline:
    """
    A wrapper around a scikit-learn pipeline that adds monitoring for data drift and performance.
    Keeps track of predictions and uses KS tests to spot distribution shifts over time.
    """

    def __init__(self, pipeline, sample_size=1000):
        self.pipeline = pipeline
        # Sample size controls how much data we use for drift checks
        self.sample_size = sample_size
        self.training_sample = None

    def fit(self, X, y=None, **fit_params):
        self.pipeline.fit(X, y, **fit_params)
        # Save a sample of the training data for later drift monitoring
        if len(X) > self.sample_size:
            self.training_sample = X.sample(self.sample_size, random_state=42)
        else:
            self.training_sample = X.copy()
        return self

    def predict(self, X, **predict_params):
        # Monitor the incoming data for any weird shifts
        self._monitor(X)
        y_pred = self.pipeline.predict(X, **predict_params)
        # Log how many predictions are phishing related for tracking
        phishing_ratio = np.mean(y_pred)
        logging.info(f"Proportion of predicted phishing emails: {phishing_ratio:.4f}")
        return y_pred

    def predict_proba(self, X, **predict_proba_params):
        # Check for drift before predicting
        self._monitor(X)
        y_proba = self.pipeline.predict_proba(X, **predict_proba_params)
        # Log the average phishing probability to keep an eye on trends
        avg_proba = np.mean(y_proba[:, 1])
        logging.info(f"Average predicted probability of phishing: {avg_proba:.4f}")
        return y_proba

    def evaluate(self, X_val, y_val):
        y_val_proba = self.pipeline.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
        auprc = auc(recall, precision)
        logging.info(f"Validation AUPRC: {auprc:.4f}")
        # Warn if AUPRC drops below 0.9
        if auprc < 0.9:
            logging.warning("AUPRC below threshold, consider retraining the model")
        return auprc

    def _monitor(self, X):
        if self.training_sample is not None:
            # Sample the new data if it’s bigger than our reference size
            if len(X) > self.sample_size:
                new_sample = X.sample(self.sample_size, random_state=42)
            else:
                new_sample = X.copy()
            for feature in X.columns:
                # KS test to see if this feature’s distribution has drifted
                stat, p_value = ks_2samp(
                    self.training_sample[feature], new_sample[feature]
                )
                if p_value < P_VALUE_THRESHOLD:
                    logging.warning(
                        f"Data drift detected in feature '{feature}' (p-value: {p_value:.4f})"
                    )
