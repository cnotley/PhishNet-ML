import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.inspection import PartialDependenceDisplay
import shap
import logging
import joblib
from utils import PhishingDetectorPipeline, LogTransformer
from sklearn.pipeline import Pipeline

# Define a constant for random state to ensure reproducibility
RANDOM_STATE = 42

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="phishing_detector.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_and_check_data(file_path):
    """Load the dataset and check for any missing values that could introduce issues later"""
    df = pd.read_csv(file_path)
    print("Missing Values:\n", df.isnull().sum())
    logging.info(f"Loaded data with shape {df.shape}")
    return df


def perform_exploratory_data_analysis(df):
    """Generate plots to get a feel for the data.  Visuals are super helpful for spotting patterns"""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="label", data=df)
    plt.title("Class Distribution (0: Safe, 1: Phishing)")
    plt.savefig("class_distribution.png")
    plt.close()

    # Loop through features to plot distributions and boxplots
    for col in df.columns[:-1]:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"{col}_distribution.png")
        plt.close()

    # Check how features vary by label
    for col in df.columns[:-1]:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x="label", y=col, data=df)
        plt.title(f"{col} by Label")
        plt.savefig(f"{col}_by_label.png")
        plt.close()

    # Correlation heatmap to see how features relate
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()


def preprocess_data(df):
    """
    Split features and target, detect outliers, handle imbalance, and split into sets.
    We're not dropping outliers since tree models can handle them fine.
    """
    X = df.drop("label", axis=1)
    y = df["label"]

    # Check for outliers with IQR method
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).sum()
    print("Outliers per feature:\n", outliers)
    logging.info(
        "Outliers detected but kept.  Tree models are robust, and RobustScaler helps."
    )

    # Look at class balance
    logging.info("Checking class imbalance")
    class_distribution = y.value_counts(normalize=True)
    print("Class Distribution:\n", class_distribution)

    # Handle imbalance if minority class is below 10%
    if class_distribution.min() < 0.1:
        adasyn = ADASYN(random_state=RANDOM_STATE)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y
        logging.info("No major imbalance found, skipping resampling")

    # Split into train, validation, and holdout sets
    X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_resampled,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    logging.info(
        f"Train size: {X_train.shape}, Val size: {X_val.shape}, Holdout size: {X_holdout.shape}"
    )
    return X_train, X_val, X_holdout, y_train, y_val, y_holdout


def train_and_tune_models(X_train, y_train, X_val, y_val):
    """Train multiple models and pick the best one based on AUPRC (great for imbalanced data)"""
    preprocessing_pipeline = Pipeline(
        [("log_transform", LogTransformer(threshold=1.0)), ("scaler", RobustScaler())]
    )

    # Models to try out
    models = {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
    }

    # Hyperparameters to tune
    param_grids = {
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.01, 0.1],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.01, 0.1],
        },
        "LightGBM": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.01, 0.1],
        },
        "Logistic Regression": {
            "classifier__C": [0.1, 1, 10],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear"],
        },
    }

    best_auprc = 0
    best_model_name = None
    best_prediction_pipeline = None

    # Loop through models and tune them
    for name, classifier in models.items():
        prediction_pipeline = Pipeline(
            [("preprocessing", preprocessing_pipeline), ("classifier", classifier)]
        )
        param_grid = param_grids[name]
        grid_search = GridSearchCV(
            prediction_pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring="average_precision",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # Log the best params for reference
        logging.info(f"Best params for {name}: {grid_search.best_params_}")

        best_pipeline = grid_search.best_estimator_
        y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
        auprc = auc(recall, precision)
        logging.info(f"{name} Validation AUPRC: {auprc:.4f}")

        if auprc > best_auprc:
            best_auprc = auprc
            best_model_name = name
            best_prediction_pipeline = best_pipeline

    logging.info(f"Best model: {best_model_name} with AUPRC {best_auprc:.4f}")
    return best_model_name, best_prediction_pipeline, best_auprc


def evaluate_model(phishing_detector, X_val, y_val, X_holdout, y_holdout):
    """Check how the model performs and pick a threshold balancing precision and recall"""
    y_val_proba = phishing_detector.predict_proba(X_val)[:, 1]
    precision_val, recall_val, thresholds_val = precision_recall_curve(
        y_val, y_val_proba
    )
    auprc_val = auc(recall_val, precision_val)
    print(f"Validation AUPRC: {auprc_val:.4f}")

    # Aim for recall >= 0.9 while maximizing precision
    candidates = np.where(recall_val >= 0.9)[0]
    if len(candidates) > 0:
        optimal_idx = candidates[np.argmax(precision_val[candidates])]
        optimal_threshold = thresholds_val[optimal_idx]
    else:
        # Fall back to best F1 if recall target isn’t met
        f1_scores = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_val[optimal_idx]

    y_holdout_proba = phishing_detector.predict_proba(X_holdout)[:, 1]
    y_holdout_pred = (y_holdout_proba >= optimal_threshold).astype(int)
    print(f"\nHoldout Set Performance (Threshold {optimal_threshold:.4f}):")
    print(classification_report(y_holdout, y_holdout_pred))
    roc_auc = roc_auc_score(y_holdout, y_holdout_proba)
    print(f"Holdout ROC-AUC: {roc_auc:.4f}")
    cm = confusion_matrix(y_holdout, y_holdout_pred)
    print("Confusion Matrix:\n", cm)

    phishing_detector.evaluate(X_holdout, y_holdout)

    # Plot PR curve for a quick visual
    plt.figure(figsize=(8, 6))
    plt.plot(recall_val, precision_val, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Validation Set)")
    plt.savefig("precision_recall_curve.png")
    plt.close()


def interpret_model(phishing_detector, X_holdout, X, df):
    """Dig into what drives the model with SHAP and feature importance"""
    explainer = shap.TreeExplainer(phishing_detector.pipeline.named_steps["classifier"])
    shap_values = explainer.shap_values(X_holdout)
    shap.summary_plot(shap_values, X_holdout, feature_names=X.columns, show=False)
    plt.savefig("shap_summary.png")
    plt.close()

    feature_importance = pd.Series(
        phishing_detector.pipeline.named_steps["classifier"].feature_importances_,
        index=X.columns,
    ).sort_values()
    top_features = feature_importance.tail(3).index.tolist()

    for feature in top_features:
        PartialDependenceDisplay.from_estimator(
            phishing_detector.pipeline, X_holdout, [feature], feature_names=X.columns
        )
        plt.savefig(f"{feature}_pdp.png")
        plt.close()

    print("\nFeature Importance:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")

    phishing_means = df[df["label"] == 1].drop("label", axis=1).mean()
    safe_means = df[df["label"] == 0].drop("label", axis=1).mean()
    print("\nTop Features Explanation:")
    for feature in top_features:
        if feature == "num_email_addresses":
            explanation = f"Safe emails have more addresses (mean={safe_means[feature]:.2f}) than phishing ones (mean={phishing_means[feature]:.2f})—maybe legit emails list more contacts."
        else:
            explanation = f"Higher {feature} (phishing mean={phishing_means[feature]:.2f} vs safe mean={safe_means[feature]:.2f}) hints at phishing tactics, like urgency keywords."
        print(f"- {explanation}")


if __name__ == "__main__":
    # Main execution flow
    logging.info("Kicking off the phishing detection process")

    # Step 1 - load and inspect the data
    df = load_and_check_data("phishing_data.csv")

    # Step 2 - explore the data visually
    perform_exploratory_data_analysis(df)

    # Step 3 - get the data ready for modeling
    X_train, X_val, X_holdout, y_train, y_val, y_holdout = preprocess_data(df)

    # Step 4 - train and pick the best model
    best_model_name, best_prediction_pipeline, best_auprc = train_and_tune_models(
        X_train, y_train, X_val, y_val
    )

    # Step 5 - wrap it in our custom pipeline
    phishing_detector = PhishingDetectorPipeline(best_prediction_pipeline)
    phishing_detector.fit(X_train, y_train)
    joblib.dump(phishing_detector, "best_phishing_model.pkl")
    logging.info("Model saved to 'best_phishing_model.pkl'")

    # Step 6 - test how it performs
    evaluate_model(phishing_detector, X_val, y_val, X_holdout, y_holdout)

    # Step 7 - understand what it’s doing
    interpret_model(phishing_detector, X_holdout, df.drop("label", axis=1), df)

    logging.info("All done—ready for review or deployment!")
