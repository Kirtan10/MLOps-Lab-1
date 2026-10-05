import os
import json
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from a .env file (used for local runs; in CI they come from GitHub Secrets)
load_dotenv()

BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')          # Google Cloud Storage bucket name
VERSION_FILE_NAME = os.getenv('VERSION_FILE_NAME')  # File in the bucket that stores the model version

import pandas as pd
import joblib
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Download the Iris dataset from scikit-learn
def download_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target)
    return features, target


# Split the data into training and testing sets
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# MODIFIED: Train a Logistic Regression model instead of a Random Forest.
# Logistic Regression is sensitive to feature scale, so the features are standardized
# first. Wrapping both steps in a Pipeline means the scaler is saved together with the
# model, so the exact same scaling is applied whenever the model makes predictions.
def train_model(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


# NEW: Evaluate the model with several metrics, not just accuracy.
# Macro averaging gives each class equal weight, which suits a multi-class problem like Iris.
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
    }
    return metrics


# Retrieve the current model version from GCS (0 if no version file exists yet)
def get_model_version(bucket_name, version_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(version_file_name)

    if blob.exists():
        version_as_string = blob.download_as_text()
        version = int(version_as_string)
    else:
        version = 0
    return version


# Update the model version stored in GCS
def update_model_version(bucket_name, version_file_name, version):
    if not isinstance(version, int):
        raise ValueError("Version must be an integer")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        blob.upload_from_string(str(version))
        return True
    except Exception as e:
        print(f"Error updating model version: {e}")
        return False


# Make sure a "folder" exists in the bucket
def ensure_folder_exists(bucket, folder_name):
    blob = bucket.blob(f"{folder_name}/")
    if not blob.exists():
        blob.upload_from_string('')
        print(f"Created folder: {folder_name}")


# Save the trained model locally and upload it to GCS
def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "model.joblib")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    ensure_folder_exists(bucket, "trained_models")

    blob = bucket.blob(blob_name)
    blob.upload_from_filename('model.joblib')


# NEW: Upload the evaluation metrics as a JSON file to GCS, so every model
# version has a permanent record of how well it performed.
def save_metrics_to_gcs(metrics, bucket_name, blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(metrics, indent=2), content_type="application/json")
        return True
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return False


def main():
    # Work out the new version number
    current_version = get_model_version(BUCKET_NAME, VERSION_FILE_NAME)
    new_version = current_version + 1

    # Load and split the data
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        print(f"{name}: {value}")

    # Save the model to GCS with the new version and a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_blob_name = f"trained_models/model_v{new_version}_{timestamp}.joblib"
    save_model_to_gcs(model, BUCKET_NAME, model_blob_name)
    print(f"Model saved to gs://{BUCKET_NAME}/{model_blob_name}")

    # NEW: Save the metrics next to the model, with details linking them to this version
    metrics_record = {
        "model_version": new_version,
        "model_type": "LogisticRegression (with StandardScaler)",
        "dataset": "iris",
        "timestamp": timestamp,
        "model_path": f"gs://{BUCKET_NAME}/{model_blob_name}",
        "metrics": metrics,
    }
    metrics_blob_name = f"metrics/metrics_v{new_version}_{timestamp}.json"
    if save_metrics_to_gcs(metrics_record, BUCKET_NAME, metrics_blob_name):
        print(f"Metrics saved to gs://{BUCKET_NAME}/{metrics_blob_name}")
    else:
        print("Failed to save metrics")

    # Update the version in GCS. The MODEL_VERSION_OUTPUT line is read by the
    # GitHub Actions workflow to tag the Docker image, so keep its format unchanged.
    if update_model_version(BUCKET_NAME, VERSION_FILE_NAME, new_version):
        print(f"Model version updated to {new_version}")
        print(f"MODEL_VERSION_OUTPUT: {new_version}")
    else:
        print("Failed to update model version")


if __name__ == "__main__":
    main()