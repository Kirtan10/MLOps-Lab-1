import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.train_and_save_model import download_data, preprocess_data, train_model
from src.train_and_save_model import evaluate_model
from src.train_and_save_model import get_model_version, update_model_version
from src.train_and_save_model import ensure_folder_exists, save_model_to_gcs
from src.train_and_save_model import save_metrics_to_gcs


# Small sample dataset reused by several tests.
# MODIFIED: it now contains two classes, because Logistic Regression (unlike
# Random Forest) cannot be trained on data that has only a single class.
def make_sample_data():
    X = pd.DataFrame({
        'sepal length (cm)': [5.1, 4.9, 4.7, 7.0, 6.4, 6.9],
        'sepal width (cm)': [3.5, 3.0, 3.2, 3.2, 3.2, 3.1],
        'petal length (cm)': [1.4, 1.4, 1.3, 4.7, 4.5, 4.9],
        'petal width (cm)': [0.2, 0.2, 0.2, 1.4, 1.5, 1.5],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    return X, y


# ----------------- Test Download ----------------- #
def test_download_data():
    X, y = download_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty
    assert X.shape[0] == y.shape[0]


# ----------------- Test Preprocess ----------------- #
def test_preprocess_data():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    assert X_train.shape[1] == X.shape[1]


# ----------------- Test Train model ----------------- #
# MODIFIED: checks for a Pipeline with a StandardScaler followed by LogisticRegression
def test_train_model():
    X, y = make_sample_data()
    model = train_model(X, y)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps['scaler'], StandardScaler)
    assert isinstance(model.named_steps['classifier'], LogisticRegression)
    assert hasattr(model, 'predict')

    # The trained model should produce one prediction per input row
    predictions = model.predict(X)
    assert len(predictions) == len(X)


# ----------------- Test Evaluate model ----------------- #
# NEW: checks that evaluate_model returns all four metrics with valid values
def test_evaluate_model():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    for value in metrics.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
    # Iris is an easy dataset, so a reasonable model should score well
    assert metrics["accuracy"] >= 0.9


# ----------------- Test Model versioning ----------------- #
def test_get_model_version():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        bucket_name = "bucket-test"
        version_file_name = "version.txt"

        # Version file exists.
        # FIXED: the original test set download_as_text's return value AFTER calling
        # the function, so it only passed because int(MagicMock()) happens to equal 1.
        # The return value is now configured before the call, and uses '3' so the
        # test proves the value is really read from the blob.
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = '3'
        version = get_model_version(bucket_name, version_file_name)

        assert version == 3
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_called_once()

        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()

        # Version file does not exist
        mock_blob.exists.return_value = False
        version = get_model_version(bucket_name, version_file_name)

        assert version == 0
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_not_called()


# ----------------- Test Update Model version ----------------- #
def test_update_model_version():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        bucket_name = 'bucket-test'
        version_file_name = 'version.txt'
        new_version = 2

        # Successful update
        result = update_model_version(bucket_name, version_file_name, new_version)
        assert result is True
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))

        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()

        # Invalid (non-integer) version
        with pytest.raises(ValueError):
            update_model_version(bucket_name, version_file_name, 'invalid_version')

        # Upload failure
        mock_blob.upload_from_string.side_effect = Exception("Upload failed")
        result = update_model_version(bucket_name, version_file_name, new_version)
        assert result is False
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))


# ----------------- Test Ensure Folder Exists ----------------- #
def test_ensure_folder_exists():
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    folder_name = "trained_models"

    # Folder does not exist -> it should be created
    mock_blob.exists.return_value = False
    ensure_folder_exists(mock_bucket, folder_name)
    mock_bucket.blob.assert_called_with(f"{folder_name}/")
    mock_blob.upload_from_string.assert_called_once_with('')

    mock_blob.reset_mock()

    # Folder exists -> nothing should be uploaded
    mock_blob.exists.return_value = True
    ensure_folder_exists(mock_bucket, folder_name)
    mock_bucket.blob.assert_called_with(f"{folder_name}/")
    mock_blob.upload_from_string.assert_not_called()


# ----------------- Test Save model to GCS ----------------- #
# MODIFIED: saves a trained Logistic Regression pipeline instead of a RandomForestClassifier
def test_save_model_to_gcs():
    X, y = make_sample_data()
    model = train_model(X, y)

    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        save_model_to_gcs(model, 'bucket-test', 'blob-test')

        mock_storage_client.assert_called_once()
        mock_storage_client.return_value.bucket.assert_called_once_with('bucket-test')
        # blob() is called twice: once for the folder check, once for the model upload
        assert mock_bucket.blob.call_count == 2
        mock_bucket.blob.assert_any_call('trained_models/')
        mock_bucket.blob.assert_any_call('blob-test')
        mock_blob.upload_from_filename.assert_called_once_with('model.joblib')


# ----------------- Test Save metrics to GCS ----------------- #
# NEW: checks that metrics are uploaded as valid JSON, and that upload errors are handled
def test_save_metrics_to_gcs():
    metrics = {
        "model_version": 5,
        "metrics": {"accuracy": 0.97, "f1_macro": 0.96},
    }

    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Successful upload
        result = save_metrics_to_gcs(metrics, 'bucket-test', 'metrics/metrics_v5.json')

        assert result is True
        mock_storage_client.return_value.bucket.assert_called_once_with('bucket-test')
        mock_bucket.blob.assert_called_once_with('metrics/metrics_v5.json')
        mock_blob.upload_from_string.assert_called_once()

        # Inspect what was uploaded: it should be JSON that matches the original dict
        args, kwargs = mock_blob.upload_from_string.call_args
        assert json.loads(args[0]) == metrics
        assert kwargs["content_type"] == "application/json"

        mock_blob.reset_mock()

        # Upload failure -> function should return False instead of crashing
        mock_blob.upload_from_string.side_effect = Exception("Upload failed")
        result = save_metrics_to_gcs(metrics, 'bucket-test', 'metrics/metrics_v5.json')
        assert result is False