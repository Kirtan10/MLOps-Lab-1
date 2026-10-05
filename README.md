# CI/CD for ML with GitHub Actions and GCP: Logistic Regression and Metrics Tracking

This project is a modified version of [Lab 4 (GitHub Labs)](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab4) from the MLOps course repository. It builds a CI/CD pipeline that, on every push to `main`, tests the code, trains a model, versions it in Google Cloud Storage, records its evaluation metrics, and packages it as a Docker image in Google Artifact Registry.

## What I changed from the original lab

| Area | Original lab | This version |
|---|---|---|
| Model | `RandomForestClassifier` | `Pipeline` of `StandardScaler` + `LogisticRegression` |
| Evaluation | Accuracy only, printed to the console | Accuracy, macro precision, macro recall and macro F1 |
| Metrics storage | Not saved | A JSON metrics record is uploaded to GCS for every model version |
| Tests | 7 tests, one of which passed by accident | 9 tests, including new tests for evaluation and metrics upload, and a fixed versioning test |
| Workflow | Written for the monorepo; several bugs | Adapted to a standalone repo, bugs fixed |

### 1. Logistic Regression instead of Random Forest

`train_model()` now returns a scikit-learn `Pipeline` with two steps: a `StandardScaler` followed by `LogisticRegression(max_iter=1000)`.

Logistic Regression is sensitive to the scale of its input features, whereas tree-based models like Random Forest are not. Standardizing the features helps the solver converge and keeps any single feature from dominating the coefficients. Because the scaler is part of the pipeline, it is saved inside the same `.joblib` file as the classifier, so any code that loads the model later applies exactly the same scaling automatically.

### 2. Richer evaluation and metrics tracking

A new `evaluate_model()` function returns four metrics: accuracy, precision, recall and F1. Precision, recall and F1 use macro averaging, which gives each of the three Iris classes equal weight.

A new `save_metrics_to_gcs()` function uploads a JSON record for each run to `metrics/metrics_v{N}_{timestamp}.json` in the bucket. Each record links the scores to a specific model version and file, so you can look back and see how every version performed:

```json
{
  "model_version": 3,
  "model_type": "LogisticRegression (with StandardScaler)",
  "dataset": "iris",
  "timestamp": "20261005143210",
  "model_path": "gs://<bucket>/trained_models/model_v3_20261005143210.joblib",
  "metrics": {
    "accuracy": 1.0,
    "precision_macro": 1.0,
    "recall_macro": 1.0,
    "f1_macro": 1.0
  }
}
```

(Iris is a small, easily separable dataset, so perfect test scores are expected here.)

### 3. Updated and extended tests

Here is what changed in `test/test_pytest.py`:

- **`test_train_model`** now checks that the model is a `Pipeline` containing a `StandardScaler` and a `LogisticRegression`. Its sample data now contains two classes, because Logistic Regression cannot be fitted on data with only one class (Random Forest can, which is why the original sample data used a single class).
- **`test_save_model_to_gcs`** now saves a trained Logistic Regression pipeline.
- **`test_evaluate_model`** (new) checks that all four metrics are returned as floats between 0 and 1, and that accuracy on Iris is at least 0.9.
- **`test_save_metrics_to_gcs`** (new) uses `MagicMock` to capture the uploaded content. It checks that the content is valid JSON matching the input and is sent with the `application/json` content type, and that an upload failure returns `False` instead of raising.
- **`test_get_model_version`** contained a bug, which I fixed. The original test set the mocked `download_as_text` return value *after* calling the function. It only passed because `int()` of a `MagicMock` happens to return `1`, the expected value. The mock is now configured before the call and returns `'3'`, so the test really proves that the version is read from the blob.

### 4. Workflow fixes

The original workflow was written to run from inside the course monorepo. I made these changes so it works as a standalone repo:

- I removed `working-directory: Labs/Github_Labs/Lab4/` and fixed the `requirements.txt` path in the cache key.
- I renamed the secret `GCP_SA_KEY2` to `GCP_SA_KEY`, to match the documented setup.
- I changed `secrets.REGION` to `env.REGION` in the image name. No such secret existed, so the image path started with `-docker.pkg.dev`.
- The Docker image is now tagged with `env.MODEL_VERSION` instead of `env.VERSION_FILE_NAME`. The latter is empty in that step, so the version tag was blank.
- I aligned the region and repository name with the setup instructions (`us-east4`, `my-repo`).
- I removed the nightly `cron` schedule, so the pipeline does not retrain and push a new image every day. It can still be run manually with `workflow_dispatch`.
- I upgraded `google-github-actions/setup-gcloud` from v1 to v2.

## How the pipeline works

On every push or pull request to `main`, `.github/workflows/ci_cd_pipeline.yml` runs these steps:

1. Checks out the code, sets up Python 3.10, and installs dependencies (with pip caching).
2. Runs the test suite with `pytest`. If any test fails, the pipeline stops here.
3. Authenticates to GCP with a service account key stored in GitHub Secrets.
4. Runs `src/train_and_save_model.py`, which:
   - reads the current version from `model_version.txt` in GCS and increments it
   - trains the Logistic Regression pipeline on Iris and evaluates it
   - uploads the model to `trained_models/` and the metrics to `metrics/`
   - writes the new version number back to GCS
5. Builds a Docker image and pushes it to Artifact Registry, tagged with both the version number and `latest`.

Resulting bucket layout:

```
<bucket>/
├── model_version.txt
├── trained_models/
│   ├── model_v1_<timestamp>.joblib
│   └── model_v2_<timestamp>.joblib
└── metrics/
    ├── metrics_v1_<timestamp>.json
    └── metrics_v2_<timestamp>.json
```

## Project structure

```
.
├── .github/workflows/ci_cd_pipeline.yml   # CI/CD workflow
├── src/train_and_save_model.py            # Training, evaluation, versioning, GCS upload
├── test/test_pytest.py                    # Unit tests (GCS calls mocked)
├── Dockerfile                             # Packages the trained model
├── Mock.md                                # Notes on MagicMock and patch
└── requirements.txt
```

## Setup

### GCP

1. Create a GCP project and enable the **Cloud Storage**, **Cloud Build** and **Artifact Registry** APIs.
2. Create a service account with the **Storage Admin**, **Storage Object Admin** and **Artifact Registry Administrator** roles, and download a JSON key for it.
3. Create a GCS bucket.
4. Create a Docker repository in Artifact Registry named `my-repo` in `us-east4`:
   ```bash
   gcloud artifacts repositories create my-repo --repository-format=docker --location=us-east4
   ```

### GitHub Secrets

In **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|---|---|
| `GCP_SA_KEY` | Full contents of the service account JSON key |
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCS_BUCKET_NAME` | Your bucket name |
| `VERSION_FILE_NAME` | `model_version.txt` |

### Local development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the tests (no GCP access needed; storage calls are mocked)
pytest test/ -v
```

To run the training script locally, create a `.env` file:

```
GCS_BUCKET_NAME=your-bucket-name
VERSION_FILE_NAME=model_version.txt
```

Then point to your service account key and run the script:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
python src/train_and_save_model.py
```

Never commit `.env` or the JSON key. Both are excluded by `.gitignore`.

## Verifying a run

After the workflow succeeds:

- **Cloud Storage → your bucket** contains the new model in `trained_models/`, its metrics JSON in `metrics/`, and an incremented `model_version.txt`.
- **Artifact Registry → my-repo → model-image** shows an image tagged with the new version number and `latest`.

## Credits

This project is based on Lab 4 of the GitHub Labs in [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps).