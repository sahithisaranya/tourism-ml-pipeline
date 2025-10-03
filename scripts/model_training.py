import pandas as pd
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import joblib
from huggingface_hub import HfApi, upload_file
from scipy.stats import uniform, randint
import os

# Set MLflow experiment
mlflow.set_experiment("tourism_xgb_experiment")

# Load processed data from Hugging Face
dataset = load_dataset("sahithisaranya/tourism_dataset_processed")
df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()

X_train = df_train.drop(columns=["ProdTaken"])
y_train = df_train["ProdTaken"].astype(int)
X_test = df_test.drop(columns=["ProdTaken"])
y_test = df_test["ProdTaken"].astype(int)

# Load the preprocessor
preprocessor = joblib.load("tourism_project/model_building/preprocessor.joblib")

# Define the pipeline (preprocessor + model)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb)
])

# Define parameter search space for XGBoost
param_distributions = {
    'model__n_estimators': randint(100, 600),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(0.01, 0.29),
    'model__subsample': uniform(0.6, 0.4),
    'model__colsample_bytree': uniform(0.5, 0.5),
    'model__reg_alpha': uniform(0.0, 0.1),
    'model__reg_lambda': uniform(0.0, 2.0),
    'model__scale_pos_weight': [(y_train == 0).sum() / (y_train == 1).sum(), 1.0, 0.5, 1.5]
}

# Tune the model (RandomizedSearchCV with stratified CV)
rkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=12,
    scoring="roc_auc",
    cv=rkf,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    return_train_score=False
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

# Evaluate the model on test set
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "pr_auc": average_precision_score(y_test, y_proba)
}

clean_params = {
    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
    for k, v in search.best_params_.items()
}

# Log to MLflow
with mlflow.start_run(run_name="xgb_best"):
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_params(clean_params)
    for k, v in metrics.items():
        mlflow.log_metric(f"test_{k}", float(v))
    mlflow.sklearn.log_model(best_model, artifact_path="model")

print("✅ Parameters, metrics, and model logged successfully to MLflow.")

# Save the best model locally
os.makedirs("tourism_project/model_building", exist_ok=True)
joblib.dump(best_model, "tourism_project/model_building/best_pipeline.joblib")

# Save the best model to Hugging Face Model Hub
api = HfApi()
model_repo = "sahithisaranya/tourismxgb_model"
hf_token = os.environ.get("HF_TOKEN") # Get HF token from environment variable

api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True, token=hf_token)
print(f"Model repo created: {model_repo}")

upload_file(
    path_or_fileobj="tourism_project/model_building/best_pipeline.joblib",
    path_in_repo="best_pipeline.joblib",
    repo_id=model_repo,
    repo_type="model",
    token=hf_token
)

print(f"✅ Model uploaded to Hugging Face Model Hub: {model_repo}")
