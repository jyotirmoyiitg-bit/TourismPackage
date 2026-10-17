"""Model training: tune XGBoost with GridSearchCV, track with MLflow, save the best model."""
import json
import os
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric = ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
           "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
           "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric),
    (OneHotEncoder(handle_unknown="ignore"), categorical),
)
# Handle class imbalance (~19% buyers)
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42,
                              eval_metric="logloss", n_jobs=-1)
pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1, 0.2],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [1, 5],
}

with mlflow.start_run(run_name="xgboost-grid-search"):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # Log every tuned parameter combination as a nested run
    results = grid.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params({k.split("__")[1]: v for k, v in params.items()})
            mlflow.log_metric("mean_cv_recall", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_recall", results["std_test_score"][i])

    best = grid.best_estimator_
    mlflow.log_params({f"best_{k.split('__')[1]}": v for k, v in grid.best_params_.items()})

    # Evaluate the best model; a 0.45 threshold is used to balance recall and precision
    threshold = 0.45
    metrics = {}
    for name, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        proba = best.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        metrics.update({
            f"{name}_accuracy": accuracy_score(y, pred),
            f"{name}_precision": precision_score(y, pred),
            f"{name}_recall": recall_score(y, pred),
            f"{name}_f1": f1_score(y, pred),
            f"{name}_roc_auc": roc_auc_score(y, proba),
        })
        if name == "test":
            print(classification_report(y, pred, target_names=["No purchase", "Purchase"]))
    mlflow.log_metrics(metrics)
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))
    print("Best params:", grid.best_params_)

    # Save the model (with decision threshold) into the deployment folder
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model.joblib"
    joblib.dump({"model": best, "threshold": threshold}, model_path)
    mlflow.log_artifact(model_path)
    print(f"Model saved to {model_path}")
