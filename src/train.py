import json
import os
import joblib
import mlflow
import pandas as pd
from src.schema import IrisSchema
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    os.makedirs("models", exist_ok=True)
    raw_df = pd.read_csv("data/data.csv")
    df = IrisSchema.validate(raw_df)

    X = df.drop(["target", "id"], axis=1, errors="ignore")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state=42, stratify=y
    )

    params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)

    cat_col = ["species"]
    preprocessing = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(drop="first"), cat_col)
        ],
        remainder="passthrough"  
    )

    pipeline = Pipeline(
        steps=[
            ("ohe", preprocessing),
            ("classifier", RandomForestClassifier(**params))
        ]
    )

    mlflow.set_experiment("mlops-demo")
    with mlflow.start_run(nested=True):
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
            "sample_rows": int(len(df)),
        }
        mlflow.log_metrics(metrics)
        print(f"Training complete. Metrics: {metrics}")

        # 5. Save Model & Metrics Artifact
        joblib.dump(pipeline, "models/model.joblib")
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
if __name__ == "__main__":
    main()
