import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/adult.csv"
MODEL_PATH = "models/income_models.joblib"

TARGET = "income"

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = [
        c.strip().lower().replace(" ", "-").replace(".", "-")
        for c in df.columns
    ]

    # Replace missing-value marker with NumPy NaN
    df = df.replace("?", np.nan)

    # Clean text columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
    	df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Remove trailing period from income labels if present
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].str.replace(".", "", regex=False).str.strip()

    missing_features = [c for c in FEATURES + [TARGET] if c not in df.columns]

    if missing_features:
        raise ValueError(f"Missing columns: {missing_features}")

    # Convert numerical columns
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows without target
    df = df.dropna(subset=[TARGET]).copy()

    # Convert income to 0/1
    df[TARGET] = df[TARGET].map({
        "<=50K": 0,
        ">50K": 1
    })

    if df[TARGET].isna().any():
        raise ValueError("Unexpected income labels found in the dataset.")

    return df


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the Kaggle Adult Census Income CSV "
            "and save it as data/adult.csv."
        )

    df = load_and_clean_data(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logistic_model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    knn_model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])

    models = {
        "Logistic Regression": logistic_model,
        "KNN": knn_model
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm.tolist()
        }

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print(classification_report(
            y_test, predictions,
            target_names=["<=50K", ">50K"],
            zero_division=0
        ))

    bundle = {
        "models": models,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "results": results
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained models to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
