# https://figshare.com/articles/dataset/PAPILA/14798004/1?file=28454352
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from papila_dataset import load_classification_matrix


def train_and_evaluate(root_dir: str | Path, test_size: float = 0.2, random_state: int = 42):
    root = Path(root_dir)
    X, y, df = load_classification_matrix(root, target_size=(64, 64))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = make_pipeline(
        PCA(n_components=200, random_state=random_state, whiten=True),
        LogisticRegression(
            random_state=random_state,
            max_iter=20000,
            class_weight="balanced",
        ),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    report = classification_report(y_test, predictions, target_names=["normal", "suspect", "glaucoma"])
    cm = confusion_matrix(y_test, predictions)

    return {
        "model": model,
        "accuracy": accuracy,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": predictions,
        "dataset": df,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = train_and_evaluate(root)

    print("Accuracy:", round(result["accuracy"], 4))
    print("Weighted F1:", round(result["f1_weighted"], 4))
    print("\nClassification report:\n")
    print(result["classification_report"])
    print("\nConfusion matrix:\n")
    print(pd.DataFrame(result["confusion_matrix"], columns=["normal", "suspect", "glaucoma"]))
