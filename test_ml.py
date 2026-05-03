from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def processed_data():
    data_path = Path(__file__).resolve().parent / "data" / "census.csv"
    data = pd.read_csv(data_path).sample(n=1000, random_state=42)
    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["salary"],
    )
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, encoder, lb


def test_process_data_returns_arrays(processed_data):
    """Verify preprocessing returns aligned arrays and a fitted encoder."""
    X_train, y_train, X_test, y_test, encoder, lb = processed_data

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert len(encoder.categories_) == len(CAT_FEATURES)
    assert set(lb.classes_) == {"<=50K", ">50K"}


def test_train_model_returns_random_forest(processed_data):
    """Verify the training helper fits the expected classifier."""
    X_train, y_train, _, _, _, _ = processed_data

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_inference_returns_binary_predictions(processed_data):
    """Verify inference returns one prediction for every input row."""
    X_train, y_train, X_test, _, _, _ = processed_data
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)

    assert preds.shape[0] == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_known_values():
    """Verify metric calculations on a small deterministic example."""
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)
