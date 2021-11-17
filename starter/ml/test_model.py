import pytest
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting \
    import HistGradientBoostingClassifier

from .data import process_data
from .model import compute_model_metrics, inference


@pytest.fixture(scope="session")
def data():
    """ Retrieve Cleaned Dataset """
    path = Path.cwd() / "data" / "census_clean.csv"
    df = pd.read_csv(path)
    return df


@pytest.fixture
def model(processed_data):
    X_train, y_train, _, _ = processed_data
    path = Path.cwd() / "model" / "hgb_classifier.pkl"
    model = pd.read_pickle(path)
    return model


@pytest.fixture
def processed_data(data):
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    path_encoder = Path.cwd() / "model" / "encoder.pkl"
    encoder = pd.read_pickle(path_encoder)

    path_lb = Path.cwd() / "model" / "lb.pkl"
    lb = pd.read_pickle(path_lb)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    return X_train, y_train, X_test, y_test


def test_model_type(model):
    assert isinstance(model, HistGradientBoostingClassifier)


def test_compute_model_metrics(processed_data, model):
    X_train, y_train, X_test, y_test = processed_data

    y_train_pred = inference(model, X_train)
    y_test_pred = inference(model, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(
        y_train, y_train_pred)
    precision_test, recall_test, fbeta_test = compute_model_metrics(
        y_test, y_test_pred)

    assert isinstance(precision_train, float)
    assert isinstance(precision_test, float)
    assert isinstance(recall_train, float)
    assert isinstance(recall_test, float)
    assert isinstance(fbeta_train, float)
    assert isinstance(fbeta_test, float)

    assert (precision_train <= 1) & (precision_train >= 0)
    assert (precision_test <= 1) & (precision_test >= 0)
    assert (recall_train <= 1) & (recall_train >= 0)
    assert (recall_test <= 1) & (recall_test >= 0)
    assert (fbeta_train <= 1) & (fbeta_train >= 0)
    assert (fbeta_test <= 1) & (fbeta_test >= 0)


def test_inference(processed_data, model):
    X_train, y_train, X_test, y_test = processed_data

    y_train_pred = inference(model, X_train)
    assert len(y_train_pred) == X_train.shape[0]
    assert len(y_train_pred) > 0

    y_test_pred = inference(model, X_test)
    assert len(y_test_pred) == X_test.shape[0]
    assert len(y_test_pred) > 0
