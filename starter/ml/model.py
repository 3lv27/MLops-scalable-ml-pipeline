import logging
import pickle
from pathlib import Path
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = HistGradientBoostingClassifier(random_state=46)
    parameters = {
        "n_estimators": (5, 10),
        "learning_rate": (0.1, 0.01, 0.001),
        "max_depth": [None, 1, 3, 5, 10, 20],
        "max_features": ("auto", "log2")
    }
    grid = GridSearchCV(model, parameters)
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, filepath, filename):
    path = Path(filepath)

    if path.exists():
        with open(f"{filepath}/{filename}", "wb") as file:
            pickle.dump(model, file)
        logging.info("model saved to: ", filepath)
        return True
    else:
        logging.error(f"path: {filepath} does not exists")
        return False
