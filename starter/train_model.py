# Script to train machine learning model.
import logging

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from .ml.data import process_data
from .ml.model import train_model, inference, compute_model_metrics, save_model

# Add code to load in the data.
try:
    data = pd.read_csv(r"data/census_clean.csv")
except FileNotFoundError:
    logging.error("data/census_clean.csv NOT found")
    raise FileNotFoundError

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

save_model(encoder, "encoder.pkl")
save_model(lb, "lb.pkl")

# Train and save a model.
classifier = train_model(X_train, y_train)
save_model(classifier, "hgb_classifier.pkl")


y_train_pred = inference(classifier, X_train)
train_precision, train_recall, train_fbeta = compute_model_metrics(
    y_train, y_train_pred)

y_test_pred = inference(classifier, X_test)
test_precision, test_recall, test_fbeta = compute_model_metrics(
    y_test, y_test_pred)
