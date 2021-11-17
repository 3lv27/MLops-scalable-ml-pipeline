import json
from pathlib import Path
from ml.data import process_data
from ml.model import inference, compute_model_metrics


def compute_slice_performance(data, model, encoder, lb, cat_features):
    """
    Function that computes performance on model slices
    Computes and saves the performance metrics when the
    value of a given feature is held fixed.
    ------
    data : pandas dataframe
        The preprocessed feature dataframe
    model : ???
        Trained machine learning model.
    encoder : binary
        The encoder used to process data
    lb: binary
        Label used to process data
    cat_features : list of strings
        The column name of the categorical feature used to slice the data
    Returns
    -------
    results : dict
        the model's performance metrics for each slice of the data
    """

    results = {}
    for col in cat_features:
        col_results = {}
        for category in data.loc[:, col].unique():
            data_temp = data[data[col] == category]
            X, y, _, _ = process_data(
                data_temp,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb)
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            col_results[category] = [precision, recall, fbeta]
        results[col] = col_results

    # write output in results dictionary to file
    path = Path.cwd() / "screenshots" / "slice_output.txt"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    return results
