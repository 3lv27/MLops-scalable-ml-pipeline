import pytest
import pandas as pd


@pytest.fixture
def data():
    """ Retrieve Cleaned Dataset """
    train_file = "starter/data/census_clean.csv"
    df = pd.read_csv(train_file)
    # exclude label
    df = df.iloc[:, :-1]
    return df


def test_data_null_values(data):
    """ Check that data has no null value """
    assert (
        data.shape == data.dropna().shape
    ), "Dropping null changes shape of dataframe."


def test_data_char_cleaned(data):
    """ Check that there are no ? characters in the categorical features """
    cat_cols = data.select_dtypes(include=[object]).columns
    for col in cat_cols:
        filtered_char = data[col] == "?"
        assert filtered_char.sum() == 0, f"Found ? character in feature {col}"


def test_data_column_name_cleaned(data):
    """ Check that there are no spaces in the column names """
    cols_name = data.columns
    for col in cols_name:
        assert " " not in col, f"Found space character in feature {col}"
