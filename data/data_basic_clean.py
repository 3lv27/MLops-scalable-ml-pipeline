import logging

import pandas as pd
from pathlib import Path
from pandas import DataFrame


def load_data_csv(path: str) -> DataFrame:
    return pd.read_csv(path)


def clean_cols_name(df: DataFrame) -> DataFrame:
    clean_df = df.copy()
    clean_df.columns = df.columns.str.replace(" ", "")
    return clean_df


def clean_categorical_cols(df: DataFrame) -> DataFrame:
    clean_df = df.copy()
    cat_cols = df.select_dtypes(include=[object]).columns
    for col in cat_cols:
        clean_df[col] = df[col].str.strip()
    return clean_df


def clean_rows_with_false_condition(df: DataFrame, cols: list, condition: str) -> DataFrame:
    clean_df = df.copy()
    for col in cols:
        clean_df = df[df[col] != condition]
    return clean_df


def save_data_csv(df: DataFrame, folder: str, file_name: str) -> bool:
    path = Path(folder)
    if path.exists():
        df.to_csv(f"{folder}/{file_name}", index=False)
        return True
    else:
        logging.error(f"path: {folder} does not exists")
        return False


if __name__ == "__main__":
    census_df = clean_categorical_cols(clean_cols_name(load_data_csv("./census.csv")))
    cleaned_census_df = clean_rows_with_false_condition(census_df, ["workclass", "occupation", "native-country"], "?")
    save_data_csv(cleaned_census_df, ".", "census_clean.csv")
