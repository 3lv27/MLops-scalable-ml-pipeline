import argparse
import requests


def prediction_request(base_url: str, data: dict) -> dict:
    res = requests.post(f"{base_url}/predict", json=data)
    print(res.json())
    return res.json()


if __name__ == "__main__":

    BASE_URL = "https://census-prediction-app-mlops.herokuapp.com"

    below_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    above_data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"}

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--example_type",
        required=True,
        type=str,
        choices=["above", "below"],
        help="The type of sample data you want to send to the api",
    )

    args = parser.parse_args()

    if args.example_type == "above":
        data = above_data
    else:
        data = below_data

    response = prediction_request(BASE_URL, data)
