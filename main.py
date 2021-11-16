"""
ML pipeline inference with FastAPI
"""
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.train_model import cat_features
from starter.ml.model import inference
from starter.ml.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull train_model") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI(title="FastAPI Census Prediction 🤖",
              description="Ask your Census thoughts to our API")


class CensusRequestModel(BaseModel):
    """Census prediction data model."""
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


class CensusResponseModel(BaseModel):
    """Census prediction response data model."""
    prediction: str


def load_artifacts():
    # load model artifacts needed for processing and inference
    model = pd.load_pkl("model/hgb_classifier.pkl")
    encoder = pd.load_pkl("model/encoder.pkl")
    lb = pd.load_pkl("model/lb.pkl")
    return model, encoder, lb


@app.get("/")
async def welcome():
    return {"Greeting": "Welcome to the FastAPI Census Prediction 🤖"}


@app.post("/predict", response_model=CensusResponseModel, status_code=200)
async def predict(data: CensusRequestModel):
    # ingest Features into dataframe for processing
    df = pd.DataFrame([{"age": data.age,
                        "workclass": data.workclass,
                        "fnlgt": data.fnlgt,
                        "education": data.education,
                        "education-num": data.education_num,
                        "marital-status": data.marital_status,
                        "occupation": data.occupation,
                        "relationship": data.relationship,
                        "race": data.race,
                        "sex": data.sex,
                        "capital-gain": data.capital_gain,
                        "capital-loss": data.capital_loss,
                        "hours-per-week": data.hours_per_week,
                        "native-country": data.native_country}])

    model, encoder, lb = load_artifacts()

    # process the data to get it into the correct format for inference
    X, _, _, _ = process_data(df,
                              categorical_features=cat_features,
                              training=False,
                              encoder=encoder,
                              lb=lb)

    # generate predictions
    preds = inference(model, X)

    # convert to human readable output
    if preds == 0:
        prediction_text = "Salary <= 50K"
    else:
        prediction_text = "Salary > 50K"

    # create response to return
    prediction_response = {"prediction": prediction_text}

    return prediction_response
