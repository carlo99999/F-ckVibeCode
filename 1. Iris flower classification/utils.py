import pickle
import pandera as pa
from pandera.typing import DataFrame
from schemas import IrisFeatures,IrisTarget
import pandas as pd

def load_model(path:str):
    return pickle.load(open(path, "rb"))


@pa.check_types
def make_prediction(model,features_values:DataFrame[IrisFeatures]):
    prediction = model.predict(features_values)
    return prediction
