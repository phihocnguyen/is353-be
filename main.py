from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import pandas
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
df = pandas.read_csv("input_time_series_data.csv")
x = df.iloc[:, 1:25].values

origins = [
    "https://is353-fe.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = tf.keras.models.load_model("model.keras")

class Input(BaseModel):
    # Assuming the model expects input as a list of floats (adjust this as needed)
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "API works successfully!"}
@app.post('/predict')
def predict(input : Input):
    scaler = StandardScaler().fit(x)
    features_array = np.array(input.features).reshape(1, -1)
    new_data_scaled = scaler.transform(features_array)
    new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
    prediction = model.predict(new_data_scaled)
    return {
        "prediction": prediction.tolist()
    }
