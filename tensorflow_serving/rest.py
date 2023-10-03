import numpy as np
from sklearn.datasets import make_regression
import json
import requests

SERVER_URL = 'http://localhost:8501/v1/models/reg_model:predict'

X, y = make_regression(n_samples=200, n_features=1, noise=20)
X_new = np.random.choice(X[:, -1], size=10).reshape(-1, 1)

input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": X_new.tolist(),
})

print(input_data_json)

response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status()  # raise an exception in case of error
response = response.json()

y_proba = np.array(response["predictions"])
y_proba.round(2)

print(y_proba)
