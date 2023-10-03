import numpy as np
from sklearn.datasets import make_regression
import tensorflow as tf
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

X, y = make_regression(n_samples=200, n_features=1, noise=20)
X_new = np.random.choice(X[:, -1], size=10).reshape(-1, 1)

request = PredictRequest()
request.model_spec.name = "reg_model"
request.model_spec.signature_name = "serving_default"
input_name = 'dense_input'
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))


channel = grpc.insecure_channel('localhost:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout=10.0)
