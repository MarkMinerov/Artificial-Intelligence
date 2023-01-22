import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def plot_predictions(test_X, test_Y, pred_Y):
  plt.figure(figsize=(13, 7))
  plt.scatter(test_X, test_Y, c="g", label="Testing data")
  plt.scatter(test_X, pred_Y, c="r", label="Predictions")
  plt.legend()
  plt.show()

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.boston_housing.load_data(
  path='boston_housing.npz', test_split=0.2
)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), metrics=["mae"])
history = model.fit(X_train, Y_train, epochs=250)
print(model.evaluate(X_test, Y_test))

pd.DataFrame(history.history).plot()
plt.show()

Y_pred = model.predict(X_test)
plot_predictions(tf.range(0, len(Y_test)), Y_test, Y_pred)