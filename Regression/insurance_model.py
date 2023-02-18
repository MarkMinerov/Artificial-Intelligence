import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
# insurance_one_hot = pd.get_dummies(insurance)
# insurance_one_hot = insurance_one_hot.drop('sex_male', axis=1)
# insurance_one_hot = insurance_one_hot.drop('smoker_no', axis=1)

# X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)

ct = make_column_transformer(
  (MinMaxScaler(), ["age", "bmi", "children"]),
  (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

X = insurance.drop("charges", axis=1)
Y = insurance["charges"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ct.fit(X_train)

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

print(X_train_normal)

# X_data = insurance_one_hot.drop("charges", axis=1)
# Y_data = insurance_one_hot['charges']

def model_report(y_test, y_pred):
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2: {r2_score(y_test, y_pred)}")

def plot_predictions(test_X, test_Y, pred_Y):
  plt.figure(figsize=(13, 7))
  plt.scatter(test_X, test_Y, c="g", label="Testing data")
  plt.scatter(test_X, pred_Y, c="r", label="Predictions")
  plt.legend()
  plt.show()

# building a model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
history = insurance_model.fit(X_train_normal, Y_train, epochs=100)

pd.DataFrame(history.history).plot()
print(insurance_model.evaluate(X_test_normal, Y_test))

Y_pred = insurance_model.predict(X_test_normal)
plot_predictions(tf.range(0, len(Y_test)), Y_test, tf.squeeze(Y_pred))

model_report(Y_test, Y_pred)