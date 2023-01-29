from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import plt_decision_boundary, compare_classification_sets

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03)

circles = pd.DataFrame({"X0":X[:, 0], "X1": X[:, 1], "label": y})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build a model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# activation="sigmoid" for binary classification
# activation="softmax" for multiclass classification

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=50, verbose=0)

pd.DataFrame(history.history).plot()
plt.title("Model loss curves")
plt.show()
plt_decision_boundary(model, X, y)
plt.show()

compare_classification_sets(model, X_train, y_train, X_test, y_test)
plt.show()