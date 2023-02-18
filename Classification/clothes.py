# Different items of clothing
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import pandas as pd
from utils import plot_confusion_matrix, plot_random_image
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(50, activation="relu"),
  tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax),
])

X_train_norm = X_train / X_train.max()
X_test_norm = X_test / X_test.max()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

# tf.keras.losses.CategoricalCrossentropy() when one-hot encoded
# tf.keras.losses.SparseCategoricalCrossentropy() when not one-hot encoded

norm_history = model.fit(X_train_norm, tf.one_hot(y_train, depth=10), epochs=10, validation_data=(X_test_norm, tf.one_hot(y_test, depth=10)))

pd.DataFrame(norm_history.history).plot(title="Normalized data")
plt.show()

y_probs = model.predict(X_test_norm)
y_preds = y_probs.argmax(axis=1)

plot_confusion_matrix(y_test, y_preds, figsize=(20, 20), classes=class_names)
plt.show()

plot_random_image(model, X_test_norm, y_test, class_names)
plt.show()