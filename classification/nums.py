import tensorflow as tf
from utils import plot_random_image, plot_confusion_matrix
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_norm = x_train / x_train.max()
x_test_norm = x_test / x_test.max()

y_train_encoded = tf.one_hot(y_train, depth=10)
y_test_encoded = tf.one_hot(y_test, depth=10)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(70, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dense(11, activation="relu"),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax),
])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

numbers_history = model.fit(
  x_train_norm,
  y_train_encoded,
  epochs=40,
  validation_data=(x_test_norm, y_test_encoded),
)

model.evaluate(x_test_norm, y_test_encoded)

y_probs = model.predict(x_test_norm)
y_preds = y_probs.argmax(axis=1)

plot_random_image(model=model, images=x_test, y_true=y_test, classes=tf.range(10))
plt.show()
plot_confusion_matrix(y_test, y_preds, figsize=(20, 20), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.show()

model.save('clothes.h5')